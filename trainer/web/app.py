"""FastAPI control panel for the local microWakeWord training pipeline."""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trainer.local.config import TrainerConfig, load_config
from trainer.local.export import TFLITE_SOURCE_RELATIVE
from trainer.local.setup_env import missing_runtime_modules
from trainer.web.jobs import (
    JobAlreadyRunning,
    JobManager,
    VALID_ACTIONS,
)


APP_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
EXAMPLE_CONFIG = APP_ROOT / "trainer" / "wakeword_config.example.yaml"
CONFIG_PATH = Path(
    os.environ.get("TRAINER_CONFIG", str(APP_ROOT / "trainer" / "wakeword_config.yaml"))
).resolve()

job_manager = JobManager(APP_ROOT)


def _ensure_config() -> None:
    if CONFIG_PATH.exists():
        return
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = EXAMPLE_CONFIG.read_text(encoding="utf-8")
    workspace_override = os.environ.get("TRAINER_WORKSPACE")
    if workspace_override:
        content = content.replace(
            'workspace: "workspace"',
            f'workspace: "{workspace_override}"',
            1,
        )
    CONFIG_PATH.write_text(content, encoding="utf-8")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _ensure_config()
    yield
    job_manager.shutdown()


app = FastAPI(
    title="Wakeword Studio",
    description="Web control panel for the microWakeWord V2 training pipeline",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ConfigPayload(BaseModel):
    content: str = Field(min_length=1, max_length=262_144)


class StartJobPayload(BaseModel):
    action: str
    probability_cutoff: float = Field(default=0.5, ge=0, le=1)
    tensor_arena_size: int = Field(default=30_000, ge=1_000, le=10_000_000)
    author: str = Field(default="", max_length=200)
    website: str = Field(default="", max_length=500)
    no_train: bool = False


def _load_runtime_config() -> TrainerConfig:
    try:
        return load_config(CONFIG_PATH)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid training config: {exc}") from exc


def _has_files(path: Path, pattern: str = "*.wav") -> bool:
    return path.exists() and next(path.rglob(pattern), None) is not None


def _feature_split_done(config: TrainerConfig, split: str) -> bool:
    return (config.features_dir / split / "wakeword_mmap" / "data.ninja").exists()


def _python_dependencies_installed() -> bool:
    return not missing_runtime_modules()


def _preview_file(config: TrainerConfig) -> Optional[Path]:
    if not config.preview_dir.exists():
        return None
    return next(iter(sorted(config.preview_dir.rglob("*.wav"))), None)


def _artifact_url(path: Path, kind: str) -> dict[str, Any]:
    try:
        version = path.stat().st_mtime_ns
    except OSError:
        version = 0
    return {
        "name": path.name,
        "url": f"/api/artifacts/{kind}?v={version}",
        "size": path.stat().st_size if path.exists() else 0,
    }


def _find_export_file(config: TrainerConfig, suffix: str) -> Optional[Path]:
    if not config.export_dir.exists():
        return None
    return next(iter(sorted(config.export_dir.glob(f"*{suffix}"))), None)


def _step_status(config: TrainerConfig) -> dict[str, dict[str, Any]]:
    sample_files = list(config.samples_dir.rglob("*.wav")) if config.samples_dir.exists() else []
    negative_names = ("speech", "dinner_party", "no_speech", "dinner_party_eval")
    download_done = (
        all((config.negative_datasets_dir / name).exists() for name in negative_names)
        and _has_files(config.rir_dir)
        and _has_files(config.audioset_dir)
        and _has_files(config.fma_dir)
    )
    setup_done = (
        config.mww_repo_dir.exists()
        and config.piper_repo_dir.exists()
        and config.piper_model_path.exists()
        and _python_dependencies_installed()
    )
    features_done = all(
        _feature_split_done(config, split)
        for split in ("training", "validation", "testing")
    )
    model_path = config.train_dir / TFLITE_SOURCE_RELATIVE
    export_model = _find_export_file(config, ".tflite")

    return {
        "setup": {"done": setup_done},
        "preview-word": {"done": _preview_file(config) is not None},
        "generate-samples": {
            "done": bool(sample_files),
            "sample_count": len(sample_files),
        },
        "download-data": {"done": download_done},
        "preview-augment": {
            "done": (config.workspace / "augmented_clip_preview.wav").exists()
        },
        "build-features": {"done": features_done},
        "train": {"done": model_path.exists()},
        "export": {"done": export_model is not None},
    }


def _artifact_path(config: TrainerConfig, kind: str) -> Optional[Path]:
    if kind == "word-preview":
        return _preview_file(config)
    if kind == "augmentation-preview":
        path = config.workspace / "augmented_clip_preview.wav"
        return path if path.exists() else None
    if kind == "model":
        return _find_export_file(config, ".tflite")
    if kind == "manifest":
        return _find_export_file(config, ".json")
    if kind == "roc":
        path = config.train_dir / "tflite_stream_state_internal_quant" / "tflite_streaming_roc.txt"
        return path if path.exists() else None
    return None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict[str, str]:
    _ensure_config()
    try:
        content = CONFIG_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read config: {exc}") from exc
    return {"path": str(CONFIG_PATH), "content": content}


@app.put("/api/config")
def put_config(payload: ConfigPayload) -> dict[str, Any]:
    if job_manager.is_running():
        raise HTTPException(status_code=409, detail="Stop the running job before editing config")

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=CONFIG_PATH.parent,
            prefix=".wakeword-config-",
            suffix=".yaml",
            delete=False,
        ) as temp:
            temp.write(payload.content)
            temp_path = Path(temp.name)
        load_config(temp_path)
        os.replace(temp_path, CONFIG_PATH)
        temp_path = None
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=422, detail=f"Config was not saved: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return {"saved": True, "path": str(CONFIG_PATH)}


@app.get("/api/status")
def status() -> dict[str, Any]:
    _ensure_config()
    job = job_manager.snapshot()
    try:
        config = load_config(CONFIG_PATH)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return {
            "config_error": str(exc),
            "config_path": str(CONFIG_PATH),
            "job": job,
        }

    try:
        disk_free = shutil.disk_usage(config.workspace.parent).free
    except OSError:
        disk_free = 0

    artifacts: dict[str, Any] = {}
    for kind in ("word-preview", "augmentation-preview", "model", "manifest", "roc"):
        path = _artifact_path(config, kind)
        if path is not None:
            artifacts[kind] = _artifact_url(path, kind)

    return {
        "config_path": str(CONFIG_PATH),
        "workspace": str(config.workspace),
        "wake_word": {
            "phonetic": config.wake_word.phonetic,
            "friendly_name": config.wake_word.friendly_name,
            "additional_phonetics": config.wake_word.additional_phonetics,
        },
        "steps": _step_status(config),
        "job": job,
        "artifacts": artifacts,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "mode": "CPU",
        },
        "disk_free_gb": round(disk_free / (1024**3), 1),
    }


@app.post("/api/jobs")
def start_job(payload: StartJobPayload) -> dict[str, Any]:
    if payload.action not in VALID_ACTIONS:
        raise HTTPException(status_code=422, detail=f"Unsupported action: {payload.action}")
    config = _load_runtime_config()
    options = {
        "probability_cutoff": payload.probability_cutoff,
        "tensor_arena_size": payload.tensor_arena_size,
        "author": payload.author,
        "website": payload.website,
        "no_train": payload.no_train,
    }
    try:
        record = job_manager.start(
            action=payload.action,
            config_path=CONFIG_PATH,
            workspace=config.workspace,
            options=options,
        )
    except JobAlreadyRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not start job: {exc}") from exc
    return {"job": record.__dict__}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    job = job_manager.snapshot()
    if job is None or job["id"] != job_id:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job_manager.cancel():
        raise HTTPException(status_code=409, detail="Job is no longer running")
    return {"cancelling": True}


@app.get("/api/artifacts/{kind}")
def artifact(kind: Literal["word-preview", "augmentation-preview", "model", "manifest", "roc"]):
    config = _load_runtime_config()
    path = _artifact_path(config, kind)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact is not available yet")

    media_types = {
        "word-preview": "audio/wav",
        "augmentation-preview": "audio/wav",
        "model": "application/octet-stream",
        "manifest": "application/json",
        "roc": "text/plain; charset=utf-8",
    }
    download = kind in {"model", "manifest", "roc"}
    return FileResponse(
        path,
        media_type=media_types[kind],
        filename=path.name if download else None,
    )
