"""Single-job process manager for the web control panel."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Optional


VALID_ACTIONS = {
    "setup",
    "preview-word",
    "generate-samples",
    "download-data",
    "preview-augment",
    "build-features",
    "train",
    "export",
    "all",
}


class JobAlreadyRunning(RuntimeError):
    """Raised when a second pipeline job is started concurrently."""


@dataclass
class JobRecord:
    id: str
    action: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    return_code: Optional[int] = None
    log_file: str = ""
    cancel_requested: bool = False
    error: Optional[str] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobManager:
    """Run one CLI job at a time and retain its recent output in memory."""

    def __init__(self, app_root: Path) -> None:
        self.app_root = app_root
        self._lock = threading.RLock()
        self._record: Optional[JobRecord] = None
        self._process: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None
        self._log_lines: Deque[str] = deque(maxlen=5000)

    def start(
        self,
        action: str,
        config_path: Path,
        workspace: Path,
        options: Optional[dict[str, Any]] = None,
    ) -> JobRecord:
        if action not in VALID_ACTIONS:
            raise ValueError(f"Unsupported action: {action}")

        options = options or {}
        workspace.mkdir(parents=True, exist_ok=True)
        log_dir = workspace / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        command = self._build_command(action, config_path, options)
        job_id = uuid.uuid4().hex[:12]
        log_path = log_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{action}-{job_id}.log"

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(self.app_root) + os.pathsep + env.get("PYTHONPATH", "")

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise JobAlreadyRunning("A pipeline job is already running")

            self._log_lines.clear()
            record = JobRecord(
                id=job_id,
                action=action,
                status="running",
                started_at=_utc_now(),
                log_file=str(log_path),
            )

            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(workspace),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            except Exception as exc:
                record.status = "failed"
                record.finished_at = _utc_now()
                record.error = str(exc)
                self._record = record
                raise

            self._record = record
            self._process = process
            thread = threading.Thread(
                target=self._run_job,
                args=(record, process, log_path),
                name=f"trainer-job-{job_id}",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return JobRecord(**record.__dict__)

    def _build_command(
        self,
        action: str,
        config_path: Path,
        options: dict[str, Any],
    ) -> list[str]:
        command = [
            sys.executable,
            "-u",
            "-m",
            "trainer.local.cli",
            action,
            "--config",
            str(config_path),
        ]

        if action == "train" and options.get("no_train"):
            command.append("--no-train")
        elif action == "export":
            command.extend(
                [
                    "--probability-cutoff",
                    str(options.get("probability_cutoff", 0.5)),
                    "--tensor-arena-size",
                    str(options.get("tensor_arena_size", 30000)),
                    "--author",
                    str(options.get("author", "")),
                    "--website",
                    str(options.get("website", "")),
                ]
            )
        return command

    def _run_job(
        self,
        record: JobRecord,
        process: subprocess.Popen[str],
        log_path: Path,
    ) -> None:
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"$ {' '.join(process.args)}\n\n")
                log_file.flush()
                if process.stdout is not None:
                    for line in process.stdout:
                        log_file.write(line)
                        log_file.flush()
                        with self._lock:
                            self._log_lines.append(line)
            return_code = process.wait()
        except Exception as exc:
            return_code = -1
            with self._lock:
                record.error = str(exc)
                self._log_lines.append(f"\n[web] Job monitor failed: {exc}\n")

        with self._lock:
            record.return_code = return_code
            record.finished_at = _utc_now()
            if record.cancel_requested:
                record.status = "cancelled"
            elif return_code == 0:
                record.status = "succeeded"
            else:
                record.status = "failed"
                if record.error is None:
                    record.error = f"Process exited with code {return_code}"
            self._process = None

    def snapshot(self) -> Optional[dict[str, Any]]:
        with self._lock:
            if self._record is None:
                return None
            return {
                **self._record.__dict__,
                "log": "".join(self._log_lines),
            }

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def cancel(self) -> bool:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return False
            if self._record is not None:
                self._record.cancel_requested = True

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except Exception:
            process.terminate()
        return True

    def shutdown(self) -> None:
        self.cancel()
        with self._lock:
            thread = self._thread
            process = self._process
        if process is not None:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
        if thread is not None:
            thread.join(timeout=3)
