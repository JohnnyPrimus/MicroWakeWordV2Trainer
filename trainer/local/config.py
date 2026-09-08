"""Config loading for the local microWakeWord training pipeline.

All pipeline steps (setup_env, generate_samples, download_datasets,
build_features, train, export) read their settings from a single YAML
file described by ``TrainerConfig``. See ../wakeword_config.example.yaml
for the documented default values.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclasses.dataclass
class WakeWordConfig:
    phonetic: str
    friendly_name: str
    # Extra phonetic spellings of the same wake word. Each gets its own full
    # batch of piper.max_samples generated samples (see generate_samples.py),
    # accumulating alongside the primary `phonetic` spelling's samples -
    # more phonetic variety generally improves model robustness.
    additional_phonetics: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class GpuConfig:
    # "cpu", "cuda" (NVIDIA), or "rocm" (AMD). For "rocm", setup installs the
    # tensorflow-rocm wheels from AMD's index (they only exist for Python
    # 3.10 and 3.12, so create the venv with one of those). For "cuda", setup
    # installs "tensorflow[and-cuda]" (needs a CUDA-capable Python <= 3.12
    # on Linux/WSL).
    backend: str = "cpu"
    # ROCm release line used to pick AMD's wheel index
    # (https://repo.radeon.com/rocm/manylinux/rocm-rel-<rocm_release>/).
    # Must match your installed ROCm: 6.4.x -> 2.18.1 max, 7.0.x -> 2.19.1
    # max, 7.1.x/7.2.x -> 2.20.0/2.19.1/2.18.1 (see rocm.docs.amd.com
    # TensorFlow compatibility page).
    rocm_release: str = "7.2"
    # tensorflow-rocm version to install (must exist in that index).
    tensorflow_rocm_version: str = "2.19.1"
    # AMD's tensorflow-rocm wheels only ship GPU kernels for the GPU targets
    # AMD built them for (e.g. gfx1100/gfx1101/gfx1102 for RDNA3), which does
    # NOT include the newer RDNA 3.5 APUs (gfx1150 "Strix" / gfx1151 "Strix
    # Halo", Radeon 780M/880M/890M/8060S) - without this, TF crashes with a
    # hip_code_object.cpp assertion when it tries to load a kernel. Setting
    # this exports HSA_OVERRIDE_GFX_VERSION for setup/training so the GPU
    # pretends to be a supported target. Use "11.0.0" for gfx1150/gfx1151.
    # Leave empty for GPUs ROCm natively supports (overriding those can break
    # things).
    hsa_override_gfx_version: str = ""


@dataclasses.dataclass
class PiperConfig:
    model_url: str = "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"
    max_samples: int = 1000
    batch_size: int = 100
    max_speakers: int = 100
    noise_scales: List[float] = dataclasses.field(default_factory=lambda: [0.667])
    noise_scale_ws: List[float] = dataclasses.field(default_factory=lambda: [0.667])
    length_scales: List[float] = dataclasses.field(default_factory=lambda: [0.85, 1.0, 1.15])


@dataclasses.dataclass
class DatasetsConfig:
    audioset_split: str = "bal_train"
    download_processes: int = 3
    download_retries: int = 3
    hf_token_env: str = "HF_TOKEN"


@dataclasses.dataclass
class AugmentationConfig:
    duration_s: float = 3.2
    probabilities: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        }
    )
    background_min_snr_db: float = -5
    background_max_snr_db: float = 10
    min_jitter_s: float = 0.195
    max_jitter_s: float = 0.205


@dataclasses.dataclass
class TrainingConfig:
    training_steps: List[int] = dataclasses.field(default_factory=lambda: [10000])
    positive_class_weight: List[float] = dataclasses.field(default_factory=lambda: [1])
    negative_class_weight: List[float] = dataclasses.field(default_factory=lambda: [20])
    learning_rates: List[float] = dataclasses.field(default_factory=lambda: [0.001])
    batch_size: int = 128
    eval_step_interval: int = 500
    clip_duration_ms: int = 1500
    target_minimization: float = 0.9
    minimization_metric: Optional[str] = None
    maximization_metric: str = "average_viable_recall"
    time_mask_max_size: List[int] = dataclasses.field(default_factory=lambda: [0])
    time_mask_count: List[int] = dataclasses.field(default_factory=lambda: [0])
    freq_mask_max_size: List[int] = dataclasses.field(default_factory=lambda: [0])
    freq_mask_count: List[int] = dataclasses.field(default_factory=lambda: [0])
    model: str = "mixednet"
    model_args: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "pointwise_filters": "64,64,64,64",
            "repeat_in_block": "1, 1, 1, 1",
            "mixconv_kernel_sizes": "[5], [7,11], [9,15], [23]",
            "residual_connection": "0,0,0,0",
            "first_conv_filters": 32,
            "first_conv_kernel_size": 5,
            "stride": 3,
        }
    )


@dataclasses.dataclass
class TrainerConfig:
    wake_word: WakeWordConfig
    workspace: Path
    gpu: GpuConfig = dataclasses.field(default_factory=GpuConfig)
    piper: PiperConfig = dataclasses.field(default_factory=PiperConfig)
    datasets: DatasetsConfig = dataclasses.field(default_factory=DatasetsConfig)
    augmentation: AugmentationConfig = dataclasses.field(default_factory=AugmentationConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    # --- derived paths, all rooted under `workspace` -----------------
    @property
    def piper_repo_dir(self) -> Path:
        return self.workspace / "piper-sample-generator"

    @property
    def piper_model_path(self) -> Path:
        # Must live in piper-sample-generator's own models/ dir: the repo
        # tracks a matching en_US-libritts_r-medium.pt.json config file
        # there (too small for the release download), which the .pt
        # checkpoint needs sitting next to it at load time.
        return self.piper_repo_dir / "models" / "en_US-libritts_r-medium.pt"

    @property
    def mww_repo_dir(self) -> Path:
        return self.workspace / "microWakeWord"

    @property
    def samples_dir(self) -> Path:
        return self.workspace / "generated_samples" / self.wake_word.friendly_name

    @property
    def preview_dir(self) -> Path:
        # Deliberately outside samples_dir's tree: build_features.py globs
        # samples_dir recursively for training data, and preview clips
        # shouldn't leak into that.
        return self.workspace / "preview_samples" / self.wake_word.friendly_name

    @property
    def rir_dir(self) -> Path:
        return self.workspace / "mit_rirs"

    @property
    def audioset_dir(self) -> Path:
        return self.workspace / "audioset_16k"

    @property
    def fma_dir(self) -> Path:
        return self.workspace / "fma_16k"

    @property
    def negative_datasets_dir(self) -> Path:
        return self.workspace / "negative_datasets"

    @property
    def features_dir(self) -> Path:
        return self.workspace / "generated_augmented_features" / self.wake_word.friendly_name

    @property
    def train_dir(self) -> Path:
        return self.workspace / "trained_models" / self.wake_word.friendly_name

    @property
    def export_dir(self) -> Path:
        return self.workspace / "exported_models" / self.wake_word.friendly_name

    @property
    def training_parameters_yaml(self) -> Path:
        return self.workspace / f"training_parameters_{self.wake_word.friendly_name}.yaml"


def _dataclass_from_dict(cls, data: Optional[Dict[str, Any]]):
    if not data:
        return cls()
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> TrainerConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "wake_word" not in raw:
        raise ValueError(f"{path} is missing the required 'wake_word' section")
    wake_word = _dataclass_from_dict(WakeWordConfig, raw["wake_word"])

    workspace = Path(raw.get("workspace", "workspace"))
    if not workspace.is_absolute():
        # Resolve relative to the config file's directory, not the CWD,
        # so the pipeline behaves the same regardless of where it's invoked from.
        workspace = (path.parent / workspace).resolve()

    gpu = _dataclass_from_dict(GpuConfig, raw.get("gpu"))
    if gpu.backend not in ("cpu", "cuda", "rocm"):
        raise ValueError(f"gpu.backend must be 'cpu', 'cuda', or 'rocm', got {gpu.backend!r}")

    return TrainerConfig(
        wake_word=wake_word,
        workspace=workspace,
        gpu=gpu,
        piper=_dataclass_from_dict(PiperConfig, raw.get("piper")),
        datasets=_dataclass_from_dict(DatasetsConfig, raw.get("datasets")),
        augmentation=_dataclass_from_dict(AugmentationConfig, raw.get("augmentation")),
        training=_dataclass_from_dict(TrainingConfig, raw.get("training")),
    )
