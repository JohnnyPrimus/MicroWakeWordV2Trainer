"""Step: set up the local environment.

Clones the microWakeWord repo into the workspace and installs it editable,
clones piper-sample-generator and installs it editable, and downloads the
Piper TTS voice checkpoint it needs.

This mirrors the "Install MicroWakeWord" cell of the original Colab
notebook, but runs against a persistent local workspace directory instead
of Colab's ephemeral VM, so it only does work that hasn't already been done.

Note: microWakeWord has no PyPI release, so it must be cloned and installed
editable. piper-sample-generator was later restructured into a proper PyPI
package (the old `generate_samples.py` script the original notebooks call
no longer exists in that repo), but that PyPI package only ships the
`piper_sample_generator` module, not the sibling `piper_train` module its
`.pt`-checkpoint (multi-speaker generator) code path imports - the PyPI
package's own `pyproject.toml` restricts packaging to `piper_sample_generator*`.
So this clones the repo instead and adds its root to PYTHONPATH when
running it (see generate_samples.py), which makes `piper_train` importable
as a plain source directory without needing it packaged/installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from trainer.local.config import GpuConfig, TrainerConfig

MICROWAKEWORD_REPO = "https://github.com/kahrendt/microWakeWord.git"
PIPER_SAMPLE_GENERATOR_REPO = "https://github.com/rhasspy/piper-sample-generator.git"

# microWakeWord's install_requires minus "tensorflow", which is handled
# separately per GPU backend (see _install_microwakeword).
MICROWAKEWORD_DEPS = [
    "audiomentations",
    "audio_metadata",
    "datasets",
    "mmap_ninja",
    "numpy>=2.0",
    "pymicro-features",
    "pyyaml",
    "webrtcvad-wheels",
    "ai-edge-litert",
]

# AMD's wheel index for tensorflow-rocm builds, one per ROCm release line.
ROCM_WHEEL_INDEX = "https://repo.radeon.com/rocm/manylinux/rocm-rel-{release}/"


def _run(cmd: list[str], **kwargs) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def gpu_env(gpu: GpuConfig) -> dict[str, str]:
    """Environment for running the GPU-enabled tools with this backend.

    AMD's tensorflow-rocm wheels only ship kernels for the GPU targets they
    were built for, which excludes the newer RDNA 3.5 APUs (gfx1150/gfx1151).
    HSA_OVERRIDE_GFX_VERSION makes the GPU present itself as a supported
    target (e.g. "11.0.0") so those kernels load - without it, TensorFlow
    aborts with a hip_code_object.cpp assertion. See the GpuConfig field's
    comment in config.py for details.
    """
    env = os.environ.copy()
    if gpu.backend == "rocm" and gpu.hsa_override_gfx_version:
        env.setdefault("HSA_OVERRIDE_GFX_VERSION", gpu.hsa_override_gfx_version)
    return env


def _clone_if_missing(repo_url: str, dest: Path) -> None:
    if dest.exists():
        print(f"{dest} already exists, skipping clone")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", repo_url, str(dest)])


def _download_if_missing(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"{dest} already exists, skipping download")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def _install_microwakeword(gpu: GpuConfig, mww_repo_dir: Path) -> None:
    if gpu.backend == "rocm":
        # AMD ships ROCm-enabled TensorFlow as the "tensorflow-rocm"
        # distribution (a distinct package name that merely provides the
        # `tensorflow` module) from its own wheel index. Because the name
        # differs, a normal `pip install -e microWakeWord` would see its
        # `tensorflow>=2.18` requirement as unsatisfied and pull the
        # CPU-only PyPI wheel on top, clobbering the ROCm build. So install
        # ROCm TensorFlow and microWakeWord's other deps first, then
        # microWakeWord itself with --no-deps.
        index = ROCM_WHEEL_INDEX.format(release=gpu.rocm_release)
        _run([sys.executable, "-m", "pip", "install", f"tensorflow-rocm=={gpu.tensorflow_rocm_version}", "-f", index])
        _run([sys.executable, "-m", "pip", "install", *MICROWAKEWORD_DEPS])
        _run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(mww_repo_dir)])
    else:
        _run([sys.executable, "-m", "pip", "install", "-e", str(mww_repo_dir)])
        if gpu.backend == "cuda":
            # microWakeWord's dependency list just says `tensorflow`, which alone
            # doesn't pull in the CUDA/cuDNN wheels (Linux/WSL only; Windows
            # native is CPU-only since TF 2.11 dropped GPU support there).
            _run([sys.executable, "-m", "pip", "install", "tensorflow[and-cuda]"])


def verify_gpu(gpu: GpuConfig) -> None:
    """Print whether TensorFlow sees a GPU; non-fatal (CPU fallback still works)."""
    print("\nVerifying TensorFlow device visibility...")
    result = subprocess.run(
        [sys.executable, "-c", "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"],
        capture_output=True,
        text=True,
        env=gpu_env(gpu),
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0 or "device_type='GPU'" not in result.stdout:
        print(
            "WARNING: TensorFlow did not report a GPU. Training will fall back to CPU.\n"
            "  For AMD/ROCm: check `rocminfo` works for this user (video/render groups),\n"
            "  that gpu.rocm_release / gpu.tensorflow_rocm_version in the config match\n"
            "  the installed ROCm, that gpu.hsa_override_gfx_version is set to\n"
            "  '11.0.0' on RDNA 3.5 APUs (gfx1150/gfx1151), and that Python is 3.10 or\n"
            "  3.12 (the only versions AMD's tensorflow-rocm wheels are built for)."
        )
    if result.stderr:
        # TF prints a lot of benign INFO chatter to stderr; only show errors.
        errors = "\n".join(l for l in result.stderr.splitlines() if "E " in l[:3])
        if errors:
            print(errors)


def run(config: TrainerConfig) -> None:
    config.workspace.mkdir(parents=True, exist_ok=True)

    _clone_if_missing(MICROWAKEWORD_REPO, config.mww_repo_dir)
    _install_microwakeword(config.gpu, config.mww_repo_dir)

    _clone_if_missing(PIPER_SAMPLE_GENERATOR_REPO, config.piper_repo_dir)
    _run([sys.executable, "-m", "pip", "install", "-e", str(config.piper_repo_dir)])
    _download_if_missing(config.piper.model_url, config.piper_model_path)

    # piper-sample-generator pins audiomentations==0.33.0, which predates the
    # AddColorNoise transform microWakeWord's augmentation.py needs. We only
    # ever use piper_sample_generator's generation path (not its own
    # `.augment` module), so it's safe to override that pin with a newer
    # version here.
    _run([sys.executable, "-m", "pip", "install", "audiomentations>=0.35"])

    # webrtcvad (pulled in by both repos) still does `import pkg_resources`
    # at load time; setuptools removed that shim in 81.0.
    _run([sys.executable, "-m", "pip", "install", "setuptools<81"])

    # microwakeword.audio.clips.Clips loads our local wav files via
    # datasets.Dataset.from_dict(...).cast_column("audio", ...). datasets>=4.0
    # requires the `torchcodec` package for that (even for local files), which
    # in turn needs a Windows-compatible system FFmpeg build - none of the
    # available prebuilt FFmpeg versions matched what our installed torchcodec
    # supports. Downgrading avoids torchcodec entirely (this is also why
    # download_datasets.py doesn't use `datasets` at all).
    _run([sys.executable, "-m", "pip", "install", "datasets<4.0"])

    # microwakeword.model_train_eval calls tf.summary.scalar for training
    # metrics, which needs tensorboard even though nothing actually views
    # the logs - it's not in microWakeWord's own dependency list.
    _run([sys.executable, "-m", "pip", "install", "tensorboard"])

    if config.gpu.backend != "cpu":
        verify_gpu(config.gpu)

    print("\nSetup complete.")
    print(f"  GPU backend: {config.gpu.backend}")
    print(f"  microWakeWord: {config.mww_repo_dir}")
    print(f"  piper-sample-generator: {config.piper_repo_dir}")
    print(f"  Piper voice checkpoint: {config.piper_model_path}")
