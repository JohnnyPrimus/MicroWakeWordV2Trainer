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

import subprocess
import sys
import urllib.request
from pathlib import Path

from trainer.local.config import TrainerConfig

MICROWAKEWORD_REPO = "https://github.com/kahrendt/microWakeWord.git"
PIPER_SAMPLE_GENERATOR_REPO = "https://github.com/rhasspy/piper-sample-generator.git"


def _run(cmd: list[str], **kwargs) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


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


def run(config: TrainerConfig) -> None:
    config.workspace.mkdir(parents=True, exist_ok=True)

    _clone_if_missing(MICROWAKEWORD_REPO, config.mww_repo_dir)
    _run([sys.executable, "-m", "pip", "install", "-e", str(config.mww_repo_dir)])

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

    print("\nSetup complete.")
    print(f"  microWakeWord: {config.mww_repo_dir}")
    print(f"  piper-sample-generator: {config.piper_repo_dir}")
    print(f"  Piper voice checkpoint: {config.piper_model_path}")
