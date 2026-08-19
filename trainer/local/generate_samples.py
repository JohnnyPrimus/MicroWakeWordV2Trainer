"""Step: generate synthetic wake word samples with Piper TTS.

Equivalent of the Colab notebook's sample-generation cells, driven by the
`piper_sample_generator` module (module invocation, since the
`generate_samples.py` script path the original notebooks used no longer
exists upstream - see setup_env.py).

`piper_sample_generator.__main__` imports `piper_train.vits.commons` for its
multi-speaker generator (.pt checkpoint) code path, but `piper_train` isn't
part of the installed package - it only exists in the cloned repo. Running
with PYTHONPATH set to the repo root makes it importable as a plain source
directory without needing it packaged.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from trainer.local.config import TrainerConfig


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "phonetic"


def _generate_batch(config: TrainerConfig, phonetic: str, output_dir: Path, max_samples: int, batch_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "piper_sample_generator",
        phonetic,
        "--model",
        str(config.piper_model_path),
        "--max-samples",
        str(max_samples),
        "--batch-size",
        str(batch_size),
        "--length-scales",
        *[str(v) for v in config.piper.length_scales],
        "--noise-scales",
        *[str(v) for v in config.piper.noise_scales],
        "--noise-scale-ws",
        *[str(v) for v in config.piper.noise_scale_ws],
        "--output-dir",
        str(output_dir),
    ]
    if config.piper.max_speakers:
        cmd += ["--max-speakers", str(config.piper.max_speakers)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(config.piper_repo_dir) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def run(config: TrainerConfig, preview_only: bool = False) -> None:
    if not config.piper_model_path.exists():
        raise FileNotFoundError(
            f"Piper voice checkpoint not found at {config.piper_model_path}. "
            "Run the 'setup' step first."
        )
    if not config.piper_repo_dir.exists():
        raise FileNotFoundError(
            f"piper-sample-generator repo not found at {config.piper_repo_dir}. "
            "Run the 'setup' step first."
        )

    if preview_only:
        # Only preview the primary spelling - additional_phonetics are
        # meant to be already-decided variants, not ones you're auditioning.
        out_dir = config.preview_dir / _slug(config.wake_word.phonetic)
        _generate_batch(config, config.wake_word.phonetic, out_dir, max_samples=1, batch_size=1)
        print(f"\nListen to {out_dir / '0.wav'} to check the phonetic spelling.")
        return

    phonetics = [config.wake_word.phonetic, *config.wake_word.additional_phonetics]
    total = 0
    for phonetic in phonetics:
        out_dir = config.samples_dir / _slug(phonetic)
        existing = len(list(out_dir.glob("*.wav"))) if out_dir.exists() else 0
        if existing >= config.piper.max_samples:
            print(f"{out_dir} already has {existing} samples, skipping '{phonetic}'")
            total += existing
            continue
        _generate_batch(config, phonetic, out_dir, config.piper.max_samples, config.piper.batch_size)
        total += len(list(out_dir.glob("*.wav")))

    print(f"\n{total} wake word sample(s) across {len(phonetics)} phonetic spelling(s) in {config.samples_dir}")
