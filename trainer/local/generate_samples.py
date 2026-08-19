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
import subprocess
import sys

from trainer.local.config import TrainerConfig


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

    config.samples_dir.mkdir(parents=True, exist_ok=True)

    max_samples = 1 if preview_only else config.piper.max_samples
    batch_size = 1 if preview_only else config.piper.batch_size

    cmd = [
        sys.executable,
        "-m",
        "piper_sample_generator",
        config.wake_word.phonetic,
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
        str(config.samples_dir),
    ]
    if config.piper.max_speakers:
        cmd += ["--max-speakers", str(config.piper.max_speakers)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(config.piper_repo_dir) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    sample_count = len(list(config.samples_dir.glob("*.wav")))
    print(f"\n{sample_count} wake word sample(s) in {config.samples_dir}")
    if preview_only:
        print(f"Listen to {config.samples_dir / '0.wav'} to check the phonetic spelling.")
