"""Step: export the trained model for ESPHome.

Copies the quantized streaming .tflite produced by train.py into an export
directory under a clean filename, and writes the ESPHome V2 model manifest
JSON next to it (see https://esphome.io/components/micro_wake_word and
https://github.com/esphome/micro-wake-word-models/tree/main/models/v2 for
the format).

`probability_cutoff` and `tensor_arena_size` are the two values you will
most likely need to tune after testing the model on-device: raise
probability_cutoff if it triggers on background noise, and raise
tensor_arena_size if ESPHome fails to load the model.
"""

from __future__ import annotations

import json
import shutil

from trainer.local.config import TrainerConfig

TFLITE_SOURCE_RELATIVE = "tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"


def run(
    config: TrainerConfig,
    author: str = "",
    website: str = "",
    probability_cutoff: float = 0.5,
    sliding_window_size: int = 5,
    tensor_arena_size: int = 30000,
    minimum_esphome_version: str = "2024.7.0",
) -> None:
    src = config.train_dir / TFLITE_SOURCE_RELATIVE
    if not src.exists():
        raise FileNotFoundError(f"Trained model not found at {src}. Run the 'train' step first.")

    config.export_dir.mkdir(parents=True, exist_ok=True)
    model_filename = f"{config.wake_word.friendly_name}.tflite"
    dest = config.export_dir / model_filename
    shutil.copyfile(src, dest)

    manifest = {
        "type": "micro",
        "wake_word": config.wake_word.phonetic,
        "author": author,
        "website": website,
        "model": model_filename,
        "trained_languages": ["en"],
        "version": 2,
        "micro": {
            "probability_cutoff": probability_cutoff,
            "feature_step_size": 10,  # matches window_step_ms hardcoded in train.py
            "sliding_window_size": sliding_window_size,
            "tensor_arena_size": tensor_arena_size,
            "minimum_esphome_version": minimum_esphome_version,
        },
    }
    manifest_path = config.export_dir / f"{config.wake_word.friendly_name}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported model:    {dest}")
    print(f"Exported manifest: {manifest_path}")
    print(
        "\nprobability_cutoff and tensor_arena_size are starting guesses - "
        "test on-device and adjust as needed."
    )
