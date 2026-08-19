"""Step: write microwakeword's training_parameters.yaml and run training.

Ports the Colab notebook's config + train cells (cell-9 and cell-10) with
one fix: the positive features_dir is `config.features_dir` (this pipeline's
actual output directory from build_features.py), not the notebook's
undefined `generated_features_output_dir`.
"""

from __future__ import annotations

import subprocess
import sys

import yaml

from trainer.local.config import TrainerConfig


def write_training_parameters(config: TrainerConfig) -> None:
    t = config.training

    training_config = {
        "window_step_ms": 10,
        "train_dir": str(config.train_dir),
        "features": [
            {
                "features_dir": str(config.features_dir),
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": str(config.negative_datasets_dir / "speech"),
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": str(config.negative_datasets_dir / "dinner_party"),
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": str(config.negative_datasets_dir / "no_speech"),
                "sampling_weight": 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                # Only used for validation and testing.
                "features_dir": str(config.negative_datasets_dir / "dinner_party_eval"),
                "sampling_weight": 0.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "split",
                "type": "mmap",
            },
        ],
        "training_steps": t.training_steps,
        "positive_class_weight": t.positive_class_weight,
        "negative_class_weight": t.negative_class_weight,
        "learning_rates": t.learning_rates,
        "batch_size": t.batch_size,
        "time_mask_max_size": t.time_mask_max_size,
        "time_mask_count": t.time_mask_count,
        "freq_mask_max_size": t.freq_mask_max_size,
        "freq_mask_count": t.freq_mask_count,
        "eval_step_interval": t.eval_step_interval,
        "clip_duration_ms": t.clip_duration_ms,
        "target_minimization": t.target_minimization,
        "minimization_metric": t.minimization_metric,
        "maximization_metric": t.maximization_metric,
    }

    config.workspace.mkdir(parents=True, exist_ok=True)
    with open(config.training_parameters_yaml, "w", encoding="utf-8") as f:
        yaml.dump(training_config, f)
    print(f"Wrote {config.training_parameters_yaml}")


def run(config: TrainerConfig, train: bool = True) -> None:
    write_training_parameters(config)

    for required in ["speech", "dinner_party", "no_speech", "dinner_party_eval"]:
        if not (config.negative_datasets_dir / required).exists():
            raise FileNotFoundError(
                f"Missing negative dataset {config.negative_datasets_dir / required}. "
                "Run the 'download-data' step first."
            )
    if not (config.features_dir / "training" / "wakeword_mmap").exists():
        raise FileNotFoundError(
            f"Missing positive features under {config.features_dir}. "
            "Run the 'build-features' step first."
        )

    model_args = config.training.model_args
    cmd = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
        f"--training_config={config.training_parameters_yaml}",
        f"--train={1 if train else 0}",
        "--restore_checkpoint=1",
        "--test_tf_nonstreaming=0",
        "--test_tflite_nonstreaming=0",
        "--test_tflite_nonstreaming_quantized=0",
        "--test_tflite_streaming=0",
        "--test_tflite_streaming_quantized=1",
        "--use_weights=best_weights",
        config.training.model,
        "--pointwise_filters",
        str(model_args["pointwise_filters"]),
        "--repeat_in_block",
        str(model_args["repeat_in_block"]),
        "--mixconv_kernel_sizes",
        str(model_args["mixconv_kernel_sizes"]),
        "--residual_connection",
        str(model_args["residual_connection"]),
        "--first_conv_filters",
        str(model_args["first_conv_filters"]),
        "--first_conv_kernel_size",
        str(model_args["first_conv_kernel_size"]),
        "--stride",
        str(model_args["stride"]),
    ]

    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(config.workspace))
