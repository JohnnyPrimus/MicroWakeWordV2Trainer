"""Local microWakeWord training pipeline CLI.

Usage:
    python -m trainer.local.cli [<command>] --config trainer/wakeword_config.yaml

Running with no command (or `menu`) launches the interactive, menu-driven
runner: it walks every step below in order and only stops for input at
points that need a human decision (see trainer/local/menu.py). Individual
commands remain available for scripting a single step directly.

Commands:
    menu               Interactive menu-driven runner (default if no command given).
    setup             Clone/install microWakeWord + piper-sample-generator, download the Piper voice checkpoint.
    preview-word       Generate one wake word sample so you can listen to the phonetic spelling.
    generate-samples   Generate the full batch of synthetic wake word samples.
    download-data      Download background/negative audio datasets for augmentation and training.
    preview-augment     Augment one random clip and save it so you can listen to it.
    build-features     Augment samples and build the spectrogram feature mmaps (training/validation/testing).
    train              Train the model and convert it to a quantized streaming .tflite.
    export             Copy the trained .tflite and write an ESPHome V2 model manifest.
    all                Run download-data, build-features, and train in sequence (assumes setup + generate-samples already ran).

See trainer/README.md for the full walkthrough and trainer/wakeword_config.example.yaml
for a documented config template.
"""

from __future__ import annotations

import argparse
import sys

from trainer.local.config import load_config


def _add_config_arg(parser: argparse.ArgumentParser, *, suppress_default: bool = False) -> None:
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS if suppress_default else "trainer/wakeword_config.yaml",
        help="Path to a wakeword_config.yaml (see trainer/wakeword_config.example.yaml)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_config_arg(parser)
    subparsers = parser.add_subparsers(dest="command")

    for name, help_text in [
        ("menu", "Interactive menu-driven runner (default if no command given)"),
        ("setup", "Install/clone dependencies"),
        ("preview-word", "Generate one wake word sample to check the phonetic spelling"),
        ("generate-samples", "Generate the full batch of wake word samples"),
        ("download-data", "Download background/negative datasets"),
        ("preview-augment", "Augment one random clip and save it for listening"),
        ("build-features", "Build spectrogram feature mmaps"),
        ("train", "Train the model"),
        ("export", "Export the trained .tflite + ESPHome manifest"),
        ("all", "Run download-data, build-features, and train in sequence"),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        _add_config_arg(sub, suppress_default=True)
        if name == "train":
            sub.add_argument(
                "--no-train",
                action="store_true",
                help="Skip training; just convert+test the existing best_weights checkpoint to .tflite",
            )
        if name == "export":
            sub.add_argument("--author", default="")
            sub.add_argument("--website", default="")
            sub.add_argument(
                "--probability-cutoff",
                type=float,
                default=0.5,
                help="See workspace/trained_models/<name>/tflite_stream_state_internal_quant/tflite_streaming_roc.txt "
                "(from the train step) to pick a cutoff with a good false-rejection-rate/false-alarm-rate tradeoff.",
            )
            sub.add_argument("--tensor-arena-size", type=int, default=30000)

    args = parser.parse_args(argv)

    if args.command is None or args.command == "menu":
        from trainer.local import menu

        return menu.main(["--config", args.config])

    config = load_config(args.config)

    if args.command == "setup":
        from trainer.local import setup_env

        setup_env.run(config)
    elif args.command == "preview-word":
        from trainer.local import generate_samples

        generate_samples.run(config, preview_only=True)
    elif args.command == "generate-samples":
        from trainer.local import generate_samples

        generate_samples.run(config)
    elif args.command == "download-data":
        from trainer.local import download_datasets

        download_datasets.run(config)
    elif args.command == "preview-augment":
        from trainer.local import build_features

        build_features.preview(config)
    elif args.command == "build-features":
        from trainer.local import build_features

        build_features.run(config)
    elif args.command == "train":
        from trainer.local import train

        train.run(config, train=not args.no_train)
    elif args.command == "export":
        from trainer.local import export

        export.run(
            config,
            author=args.author,
            website=args.website,
            probability_cutoff=args.probability_cutoff,
            tensor_arena_size=args.tensor_arena_size,
        )
    elif args.command == "all":
        from trainer.local import build_features, download_datasets, train

        download_datasets.run(config)
        build_features.run(config)
        train.run(config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
