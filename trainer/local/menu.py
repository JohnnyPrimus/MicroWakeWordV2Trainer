"""Interactive, menu-driven runner for the local microWakeWord pipeline.

Walks the pipeline steps (setup -> preview-word -> generate-samples ->
download-data -> preview-augment -> build-features -> train -> export) in
order, running each one automatically, and only pauses when a step needs a
human decision: confirming a phonetic spelling sounds right, previewing
augmentation, choosing whether to retrain over an existing checkpoint,
picking a probability cutoff after looking at the ROC results, and entering
export metadata. Long-running steps with no decision to make (generate-
samples, download-data, build-features) just run straight through.

Run with:
    python -m trainer.local.menu [--config trainer/wakeword_config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional

import yaml

from trainer.local.config import TrainerConfig, load_config


class Cancelled(Exception):
    """Raised to bail out of the running pipeline back to the main menu."""


def _banner(text: str) -> None:
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def _ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        answer = input(f"{prompt}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default


def _confirm(prompt: str, default_yes: bool) -> bool:
    hint = "Y/n" if default_yes else "y/N"
    answer = input(f"{prompt} [{hint}]: ").strip().lower()
    if not answer:
        return default_yes
    return answer.startswith("y")


def _pause_menu(prompt: str, options: List[tuple]) -> str:
    print(f"\n{prompt}")
    for key, label in options:
        print(f"  [{key}] {label}")
    valid = {key.lower() for key, _ in options}
    while True:
        choice = input("> ").strip().lower()
        if choice in valid:
            return choice
        print(f"Enter one of: {', '.join(key for key, _ in options)}")


class Session:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config: TrainerConfig = load_config(config_path)

    def reload(self) -> None:
        self.config = load_config(self.config_path)


def _bootstrap_config(config_path: Path) -> None:
    example_path = config_path.parent / "wakeword_config.example.yaml"
    if not example_path.exists():
        raise FileNotFoundError(f"Missing template {example_path}")

    print(f"No config found at {config_path}.")
    if not _confirm("Create one now?", True):
        raise Cancelled()

    phonetic = _ask("Phonetic spelling of your wake word (e.g. 'hey wabi')")
    import re

    default_friendly = re.sub(r"[^a-z0-9]+", "_", phonetic.lower()).strip("_") or "wake_word"
    friendly = _ask("Friendly name (directory-safe, used for files/folders)", default_friendly)

    with open(example_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["wake_word"] = {"phonetic": phonetic, "friendly_name": friendly, "additional_phonetics": []}

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, sort_keys=False)

    print(
        f"\nWrote {config_path}. It has every field with its default value - see "
        f"{example_path.name} for comments on what each one does. You can edit it "
        "further at any time and reload from the main menu."
    )


# --- per-step "already done" checks, used only for status labels ---------


def _has_wavs(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.wav"))


def _setup_done(config: TrainerConfig) -> bool:
    return config.mww_repo_dir.exists() and config.piper_repo_dir.exists() and config.piper_model_path.exists()


def _samples_done(config: TrainerConfig) -> bool:
    return _has_wavs(config.samples_dir)


def _download_done(config: TrainerConfig) -> bool:
    required = ["speech", "dinner_party", "no_speech", "dinner_party_eval"]
    negatives_ok = all((config.negative_datasets_dir / r).exists() for r in required)
    return negatives_ok and _has_wavs(config.rir_dir) and _has_wavs(config.audioset_dir) and _has_wavs(config.fma_dir)


def _features_done(config: TrainerConfig) -> bool:
    return all(
        (config.features_dir / split / "wakeword_mmap" / "data.ninja").exists()
        for split in ("training", "validation", "testing")
    )


def _train_done(config: TrainerConfig) -> bool:
    from trainer.local.export import TFLITE_SOURCE_RELATIVE

    return (config.train_dir / TFLITE_SOURCE_RELATIVE).exists()


def _export_done(config: TrainerConfig) -> bool:
    return config.export_dir.exists() and any(config.export_dir.glob("*.tflite"))


# --- pipeline steps --------------------------------------------------------


def step_setup(session: Session) -> None:
    from trainer.local import setup_env

    _banner("Step 1/8: Setup - install microWakeWord + piper-sample-generator")
    setup_env.run(session.config)


def step_preview_word(session: Session) -> None:
    from trainer.local import generate_samples
    from trainer.local.generate_samples import _slug

    while True:
        config = session.config
        _banner("Step 2/8: Preview phonetic spelling")
        print(f'Generating one sample for phonetic spelling: "{config.wake_word.phonetic}"')
        generate_samples.run(config, preview_only=True)
        out_path = config.preview_dir / _slug(config.wake_word.phonetic) / "0.wav"
        print(f"\nListen to: {out_path.resolve()}")
        choice = _pause_menu(
            "Does the phonetic spelling sound right?",
            [
                ("c", "Continue to generating the full sample batch"),
                ("r", "I edited wake_word.phonetic in the config file - regenerate the preview"),
                ("q", "Stop here (back to main menu)"),
            ],
        )
        if choice == "c":
            return
        if choice == "q":
            raise Cancelled()
        session.reload()


def step_generate_samples(session: Session) -> None:
    from trainer.local import generate_samples

    _banner("Step 3/8: Generate full sample batch")
    generate_samples.run(session.config)


def step_download_data(session: Session) -> None:
    from trainer.local import download_datasets

    _banner("Step 4/8: Download background/negative audio data (slowest step - can take 30-45+ min)")
    download_datasets.run(session.config)


def step_preview_augment(session: Session) -> None:
    from trainer.local import build_features

    _banner("Step 5/8: Preview augmentation (optional)")
    if not _confirm("Generate a sample augmented clip to listen to before building features?", False):
        return
    while True:
        config = session.config
        dest = build_features.preview(config)
        print(f"\nListen to: {dest.resolve()}")
        choice = _pause_menu(
            "Sound OK?",
            [
                ("c", "Continue to building features"),
                ("r", "I edited augmentation settings in the config - preview again"),
                ("s", "Skip further previews and continue"),
            ],
        )
        if choice in ("c", "s"):
            return
        session.reload()


def step_build_features(session: Session) -> None:
    from trainer.local import build_features

    _banner("Step 6/8: Build spectrogram features")
    build_features.run(session.config)


def step_train(session: Session) -> None:
    from trainer.local import train as train_mod
    from trainer.local.export import TFLITE_SOURCE_RELATIVE

    _banner("Step 7/8: Train")
    config = session.config
    tflite_path = config.train_dir / TFLITE_SOURCE_RELATIVE
    do_train = True
    if tflite_path.exists():
        choice = _pause_menu(
            f"A trained model already exists at {tflite_path}.",
            [
                ("r", "Retrain / continue training more steps"),
                ("s", "Skip training, reuse the existing model as-is"),
                ("q", "Stop here (back to main menu)"),
            ],
        )
        if choice == "q":
            raise Cancelled()
        do_train = choice == "r"
    else:
        print(f"Training for {config.training.training_steps} step(s) (see training: in the config to change this)...")
    train_mod.run(config, train=do_train)


def step_export(session: Session) -> None:
    from trainer.local import export as export_mod

    _banner("Step 8/8: Export for ESPHome")
    config = session.config
    roc_path = config.train_dir / "tflite_stream_state_internal_quant" / "tflite_streaming_roc.txt"
    if roc_path.exists():
        print(f"ROC results (frr = false-rejection rate, faph = false alarms/hour): {roc_path}")
        lines = roc_path.read_text(encoding="utf-8").splitlines()
        preview_lines = lines[:15]
        print("\n".join(preview_lines))
        if len(lines) > len(preview_lines):
            print(f"... ({len(lines) - len(preview_lines)} more lines - see the file above)")
        print("Pick the lowest cutoff with faph=0.000 as a reasonable default.")
    else:
        print(f"No ROC results found at {roc_path} (did the train step complete?)")

    cutoff = float(_ask("Probability cutoff to export with", "0.5"))
    author = _ask("Author (optional)", "")
    website = _ask("Website (optional)", "")
    tensor_arena = int(_ask("Tensor arena size (raise if ESPHome fails to load the model)", "30000"))

    export_mod.run(
        config,
        author=author,
        website=website,
        probability_cutoff=cutoff,
        tensor_arena_size=tensor_arena,
    )


STEP_FUNCS: List[Callable[[Session], None]] = [
    step_setup,
    step_preview_word,
    step_generate_samples,
    step_download_data,
    step_preview_augment,
    step_build_features,
    step_train,
    step_export,
]

STEP_NAMES = [
    "Setup",
    "Preview phonetic spelling",
    "Generate sample batch",
    "Download background/negative data",
    "Preview augmentation (optional)",
    "Build features",
    "Train",
    "Export",
]

_STEP_DONE_CHECKS = [
    _setup_done,
    None,
    _samples_done,
    _download_done,
    None,
    _features_done,
    _train_done,
    _export_done,
]


def _step_label(config: TrainerConfig, index: int) -> str:
    check = _STEP_DONE_CHECKS[index]
    if check is None:
        return ""
    return " (done)" if check(config) else ""


def run_pipeline(session: Session, start_index: int) -> None:
    i = start_index
    while i < len(STEP_FUNCS):
        try:
            STEP_FUNCS[i](session)
        except Cancelled:
            print("\nStopped. Back to the main menu.")
            return
        except Exception as e:
            print(f"\n{STEP_NAMES[i]} failed: {e}")
            choice = _pause_menu("What now?", [("r", "Retry this step"), ("m", "Back to main menu")])
            if choice == "m":
                return
            continue
        i += 1
    _banner("Pipeline complete")
    print(f"Exported model directory: {session.config.export_dir}")


def main_menu(session: Session) -> None:
    while True:
        config = session.config
        print()
        print("=" * 70)
        print(f'MicroWakeWord Trainer - wake word: "{config.wake_word.phonetic}" ({config.wake_word.friendly_name})')
        print(f"Config: {session.config_path}")
        print("=" * 70)
        print("  [1] Run full pipeline (setup through export, in order)")
        print("  [2] Run starting from a specific step")
        print("  [3] Reload config (after you've edited the file)")
        print("  [4] Exit")
        choice = input("> ").strip()
        if choice == "1":
            run_pipeline(session, 0)
        elif choice == "2":
            print()
            for i, name in enumerate(STEP_NAMES, 1):
                print(f"  [{i}] {name}{_step_label(config, i - 1)}")
            sel = input("Start from step > ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(STEP_NAMES):
                run_pipeline(session, int(sel) - 1)
            else:
                print("Not a valid step number.")
        elif choice == "3":
            session.reload()
            print("Config reloaded.")
        elif choice == "4":
            return
        else:
            print("Enter 1-4.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive menu-driven runner for the local microWakeWord pipeline.")
    parser.add_argument(
        "--config",
        default="trainer/wakeword_config.yaml",
        help="Path to a wakeword_config.yaml (see trainer/wakeword_config.example.yaml)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        try:
            _bootstrap_config(config_path)
        except Cancelled:
            print("No config - exiting.")
            return 1

    session = Session(config_path)
    try:
        main_menu(session)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
