"""Step: augment wake word clips and build spectrogram feature mmaps.

This is the step missing entirely from the baseline Colab notebook
(trainer/microwakeword_trainer_colab.ipynb): its training-config cell
references a `generated_features_output_dir` variable that is never
defined anywhere, because the cell that was supposed to build it - turning
augmented wake word clips into the ragged mmap spectrogram features
microwakeword.model_train_eval actually trains on - is absent.

Ported from microWakeWord's own upstream notebooks/basic_training_notebook.ipynb
(cells 5-7), which does define this step correctly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from trainer.local.config import TrainerConfig

# (split name in our output dirs, split name microwakeword's Clips API expects,
#  repetition count, slide_frames)
# slide_frames=10 for train/validation repeats each spectrogram shifted by one
# frame at a time to simulate streaming inference during non-streaming training.
# The test set uses slide_frames=1 since it's evaluated with the streaming model.
_SPLITS = [
    ("training", "train", 2, 10),
    ("validation", "validation", 1, 10),
    ("testing", "test", 1, 1),
]


def _build_clips_and_augmenter(config: TrainerConfig):
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips

    clips = Clips(
        input_directory=str(config.samples_dir),
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    augmenter = Augmentation(
        augmentation_duration_s=config.augmentation.duration_s,
        augmentation_probabilities=config.augmentation.probabilities,
        impulse_paths=[str(config.rir_dir)],
        background_paths=[str(config.fma_dir), str(config.audioset_dir)],
        background_min_snr_db=config.augmentation.background_min_snr_db,
        background_max_snr_db=config.augmentation.background_max_snr_db,
        min_jitter_s=config.augmentation.min_jitter_s,
        max_jitter_s=config.augmentation.max_jitter_s,
    )
    return clips, augmenter


def preview(config: TrainerConfig, output_path: str = "augmented_clip_preview.wav") -> Path:
    """Augment one random clip and save it so you can listen to it."""
    from microwakeword.audio.audio_utils import save_clip

    clips, augmenter = _build_clips_and_augmenter(config)
    random_clip = clips.get_random_clip()
    augmented_clip = augmenter.augment_clip(random_clip)
    dest = Path(output_path)
    save_clip(augmented_clip, str(dest))
    print(f"Wrote preview clip to {dest}")
    return dest


def run(config: TrainerConfig) -> None:
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    if not any(config.samples_dir.glob("*.wav")):
        raise FileNotFoundError(
            f"No wake word samples found in {config.samples_dir}. "
            "Run the 'generate-samples' step first."
        )

    clips, augmenter = _build_clips_and_augmenter(config)

    for out_name, split_name, repetition, slide_frames in _SPLITS:
        out_dir = config.features_dir / out_name
        mmap_dir = out_dir / "wakeword_mmap"
        # RaggedMmap.from_generator creates mmap_dir up front, before writing
        # any data, so a crash partway through a previous run can leave an
        # empty (but existing) directory behind. Check for the data file
        # itself, not just the directory, or a resumed run will silently
        # treat that stub as "done" and skip rebuilding it.
        if (mmap_dir / "data.ninja").exists():
            print(f"{mmap_dir} already exists, skipping {out_name} split")
            continue
        if mmap_dir.exists():
            print(f"Removing incomplete {mmap_dir} from a previous run")
            shutil.rmtree(mmap_dir)

        out_dir.mkdir(parents=True, exist_ok=True)
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=slide_frames,
            step_ms=10,
        )
        print(f"Building {out_name} features (split={split_name}, repeat={repetition})...")
        RaggedMmap.from_generator(
            out_dir=str(mmap_dir),
            sample_generator=spectrograms.spectrogram_generator(split=split_name, repeat=repetition),
            batch_size=100,
            verbose=True,
        )

    print(f"\nFeatures written under {config.features_dir}")
