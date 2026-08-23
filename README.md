# Local microWakeWord Trainer

A local, script-driven alternative to `trainer/microwakeword_trainer_colab.ipynb`.
It runs the same underlying pipeline as the Colab notebook — synthesize wake
word audio with Piper TTS, augment it with background noise and room
impulse responses, turn that into spectrogram features, and train a
[microWakeWord](https://github.com/kahrendt/microWakeWord) model — as a
plain Python CLI you run on your own machine instead of in a Colab VM.

## What it does differently from the baseline notebook

- **Fixes a missing step.** The notebook's training-config cell references a
  `generated_features_output_dir` variable that's never defined anywhere -
  the cell that actually turns augmented clips into the spectrogram features
  microWakeWord trains on is absent. `build_features.py` (the
  `build-features` command) is that missing step, ported from microWakeWord's
  own upstream `notebooks/basic_training_notebook.ipynb`.
- **Supports multiple phonetic spellings of the same wake word**
  (`wake_word.additional_phonetics` in the config) - each spelling gets its
  own full batch of generated samples, which generally improves the model's
  robustness to different pronunciations.
- **Resumable, idempotent steps.** Every step checks what it's already done
  and skips it, so you can re-run the same command after an interruption
  (or after changing a downstream config value) without redoing finished
  work.
- **Routes around several environment-specific dependency issues** (see
  [Troubleshooting](#troubleshooting--why-does-setup-do-all-that) below) that
  the original notebook doesn't have to deal with, since Colab's VM image
  and package versions are fixed at the time the notebook was written.
- **Menu-driven by default.** `python -m trainer.local.cli` (no arguments)
  launches an interactive menu that runs every step below in order and only
  stops to ask you something when a step genuinely needs a human decision -
  see [Running the pipeline](#running-the-pipeline).

## Pipeline overview

```
setup              → clone/install microWakeWord + piper-sample-generator, download the Piper voice model
preview-word       → generate 1 sample, listen to it, adjust phonetic spelling
generate-samples   → generate the full batch of synthetic wake word samples
download-data      → download RIR / Audioset / FMA background audio + pre-built negative feature sets
preview-augment    → (optional) generate 1 augmented clip to sanity-check augmentation settings
build-features     → augment samples, build train/validation/test spectrogram feature mmaps
train              → train the model, quantize + convert to a streaming .tflite
export             → copy the .tflite + write an ESPHome V2 model manifest .json
```

Everything a run produces - cloned repos, downloaded datasets, generated
samples, feature mmaps, trained checkpoints, and the final exported model -
lives under one `workspace/` directory (configurable), so different wake
words can have entirely independent workspaces.

## Requirements

- Python 3.10+ (developed and tested on 3.11)
- `git`
- ~30-40 GB free disk per workspace (background/negative audio datasets are
  the bulk of it: Audioset alone is several GB per split)
- An NVIDIA GPU is optional but strongly recommended for training speed -
  see the GPU note below. CPU-only works, just slowly (10,000 training
  steps can take an hour or more on a laptop CPU).

### GPU note (Windows)

TensorFlow dropped native GPU support on Windows after version 2.10, and
microWakeWord requires `tensorflow>=2.18`. That means:

- **Native Windows** (PowerShell, no WSL): training runs **CPU-only**, no
  matter what GPU you have.
- **WSL2** (Windows Subsystem for Linux): TensorFlow can use an NVIDIA GPU
  normally, via the same driver passthrough any other WSL2 CUDA workload
  uses.

The commands below are identical on both; only the initial environment setup
differs. Pick whichever section matches how you want to run it.

---

## Setup & run — Windows (native, CPU-only)

1. Install [Python 3.11](https://www.python.org/downloads/) and
   [Git](https://git-scm.com/download/win) if you don't already have them.

2. From the repo root, in PowerShell:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r trainer\requirements.txt
   ```

3. Create your wake word config:

   ```powershell
   Copy-Item trainer\wakeword_config.example.yaml trainer\wakeword_config.yaml
   ```

   Edit `trainer\wakeword_config.yaml` - at minimum set `wake_word.phonetic`
   and `wake_word.friendly_name`. See the comments in
   `trainer\wakeword_config.example.yaml` for what every other field does.

4. Launch the interactive menu - it installs the rest of the toolchain
   (microWakeWord, piper-sample-generator, the Piper voice checkpoint) as its
   first step, then walks you through the rest of the pipeline. See
   [Running the pipeline](#running-the-pipeline) below.

   ```powershell
   python -m trainer.local.cli
   ```

---

## Setup & run — WSL2 (recommended for GPU training)

1. **Install WSL2 with Ubuntu**, if you haven't already. From an elevated
   PowerShell prompt:

   ```powershell
   wsl --install -d Ubuntu
   ```

   Reboot if prompted, then open the Ubuntu app once to finish first-time
   setup (creates your Linux user account).

2. **Install an NVIDIA driver on the Windows side only.** Download the
   latest driver from
   [nvidia.com](https://www.nvidia.com/Download/index.aspx) and install it
   normally on Windows. Do **not** install a separate NVIDIA Linux driver
   inside WSL - WSL2 shares the Windows host's driver via GPU passthrough,
   and installing one inside Ubuntu will break it.

3. **Verify the GPU is visible inside WSL:**

   ```bash
   nvidia-smi
   ```

   This should print your GPU, driver version, and CUDA version without any
   extra setup. If it fails, the Windows-side driver install above is the
   thing to fix first.

4. **Install build tools and Python** inside Ubuntu (WSL):

   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip build-essential git
   ```

5. **Get to the repo.** Either work on your existing Windows checkout via
   the `/mnt/c/...` path:

   ```bash
   cd "/mnt/c/Users/<you>/git/MicroWakeWordV2Trainer"
   ```

   or clone a fresh copy into WSL's native filesystem for better I/O
   performance (matters most for the download-data step, which writes
   tens of thousands of small files):

   ```bash
   git clone <repo-url> ~/MicroWakeWordV2Trainer
   cd ~/MicroWakeWordV2Trainer
   ```

6. **Set up the Python environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r trainer/requirements.txt
   ```

7. **Create your wake word config** (same as the Windows instructions):

   ```bash
   cp trainer/wakeword_config.example.yaml trainer/wakeword_config.yaml
   ```

   Edit it - at minimum `wake_word.phonetic` and `wake_word.friendly_name`.

8. **Run setup**, then explicitly install the CUDA-enabled TensorFlow
   extras (microWakeWord's own dependency list just says `tensorflow`,
   which alone doesn't pull in the CUDA/cuDNN wheels):

   ```bash
   python3 -m trainer.local.cli setup
   pip install "tensorflow[and-cuda]"
   ```

9. **Verify TensorFlow sees the GPU:**

   ```bash
   python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

   You should see a `PhysicalDevice` entry with `device_type='GPU'`. If the
   list is empty, re-check steps 2-3 and step 8's `tensorflow[and-cuda]`
   install.

10. Launch the interactive menu for the rest of the pipeline - see
    [Running the pipeline](#running-the-pipeline) below. It's `python3 -m
    trainer.local.cli` inside WSL (same as Windows, just `python3` instead of
    `python`). Its first step re-runs `setup`, which is safe here too - it
    won't touch the CUDA-enabled TensorFlow you just installed in step 8.

If your GPU has limited VRAM (e.g. a laptop GPU with 6-8 GB), and training
fails with an out-of-memory error, lower `training.batch_size` in the
config (default 128).

---

## Running the pipeline

The recommended way to run everything is the interactive menu:

```
python -m trainer.local.cli
```

(`python3` on WSL/Linux.) It walks through every step below in order -
setup, preview-word, generate-samples, download-data, preview-augment,
build-features, train, export - running each one automatically, and only
stops to ask you something at the points that actually need a human
decision: listening to a preview and confirming it sounds right, choosing
whether to retrain over an existing checkpoint, picking a probability
cutoff after looking at the ROC results, and entering export metadata.
Everything else just runs straight through. If `trainer/wakeword_config.yaml`
doesn't exist yet, it offers to create one interactively on first launch.
From the main menu you can also jump in and re-run starting from any
individual step (e.g. after editing the config).

Pass `--config <path>` (before or after the subcommand) to point at a
different config file - e.g. to train several wake words side by side, since
each config's `workspace` directory is independent.

Every step is also available as its own `python -m trainer.local.cli
<command>` for scripting a single step directly, documented below. Both
forms are equivalent - the menu just sequences the same commands and pauses
in the right places. Every step is safe to re-run either way: it checks
what it's already produced and skips finished work.

1. **Check the phonetic spelling** by generating one sample and listening to it:

   ```
   python -m trainer.local.cli preview-word
   ```

   Listen to `workspace/preview_samples/<friendly_name>/<slug>/0.wav`.
   Adjust `wake_word.phonetic` in the config until it sounds right, then
   re-run.

2. **Generate the full sample batch:**

   ```
   python -m trainer.local.cli generate-samples
   ```

   Generates `piper.max_samples` samples for `wake_word.phonetic`, plus a
   full batch for each entry in `wake_word.additional_phonetics` (see
   [Multiple phonetic spellings](#multiple-phonetic-spellings) below).
   Controlled by `piper.max_samples` / `piper.batch_size` in the config -
   this is the step to revisit first when trying to improve a model: more
   samples, more phonetic variants, or different `length_scales`/
   `noise_scales` all help.

3. **Download background/negative audio data** (RIRs, Audioset, FMA, and
   pre-generated negative spectrogram sets). This is the slowest step -
   expect 30-45+ minutes and several GB of disk:

   ```
   python -m trainer.local.cli download-data
   ```

   > **License note:** this data mixes several licenses and usage
   > restrictions. Treat models trained with it as personal-use only unless
   > you check the license of each source for commercial use.

4. **(Optional) Preview an augmented clip:**

   ```
   python -m trainer.local.cli preview-augment
   ```

   Writes `augmented_clip_preview.wav` in the current directory, so you can
   listen to what the augmentation pipeline (background noise mixing, RIR,
   pitch/gain/EQ perturbation) actually sounds like.

5. **Build spectrogram features** (augments samples, builds the
   training/validation/testing feature mmaps):

   ```
   python -m trainer.local.cli build-features
   ```

   Re-run this after adding samples (a new phonetic spelling, more
   generated samples, etc.) - it globs `generated_samples/<name>/**/*.wav`
   fresh each time it needs to rebuild a split.

6. **Train:**

   ```
   python -m trainer.local.cli train
   ```

   Hyperparameters live under `training:` in the config (training steps,
   class weights, learning rate, model architecture args, etc.) - these
   matter a lot for model quality, especially for very short or long wake
   words. This step also quantizes and converts the best checkpoint to a
   streaming `.tflite`, and writes real test results to
   `workspace/trained_models/<name>/tflite_stream_state_internal_quant/tflite_streaming_roc.txt`
   (see [Choosing a probability cutoff](#choosing-a-probability-cutoff)).

   **Resuming:** if training is interrupted, just run the same command
   again - it restores the last checkpoint's *weights* automatically. Note
   that microWakeWord's own training script always re-runs the full
   `training_steps` budget from step 0 on resume (it doesn't remember how
   many steps it had already done), so a resumed run isn't "picking up
   where it left off" step-count-wise, even though it's warm-started from
   good weights. If the model already looks converged (check the
   `Step N (nonstreaming): Validation: ...` log lines - `average_viable_recall`
   near 1.0 with 100% precision is a good sign), you can skip straight to
   converting the existing best checkpoint instead of retraining:

   ```
   python -m trainer.local.cli train --no-train
   ```

7. **Export for ESPHome:**

   ```
   python -m trainer.local.cli export --probability-cutoff 0.9
   ```

   Writes `workspace/exported_models/<name>/<name>.tflite` and a matching
   `.json` V2 model manifest (see the
   [ESPHome docs](https://esphome.io/components/micro_wake_word) and
   [example manifests](https://github.com/esphome/micro-wake-word-models/tree/main/models/v2)).
   Flags: `--probability-cutoff` (default 0.5), `--tensor-arena-size`
   (default 30000), `--author`, `--website`. Raise `--tensor-arena-size` if
   ESPHome fails to load the model.

Steps 3, 5, and 6 can also be run together once setup/samples are done:

```
python -m trainer.local.cli all
```

## Multiple phonetic spellings

Add alternate phonetic spellings of the same wake word to broaden the
model's coverage of how it might actually be pronounced:

```yaml
wake_word:
  phonetic: "hey wabi"
  friendly_name: "hey_wabi"
  additional_phonetics:
    - "hey wahbee"
    - "hey wah-bee"
```

`generate-samples` generates a full `piper.max_samples`-sized batch for
each spelling, into its own subdirectory under `generated_samples/<name>/`.
`build-features` picks up every spelling's samples automatically. In
testing, adding a second spelling to an already-good model did lower the
best achievable false-rejection-rate-at-zero-false-alarms slightly (broader
phonetic coverage makes the classification task harder) - this is a real
accuracy/robustness tradeoff, not a bug, so check
`tflite_streaming_roc.txt` after retraining rather than assuming more
spellings is strictly better for your specific wake word.

## Choosing a probability cutoff

After `train`, `workspace/trained_models/<name>/tflite_stream_state_internal_quant/tflite_streaming_roc.txt`
lists, for a range of cutoffs: `frr` (false-rejection rate - how often a
real wake word utterance is missed) and `faph` (estimated false alarms per
hour on ambient/negative audio). For a voice assistant wake word, false
alarms are usually more annoying than an occasional missed trigger, so
picking the lowest cutoff with `faph=0.000` is a reasonable default -
that's what `export`'s `--probability-cutoff` flag should be set to. Lower
the cutoff (accepting some false alarms) if it feels unresponsive in
practice; raise it if it triggers on background noise or speech.

## Troubleshooting / why does `setup` do all that?

Several dependency issues showed up getting this running on Windows that
you likely won't hit again since `setup_env.py` already works around them,
but are worth knowing about if something still breaks:

- **`ModuleNotFoundError: No module named 'torchcodec'` /
  `ImportError: ... please install 'torchcodec'`**: newer `datasets`
  library versions require `torchcodec` (which itself needs a matching
  system FFmpeg install) just to decode/encode audio, even for local files.
  We pin `datasets<4.0` and don't use `datasets` at all in
  `download_datasets.py`, specifically to avoid this. If you see this
  error, something reinstalled a newer `datasets` - re-run
  `pip install "datasets<4.0"`.
- **`ModuleNotFoundError: No module named 'piper_train'`**: piper-sample-generator's
  PyPI package only ships the `piper_sample_generator` module, not the
  sibling `piper_train` module its multi-speaker generator code path
  imports. `setup` clones the full repo instead of just pip-installing the
  package, and `generate_samples.py` puts the repo root on `PYTHONPATH`.
- **`AttributeError: module 'audiomentations' has no attribute 'AddColorNoise'`**:
  piper-sample-generator pins `audiomentations==0.33.0`, which predates
  that transform, but microWakeWord's augmentation needs it. `setup`
  installs a newer `audiomentations` afterward, overriding that pin (safe -
  we never use piper-sample-generator's own `.augment` module).
- **`ModuleNotFoundError: No module named 'pkg_resources'`**: `webrtcvad`
  still does `import pkg_resources` at load time; `setuptools>=81` removed
  that shim. `setup` pins `setuptools<81`.
- **`tensorflow.python.summary.tb_summary.TBNotInstalledError`**:
  microWakeWord's training script logs metrics via `tf.summary.scalar`,
  which needs `tensorboard` installed even though nothing views the logs.
  `setup` installs it.
- **A rebuilt `build-features` split turns out empty**: `RaggedMmap.from_generator`
  creates its output directory before writing any data, so a crash
  mid-build can leave an empty-but-existing directory. `build_features.py`
  checks for the actual data file, not just the directory, before deciding
  a split is already done - if you're on an older checkout without that
  fix, delete `workspace/generated_augmented_features/<name>/<split>/`
  and re-run.
