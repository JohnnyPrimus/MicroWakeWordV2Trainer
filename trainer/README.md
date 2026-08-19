# Local microWakeWord trainer

A local, script-driven alternative to `trainer/microwakeword_trainer_colab.ipynb`.
Same underlying pipeline (Piper-synthesized wake word samples, augmented with
background noise/RIRs, converted to spectrogram features, trained with
[microWakeWord](https://github.com/kahrendt/microWakeWord)) but as plain
Python you run on your own machine instead of a Colab notebook.

It also fixes a bug in the baseline notebook: the notebook's training-config
cell references a `generated_features_output_dir` variable that is never
defined anywhere - the cell that builds the actual spectrogram features from
the augmented clips is missing. `trainer/local/build_features.py` (the
`build-features` command below) is that missing step, ported from
microWakeWord's own upstream `notebooks/basic_training_notebook.ipynb`.

## Before you start: GPU note (Windows)

TensorFlow dropped native GPU support on Windows after 2.10, and
microWakeWord requires `tensorflow>=2.18`. On native Windows this pipeline
will train on **CPU only** - it'll work, just slowly. For real GPU-accelerated
training on an NVIDIA card, run this from inside **WSL2** (Ubuntu) instead,
where TensorFlow can use CUDA normally. Everything below works identically
in WSL2.

## Setup

```
pip install -r trainer/requirements.txt
cp trainer/wakeword_config.example.yaml trainer/wakeword_config.yaml
```

Edit `trainer/wakeword_config.yaml`: at minimum set `wake_word.phonetic` and
`wake_word.friendly_name`. See the comments in
`trainer/wakeword_config.example.yaml` for what everything else does.

All commands below default to `--config trainer/wakeword_config.yaml`; pass
`--config <path>` to use a different file (e.g. to train several wake words
side by side - each config's `workspace` directory is independent).

```
python -m trainer.local.cli setup
```

Clones `microWakeWord` and installs it editable, installs the
`piper-sample-generator` package, and downloads the Piper voice checkpoint
used to synthesize samples. Only needs to run once per workspace.

## Pipeline

Run these in order. Each step skips work it's already done (safe to re-run
or resume after an interruption).

1. **Check the phonetic spelling** by generating one sample and listening to it:

   ```
   python -m trainer.local.cli preview-word
   ```

   Listen to `workspace/generated_samples/<friendly_name>/0.wav`. Adjust
   `wake_word.phonetic` in the config until it sounds right, then re-run.

2. **Generate the full sample batch:**

   ```
   python -m trainer.local.cli generate-samples
   ```

   Controlled by `piper.max_samples` / `piper.batch_size` in the config.
   Start here when trying to improve a model - more samples, more phonetic
   variants, or different `length_scales`/`noise_scales` all help.

3. **Download background/negative audio data** (RIRs, Audioset, FMA, and
   pre-generated negative spectrogram sets). This is the slowest step -
   expect 30-45+ minutes and several GB of disk:

   ```
   python -m trainer.local.cli download-data
   ```

   > **License note:** this data mixes several licenses/usage restrictions.
   > Treat models trained with it as personal-use only unless you check the
   > license of each source for commercial use.

4. **(Optional) Preview an augmented clip:**

   ```
   python -m trainer.local.cli preview-augment
   ```

   Writes `augmented_clip_preview.wav` in the current directory.

5. **Build spectrogram features** (augments samples, builds the
   training/validation/testing mmaps):

   ```
   python -m trainer.local.cli build-features
   ```

6. **Train:**

   ```
   python -m trainer.local.cli train
   ```

   Hyperparameters live under `training:` in the config (training steps,
   class weights, learning rate, model architecture args, etc.) - these
   matter a lot for model quality, especially for very short or long wake
   words. This step also quantizes and converts the trained model to a
   streaming `.tflite`.

7. **Export for ESPHome:**

   ```
   python -m trainer.local.cli export
   ```

   Writes `workspace/exported_models/<friendly_name>/<friendly_name>.tflite`
   and a matching `.json` V2 model manifest (see the
   [ESPHome docs](https://esphome.io/components/micro_wake_word) and
   [example manifests](https://github.com/esphome/micro-wake-word-models/tree/main/models/v2)).
   `probability_cutoff` and `tensor_arena_size` in the manifest are starting
   guesses - test on-device and adjust: raise `probability_cutoff` if it
   triggers on background noise, raise `tensor_arena_size` if ESPHome fails
   to load the model.

Steps 3, 5, and 6 can also be run together once setup/samples are done:

```
python -m trainer.local.cli all
```
