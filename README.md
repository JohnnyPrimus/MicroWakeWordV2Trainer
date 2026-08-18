# MicroWakeWordV2Trainer

Trains custom wake word models for ESP32-based Home Assistant satellites using a Jupyter notebook pipeline.

## Features

- **Piper TTS** — generates synthetic training samples across multiple voices with speed variation
- **Wyoming protocol** — includes a ready-to-run server for Home Assistant voice stack integration
- **INT8 quantization** — full integer quantization for efficient ESP32 deployment via ESPHome
- **Audio augmentation** — noise, room simulation, pitch shift, EQ via `audiomentations`
- **Two architectures** — Inception or MixedNet (depthwise separable convolutions)
- **Auto-evaluation** — ROC curve, confusion matrix, and optimal threshold suggestion

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Open the notebook

```bash
jupyter notebook MicroWakeWordV2_Trainer.ipynb
```

### 3. Configure your wake word

Edit the `TrainingConfig` dataclass in **Section 1** of the notebook:

```python
cfg = TrainingConfig(
    wake_word="hey computer",
    piper_voices=["en_US-lessac-medium", "en_US-ryan-high"],
    positive_samples_per_voice=500,
)
```

### 4. Run all cells

The notebook will:
1. Download Piper voice models automatically
2. Generate thousands of positive and negative audio samples
3. Augment samples with realistic noise and room acoustics
4. Train and quantize the model to INT8 TFLite
5. Export `model.tflite`, `manifest.json`, and an ESPHome YAML snippet

## Output Files

| File | Description |
|------|-------------|
| `output/model.tflite` | INT8 quantized model for ESP32 |
| `output/model_float32.tflite` | Float32 model for Wyoming server |
| `output/manifest.json` | ESPHome `micro_wake_word` manifest |
| `output/esphome_snippet.yaml` | ESPHome configuration snippet |
| `output/wyoming_config.yaml` | Wyoming server configuration |
| `wyoming_server.py` | Wyoming protocol server script |

## ESPHome Deployment

Copy `model.tflite` and `manifest.json` to a web server, then add to your ESPHome config:

```yaml
micro_wake_word:
  on_wake_word_detected:
    - logger.log: "Wake word detected!"
  models:
    - model: http://YOUR_SERVER/hey_computer/manifest.json
```

## Wyoming Server (Home Assistant)

Run the Wyoming server for server-side detection:

```bash
python wyoming_server.py --config output/wyoming_config.yaml
```

Then add a Wyoming integration in Home Assistant pointing to `<host>:10400`.

## Requirements

- Python 3.10+
- TensorFlow 2.13+
- `piper` CLI (installed via `piper-tts` package)
- ~2 GB disk space for MUSAN noise dataset
