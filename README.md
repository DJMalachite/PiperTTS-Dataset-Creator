
# PiperTTS Dataset Creator

A Python tool to automatically generate a [PiperTTS](https://github.com/rhasspy/piper/blob/master/TRAINING.md) training dataset from long audio recordings. It handles audio chunking, transcription, format conversion, and metadata generation in a single run.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Python Dependencies](#python-dependencies)
  - [System Dependencies](#system-dependencies)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

---

## Features

- Automatically splits long audio files into smaller, manageable chunks
- Transcribes audio using Whisper
- Converts audio to `.wav` format compatible with PiperTTS
- Generates the necessary `metadata.csv` and organizes outputs in the `wavs/` folder

---

## Installation

### Python Dependencies

Ensure you're using **Python 3.12**, then install required Python packages:

```bash
pip install -r requirements.txt
```

### System Dependencies

Install `ffmpeg` for audio processing:

```bash
sudo apt install ffmpeg
```

---

## Configuration

Before running the script, configure the following parameters within the script file (`.py`):

```python
# ========== Configuration ==========
INPUT_EXTENSION = ".wav"  # change to ".wav" or ".mp3" as needed
MIN_SILENCE_LEN = 500     # in ms
SILENCE_THRESH = -40      # in dBFS
KEEP_SILENCE = 250        # in ms
WHISPER_MODEL_SIZE = "turbo" # Set the OpenAI Whisper model, examples can be found here: https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
WHISPER_DEVICE = "cuda" # Set between CPU or CUDA for GPU accelleration
# ===================================
```

---

## Usage

Place your input audio file in the same directory as the scrupt, then run:

```bash
python3 your_script_name.py
```

The output will include:

- A `wavs/` directory containing chunked `.wav` files
- A `metadata.csv` file with corresponding transcriptions

---

## Examples

Coming soon.

---

## Troubleshooting

- **Audio not splitting correctly**: Adjust `MIN_SILENCE_LEN` and `SILENCE_THRESH`.
- **Transcriptions missing or inaccurate**: Try changing the `WHISPER_MODEL_SIZE` to a more accurate model.
- **Permission issues with `ffmpeg`**: Ensure `ffmpeg` is installed and available in your system's PATH.

---

## Contributors

- DJMalachite

---

