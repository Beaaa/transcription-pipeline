# 🎙️ Transcription Pipeline

Complete pipeline for transcribing long audio recordings (meetings, interviews, consultations) with **speaker diarization**, voice isolation, and high-quality transcription.

Ideal for work meetings, research interviews, professional consultations, and any audio with multiple participants where you need to know who said what.

---

## ✨ Features

- 🎛️ **Voice isolation** with [Demucs](https://github.com/facebookresearch/demucs) — separates human voice from background noise, music, and ambient sounds
- 👥 **Speaker diarization** with [pyannote.audio](https://github.com/pyannote/pyannote-audio) — identifies who speaks when
- 📝 **Transcription** with [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text) (`whisper-1` model) — excellent quality across many languages, including Portuguese (BR)
- ⚡ **GPU acceleration** (CUDA) — 1 hour of audio processed in ~10 minutes
- 🔁 **Resumable execution** — the `--resume` flag skips stages already completed
- 📂 **Automatic archiving** — processed audio files are moved to `already_transcripted/`
- 🐍 **Managed with [uv](https://github.com/astral-sh/uv)** — one-command setup

---

## 🏗️ Architecture

```
┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Audio file   │ →  │ ffmpeg      │ →  │ Demucs       │ →  │ pyannote     │ →  │ Whisper API  │
│ (m4a/mp3/…)  │    │ WAV 16kHz   │    │ Vocal stem   │    │ Diarization  │    │ Transcription│
└──────────────┘    └─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    └─────────┬─────────┘
                                                                              ↓
                                                                    ┌─────────────────────┐
                                                                    │ Timestamp merge     │
                                                                    │ → final transcript  │
                                                                    └─────────────────────┘
```

---

## 📋 Prerequisites

### System

| Tool       | Version | Link |
|------------|---------|------|
| ffmpeg     | recent  | [gyan.dev builds](https://www.gyan.dev/ffmpeg/builds/) |
| uv         | latest  | [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| NVIDIA GPU | ≥ 4GB VRAM (recommended) | - |

> 💡 **Works without a GPU**, but Demucs becomes impractically slow on CPU (30+ min per hour of audio). Tested and validated on RTX 3060 Laptop (6GB VRAM).

### Accounts and credentials

| Service       | Purpose                     | Cost                        |
|---------------|-----------------------------|-----------------------------|
| OpenAI        | Whisper API                 | ~US$ 0.36 per hour of audio |
| Hugging Face  | pyannote models             | Free                        |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/transcription-pipeline.git
cd transcription-pipeline
```

### 2. Install ffmpeg

**Windows:** Download the `release full` from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), extract to `C:\ffmpeg\`, and add `C:\ffmpeg\bin` to your PATH.

**Linux:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

Verify with `ffmpeg -version` in a **new** terminal.

### 3. Install uv

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 4. Install dependencies

From the project folder:

```bash
uv sync
```

`uv` handles Python installation, virtual environment creation, and PyTorch with CUDA automatically. First run takes ~5 min (~2.5GB download).

### 5. Set up accounts

**OpenAI:**
1. Create an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Add credit at [platform.openai.com/settings/organization/billing](https://platform.openai.com/settings/organization/billing) — US$ 5 covers ~14 hours of transcription

**Hugging Face:**
1. Create an account at [huggingface.co/join](https://huggingface.co/join)
2. **Accept the terms of use** for both models below (required for pyannote to work):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Generate a token with `Read` permission at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 6. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
OPENAI_API_KEY=sk-proj-...
HF_TOKEN=hf_...
```

### 7. Verify installation

```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Should return `CUDA: True` along with your GPU name.

---

## 🎬 Usage

### Basic usage

Place your audio file in `audio_original/` and run:

```bash
uv run python pipeline.py audio_original/my_meeting.m4a --num-speakers 3
```

Supported formats: `.m4a`, `.mp3`, `.wav`, `.mp4`, `.ogg`, `.flac` (anything ffmpeg can read).

### Available flags

| Flag                  | Description                                                                    |
|-----------------------|--------------------------------------------------------------------------------|
| `--num-speakers N`    | Forces pyannote to identify exactly N speakers. Omit for automatic detection.  |
| `--skip-denoise`      | Skips the Demucs step (faster, lower quality)                                  |
| `--resume`            | Skips steps whose output files already exist (useful after interruptions)      |

### Examples

```bash
# 3-person meeting
uv run python pipeline.py audio_original/standup.m4a --num-speakers 3

# 1-on-1 interview, no denoise
uv run python pipeline.py audio_original/interview.mp3 --num-speakers 2 --skip-denoise

# Resume an execution that crashed at step 3
uv run python pipeline.py audio_original/lecture.wav --num-speakers 1 --resume
```

---

## 📂 Folder structure

```
transcription-pipeline/
├── pipeline.py                 # Main script
├── pyproject.toml              # Dependencies (uv)
├── .env                        # Your credentials (gitignored)
├── .env.example                # Credentials template
│
├── audio_original/             # ← Drop audio files to process here
├── already_transcripted/       # ← Audio files moved here after processing
│
└── output/
    └── <audio_name>/
        ├── 01_converted.wav           # Audio converted to WAV 16kHz mono
        ├── 02_denoised.wav            # Vocal stem isolated by Demucs
        ├── 03_diarization.rttm        # Diarization in standard RTTM format
        ├── 04_transcription.json      # Raw Whisper transcription
        ├── 05_merged.json             # Diarization + transcription merged
        └── 06_transcricao_final.txt   # ✨ Final human-readable transcript
```

### Sample output

```
[00:00:02] SPEAKER_00:
  Good morning everyone, let's get started with the meeting.
  Any urgent items before we dive in?

[00:00:15] SPEAKER_01:
  I have a question about yesterday's deploy.

[00:00:19] SPEAKER_00:
  Go ahead.
```

---

## ⏱️ Performance

Benchmarked on **RTX 3060 Laptop (6GB VRAM)** with a 1-hour audio file:

| Step          | Time       |
|---------------|------------|
| Conversion    | ~10s       |
| Demucs        | ~2 min     |
| Diarization   | ~3 min     |
| Whisper API   | ~2 min     |
| **Total**     | **~7 min** |

**Cost per hour of audio:** ~US$ 0.36 (Whisper API only).

---

## 🛠️ Useful uv commands

```bash
# Add a dependency
uv add library-name

# Remove a dependency
uv remove library-name

# Upgrade all dependencies
uv sync --upgrade

# Run any script inside the env
uv run python <script.py>
```

---

## 🐛 Troubleshooting

<details>
<summary><b>401 Unauthorized when downloading pyannote models</b></summary>

You haven't accepted the terms of use for the Hugging Face models. Access **with the same account as the token in your `.env`**:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

Click "Agree and access repository" on both.
</details>

<details>
<summary><b>TypeError: Pipeline.from_pretrained() got an unexpected keyword argument 'token'</b></summary>

Version mismatch between `pyannote.audio` and `huggingface_hub`. Fix:

```bash
uv add "huggingface-hub<0.28"
```
</details>

<details>
<summary><b>CUDA: False on verification</b></summary>

PyTorch was installed without CUDA support. Ensure your `pyproject.toml` includes the `[[tool.uv.index]]` block pointing to `pytorch-cu121`, then run:

```bash
uv sync --reinstall
```
</details>

<details>
<summary><b>Speakers swapped or incorrect in the output</b></summary>

Diarization isn't perfect, especially with similar-sounding voices. Open `06_transcricao_final.txt`, listen to a few segments, and find-replace the generic labels (`SPEAKER_00` → "Alice", etc.).
</details>

<details>
<summary><b>CUDA out of memory</b></summary>

Audio too long or GPU with limited VRAM. Try `--skip-denoise` or split the audio into smaller chunks with ffmpeg before processing.
</details>

---

## 📜 License

MIT — use freely.

---

## 🙋 Credits

Built by **[Beatriz Alves Silva](https://github.com/YOUR_USERNAME)** with Claude's help.

Pipeline powered by amazing open-source projects:
- [Demucs](https://github.com/facebookresearch/demucs) (Meta AI Research)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Whisper](https://openai.com/research/whisper) (OpenAI)
- [uv](https://github.com/astral-sh/uv) (Astral)