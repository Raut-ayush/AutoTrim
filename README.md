# AutoTrim

AutoTrim turns a YouTube video into transcript files and one detailed Substack article. It is a local command-line workflow with resumable outputs.

## Requirements

- Python **3.13** (the pinned versions below were collected from the project's Python 3.13.11 environment).
- FFmpeg available on `PATH`; yt-dlp uses it to extract downloaded audio as WAV.
- A Gemini API key for the default `autotrim.py` pipeline. Ollama can be used with `content_formatter.py` as an alternative for content analysis.
- Internet access for YouTube, model downloads, and Gemini. Transcription itself runs locally.

## Installation

Create and activate a virtual environment from the repository directory:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch and Torchaudio **first**, using the official PyTorch instructions for your platform and either CPU or the CUDA version supported by your NVIDIA driver. AutoTrim's environment used PyTorch/Torchaudio **2.8.0+cu128** (CUDA 12.8). CPU-only installs are supported; WhisperX selects CPU automatically when CUDA is unavailable, using `int8` compute instead of GPU `float16`. CPU transcription is slower, and larger models need more memory. No GPU is required.

Then install the pinned application dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install FFmpeg and make its `bin` directory available on `PATH`. On Windows, for example, install it using `winget install Gyan.FFmpeg` or `choco install ffmpeg`, then open a new terminal. Verify with `ffmpeg -version`.

Create a local `.env` file and set your Gemini API key:

```powershell
Copy-Item .env.example .env
```

Replace the placeholder in `.env` with the key from Google AI Studio. Keep `.env` private; it is excluded by `.gitignore`. Alternatively, set `GEMINI_API_KEY` in the process environment. A key is only required when new Gemini content is generated; a run reusing existing content does not need to call Gemini.

## Run

Run the complete resumable workflow:

```powershell
python autotrim.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Transcribe only:

```powershell
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID" --model small --batch-size 4
```

Repurpose a transcript directly:

```powershell
python repurposer.py "output/VIDEO/transcript.txt"
```

For the Ollama-backed content formatter, start Ollama and pull the configured model first (default `qwen2.5-coder:7b`):

```powershell
ollama pull qwen2.5-coder:7b
python content_formatter.py "output/VIDEO/transcript.json" --provider ollama
```

## Local dashboard

Start the browser dashboard from the repository directory using the same virtual environment:

```powershell
python -m ui
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). It listens on loopback only and does not need another package. The dashboard can queue multiple video links, runs them one at a time, shows the current stage and streamed logs, and samples the AutoTrim process tree's CPU and memory use. NVIDIA GPU load and VRAM are shown when `nvidia-smi` is available; otherwise they are marked unavailable.

Paste up to 10 links (one per line) to queue several videos. Select an existing transcript in the library to compare it beside the generated Substack article, copy either text, or download the transcript and generated files. The detailed article is saved as `content/substack_post.txt`; `repurposed.json` is also saved for dashboard previews. **Regenerate from transcript** starts only the Gemini generation stage and reuses the selected `.txt` file. Generation model, temperature, retry count, WhisperX model, batch size, and an output subfolder under `output/` can be set per run.

Job status snapshots and logs are stored under `ui/data/` and excluded from Git. The current full publishing pipeline generates with Gemini; the separate Ollama content formatter remains available through its CLI.

Outputs, logs, and benchmark files are stored in the repository's `output/`, `logs/`, and `benchmark/` directories, regardless of the shell's current directory. WhisperX model weights are downloaded on first use. The default transcription model is `small`; first runs therefore need network access and enough disk space for model weights.

## Startup checks and exit codes

The orchestrator checks for FFmpeg before starting a new transcription and checks for `GEMINI_API_KEY` before starting new Gemini generation. Existing transcript/content artifacts are reused without requiring those stages' dependencies. `content_formatter.py` checks the local Ollama endpoint before generation. YouTube/network errors and model download failures are printed with the failed step and the transcription log path.

CLI exit codes are consistent across the workflow: `0` success, `2` command-line usage, `3` configuration (such as a missing API key), `4` missing local dependency (FFmpeg), `5` unavailable external service, `6` processing or output failure, and `130` user cancellation. Child-process exit codes propagate through `autotrim.py`.

## Configuration

Defaults are in [config.py](./config.py): WhisperX model and batch size, Gemini/Ollama model names, generation temperatures/retries, and output/log/benchmark paths. CLI flags and dashboard controls override defaults where supported. Paths are rooted at the project directory.

## Project documentation

See [docs/project.md](./docs/project.md), [docs/architecture.md](./docs/architecture.md), [docs/state.md](./docs/state.md), [docs/tasks.md](./docs/tasks.md), [docs/decisions.md](./docs/decisions.md), and [docs/changes.md](./docs/changes.md).
