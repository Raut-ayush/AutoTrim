# Architecture

## Overview

AutoTrim is implemented as a modular Python pipeline. The orchestration flow is centered on `autotrim.py`, which coordinates the stages of:

1. identifying a video,
2. checking whether transcript outputs already exist,
3. transcribing audio with WhisperX,
4. generating repurposed content with an LLM,
5. storing results in a per-video output directory.

## Tech stack

- Python 3 for orchestration and scripts
- `yt-dlp` for metadata and audio extraction
- WhisperX for transcription and word alignment
- PyTorch and CUDA-capable runtime when available
- Google GenAI SDK for Gemini access
- Local HTTP API client for Ollama
- Pydantic models for structured LLM output validation
- dotenv-based environment loading for API keys

## Core modules

### `autotrim.py`

This is the main entry point and orchestrator. It is responsible for:

- parsing CLI arguments,
- extracting video IDs from URL formats,
- checking whether a transcript already exists,
- invoking `transcribe.py` when needed,
- invoking `repurposer.py` when needed,
- printing a concise summary of final outputs.

The design is intentionally resumable and deterministic: it tries to reuse transcript and repurposed content instead of regenerating them each run.

### `transcribe.py`

This module performs the end-to-end transcription pipeline:

- fetches YouTube metadata,
- downloads the audio stream,
- loads the WhisperX model,
- transcribes and aligns segments,
- saves transcript.txt, transcript.srt, and transcript.json,
- records benchmark metrics in JSON and CSV logs.

It includes logic for:

- duplicate output folder handling,
- system metadata capture,
- GPU memory tracking,
- paragraph merging from raw ASR segments,
- benchmark output for analysis and debugging.

### `repurposer.py`

This module transforms a transcript into publication-oriented content. It uses a prompt-driven LLM process and expects the response to follow a strict structured schema.

The module creates one UTF-8 Substack article and a structured JSON file used by the dashboard:

- `substack_post.txt` for a Substack title and long-form content,
- `repurposed.json` for the article title and body used in dashboard previews.

The prompt emphasizes source fidelity, human-sounding writing, and avoidance of unsupported facts.

### `content_formatter.py`

This file is a content intelligence layer. It takes transcript JSON and asks the model to identify:

- main topic and subtopics,
- important claims and supporting text,
- key quotes,
- content angles,
- candidate short-form segments.

The system prompt demands strict JSON-only output and validates timestamps against transcript data.

### `llm/gemini_provider.py`

Provides a `GeminiProvider` with retry logic for transient API errors such as 429 and 503 responses. It loads the API key from the environment and wraps the Google GenAI SDK.

### `llm/ollama_provider.py`

Provides a lightweight Ollama provider that talks to the local Ollama HTTP API. This allows the project to run without a hosted Gemini dependency when a local model is available.

### `ui/`

The local browser dashboard is an optional loopback-only front end over the existing CLI pipeline. `ui/server.py` serves the static interface and JSON endpoints; `ui/jobs.py` persists each job's `status.json` and streamed `log.txt`, and runs one child process at a time. It supports full video jobs and generation-only runs from an existing transcript. Runtime state stays in the ignored `ui/data/` folder.

The interface is plain HTML/CSS/JavaScript and uses Python's standard library for its HTTP server. It samples AutoTrim's process tree through `psutil` and reads optional NVIDIA telemetry from `nvidia-smi`. Dashboard-selected output subfolders and transcript regeneration are confined to the configured output directory.

## Data flow

1. User provides a YouTube URL.
2. `autotrim.py` extracts the video ID and checks for existing output.
3. If no transcript exists, `transcribe.py` downloads the audio and runs WhisperX.
4. Transcript results are saved as JSON, TXT, and SRT.
5. `autotrim.py` checks whether repurposed content already exists.
6. If missing, `repurposer.py` reads the transcript text and asks an LLM to generate structured content.
7. The Substack text file and structured JSON are written under the per-video `content/` subfolder.
8. Benchmark logs capture runtime, model, duration, and transcript statistics.

## Design decisions

- Script-first over framework-first: no server or app skeleton; the workflow is a command-line pipeline.
- Resumable operation: transcript and content generation are skipped when valid outputs already exist.
- Source fidelity over creativity: repurposing prompts explicitly reject unsupported claims and outside knowledge.
- Multi-provider LLM support: Gemini and Ollama both fit the project’s local and API-driven use cases.
- Structured output schema: Pydantic validation and strict JSON output reduce response drift.

## Current architectural strengths

- Clear separation between orchestration, transcription, and content generation.
- Easy to extend with more providers or formats.
- Output directories are intuitive and portable.
- Benchmark logs support monitoring and tuning.

## Current architectural weaknesses

- Dependencies are pinned in `requirements.txt`, while PyTorch CPU/CUDA wheel selection remains a separate platform-specific install step.
- `config.py` centralizes model defaults, generation settings, and repository-rooted output paths.
- CLI exit codes are standardized and checks catch missing FFmpeg, missing Gemini credentials, and unavailable Ollama before the relevant stage.
- Partial downloads and interrupted writes still need stronger recovery and cleanup handling.
- The repo is still primarily a local script project, not a distribution-ready application.
