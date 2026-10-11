# Changes log

## 2026-10-11 — Setup and runtime reliability

- Added pinned direct dependencies and setup instructions for Python 3.13, FFmpeg, PyTorch CPU/CUDA selection, and Gemini credentials.
- Added shared configuration for model defaults, generation settings, and project-rooted output paths.
- Added early FFmpeg and Gemini key checks, clearer Ollama service checks, and shared CLI exit codes.
- Kept model downloads and network failures attached to their failed pipeline step and transcription log.

## 2026-10-11 — Documentation baseline created for the project

- Added the initial documentation set under `docs/` for project overview, product intent, architecture, project state, tasks, technical decisions, and change history.
- Synthesized the project description from the repo’s code, Git history, and execution flow rather than from a separate product brief.
- Recorded the current state as a functional prototype with clear remaining gaps in packaging, tests, and operational robustness.
