# Changes log

## 2026-10-20 — Dashboard settings and dark mode

- Added a browser-persisted light/dark theme toggle.
- Added controls to save, replace, or remove `GEMINI_API_KEY` in the ignored local `.env` file; the API never returns the secret.
- Documented the dashboard settings and theme behavior.

## 2026-10-20 — Substack-only generation

- Replaced X and Reddit generation with a single detailed Substack article and transcript-length-based word targets.
- Kept the structured article JSON for dashboard previews and removed obsolete X/Reddit review controls.
- Detects previous multi-format output as stale and replaces it with Substack-only output on the next generation.
- Disabled unused automatic function calling for the Gemini request to avoid the SDK's AFC warning.

## 2026-10-20 — Text outputs and Windows console encoding

- Saved generated X, Reddit, and Substack content to UTF-8 `.txt` files while retaining `repurposed.json` for structured dashboard previews.
- Updated resumable content detection and final output paths to use the `.txt` files.
- Configured CLI stdout and stderr to use UTF-8 so Unicode transcription banners work in Windows consoles.

## 2026-10-11 — Local dashboard

- Added the loopback-only web dashboard in `ui/`, with a persistent sequential job runner, live stage/log updates, CPU/RAM sampling, and optional NVIDIA GPU telemetry.
- Added the existing transcript library, generation-only regeneration, side-by-side source/output review, Markdown display, X character warnings, copy controls, and artifact downloads.
- Exposed transcription model/batch size, Gemini model/temperature/retries, and an output subfolder per job; left prompt editing for a later iteration.
- Added multi-link queue submission with up to ten links at a time.
- Added CLI generation setting overrides and ignored runtime dashboard job state.

## 2026-10-11 — Setup and runtime reliability

- Added pinned direct dependencies and setup instructions for Python 3.13, FFmpeg, PyTorch CPU/CUDA selection, and Gemini credentials.
- Added shared configuration for model defaults, generation settings, and project-rooted output paths.
- Added early FFmpeg and Gemini key checks, clearer Ollama service checks, and shared CLI exit codes.
- Kept model downloads and network failures attached to their failed pipeline step and transcription log.

## 2026-10-11 — Documentation baseline created for the project

- Added the initial documentation set under `docs/` for project overview, product intent, architecture, project state, tasks, technical decisions, and change history.
- Synthesized the project description from the repo’s code, Git history, and execution flow rather than from a separate product brief.
- Recorded the current state as a functional prototype with clear remaining gaps in packaging, tests, and operational robustness.
