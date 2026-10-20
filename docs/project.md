# AutoTrim

AutoTrim is a Python-based workflow that turns a YouTube video into transcript files, benchmark metadata, and a detailed Substack article. The project is built as a resumable pipeline so it can skip work that has already been done and continue from the latest valid stage.

## Key stats

- Primary language: Python
- Core modules: 9 Python source files
- Git history: 6 commits, active on the `main` branch
- Current focus: video transcription, transcript cleanup, and content repurposing
- External dependencies: WhisperX, yt-dlp, PyTorch, Google GenAI SDK, and optional local Ollama

## Documentation index

- [PRD.md](./PRD.md) — original intent, users, features, and scope
- [architecture.md](./architecture.md) — current system design and execution flow
- [state.md](./state.md) — current implementation status and remaining gaps
- [tasks.md](./tasks.md) — concrete work checklist
- [decisions.md](./decisions.md) — technical decisions and rationale
- [changes.md](./changes.md) — changelog of project evolution

## Short summary

This project sits in the creator-automation space: it takes a source video, downloads the audio, transcribes it with WhisperX, stores structured metadata, and then uses an LLM to create one detailed Substack article from the transcript. The code is intentionally script-first and workflow-oriented rather than web-app-oriented, which makes it easy to run locally and reuse between videos.
