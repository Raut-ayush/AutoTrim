# Decisions log

## 2026-10-11 — Python-first local automation pipeline

**Context**: The repository is a script-driven project focused on downloading video audio, transcribing it, and turning the transcript into multiple written outputs. There is no web app, API server, or framework-based UI.

**Decision**: Keep the implementation in Python and structure it as a sequence of CLI-oriented modules rather than a web service.

**Why**: This matches the project’s practical goal: local content processing with direct control over transcription, model usage, and reusable output artifacts.

## 2026-10-11 — WhisperX as the transcription engine

**Context**: The code downloads audio and then performs ASR and alignment. The repository also includes benchmark metadata and transcript timing details that indicate a focus on accuracy and cleanup.

**Decision**: Use WhisperX for transcription and word alignment.

**Why**: It offers a strong local transcription workflow and integrates well with timestamp-aware transcript analysis, which is important for downstream content repurposing.

## 2026-10-11 — Gemini + optional Ollama support

**Context**: The project needs content generation from transcripts but should support both hosted and local model workflows.

**Decision**: Support Gemini via the Google GenAI SDK and allow a local Ollama backend as an alternative.

**Why**: This supports both convenience and flexibility, while keeping the project usable in environments with or without a hosted API key.

## 2026-10-11 — Resumable output flow

**Context**: The workflow is expensive in time and compute, especially for transcription and AI generation.

**Decision**: Reuse existing transcript and content outputs when valid files are already present.

**Why**: This makes repeated runs efficient and reduces wasted cost and processing time.

## 2026-10-11 — Source fidelity over creative rewriting

**Context**: Generated content is derived from transcripts, which can be noisy or imperfect.

**Decision**: Prompt engineering emphasizes factual grounding, preserving transcript meaning, and not inventing unsupported details.

**Why**: This is essential for a creator workflow that repurposes educational or explanatory content without drifting into hallucination.
