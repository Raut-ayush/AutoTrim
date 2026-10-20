# Product Requirements Document

## Original idea

AutoTrim was built to reduce the manual work of turning long-form video content into multiple written content formats. The underlying goal is to take one source video and convert it into reusable, platform-specific material without repeating the same work for every post, article, or thread.

This is an inferred product direction from the code and Git history, not a formally written product brief. The repo does not include a separate PRD, so this reflects the strongest evidence from the implementation and recent commit messages.

## Problem to solve

Creators and knowledge workers often produce long videos but then manually recreate the same information for:

- X/Twitter threads
- Reddit posts
- Substack-style long-form articles
- short-form content candidates

This creates duplication, slows publishing, and makes it harder to reuse valuable information from existing video content.

## Target users

- Solo creators and YouTubers
- Educators and explainers creators
- Researchers or communicators who want one source transcript turned into multiple outputs
- Developers or hobbyists building content pipelines locally with AI tools

## Must-have features

1. Accept a YouTube URL and identify the video.
2. Download the audio for the video.
3. Transcribe audio with WhisperX.
4. Align words and generate readable transcripts with timestamps.
5. Save transcript output in multiple formats: TXT, SRT, and JSON.
6. Reuse transcript output when rerunning the pipeline to avoid redundant work.
7. Generate repurposed written content from transcript text for:
   - one detailed Substack post
8. Support a resumable pipeline that resumes after a failure or from a prior complete stage.
9. Support multiple LLM backends: Gemini and Ollama.
10. Record benchmark and performance metadata for each run.

## Non-goals

- Full web application or SaaS interface
- Large-scale distributed processing
- Multi-user publishing system
- CMS integration or scheduler
- Full moderation or editorial approval system
- Browser UI or dashboard
- A polished commercial product experience

## Known pivots and adjustments

- The codebase originally included a blog-oriented repurposing path and later was updated to prioritize Substack-style output, as shown by the Git history entry "Replaced Blog with Substack".
- The project evolved from a simple transcription script toward a more deliberate content-engineering pipeline, adding a structured content analysis layer in `content_formatter.py`.
- The publishing pipeline now focuses on a detailed Substack article; X and Reddit generation were removed so output can focus on one in-depth piece.
- The architecture favors local execution and script control over hosted service orchestration, which implies a DIY content pipeline mindset rather than a managed platform model.

## Acceptance criteria (inferred)

The project is considered successful if:

- a valid YouTube URL produces a transcript and detailed Substack article,
- reruns do not duplicate prior work unnecessarily,
- outputs are stored in a predictable, per-video folder structure,
- the generated writing remains grounded in the transcript and avoids unsupported claims,
- the project can be run on a local machine with Python and a compatible ML environment.

## Risks and caveats

- Dependence on heavy ML libraries and model downloads can make setup fragile.
- LLM output quality depends on the underlying model and prompt design.
- Transcription quality varies with audio quality and noise.
- The pipeline is intentionally script-based, so operational reliability depends on local environment health rather than cloud infrastructure.

## Summary

The project intends to automate the conversion of long-form video into publishable written content in a way that is repeatable, resumable, and grounded in the transcript. It is best understood as a local AI content workflow tool, not a large SaaS platform.
