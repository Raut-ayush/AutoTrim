"""Generate one detailed Substack article from a transcript."""

import argparse
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config import (
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL,
    PROJECT_ROOT,
    REPURPOSER_MAX_RETRIES,
    REPURPOSER_TEMPERATURE,
)
from errors import ConfigurationError, ExitCode, ServiceError, TranscriptQualityError


load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = GEMINI_MODEL
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_OUTPUT_TOKENS = 8192
REPEATED_TOKEN_WARNING = 5
REPEATED_TOKEN_BLOCK = 8

LEGACY_OUTPUTS = (
    "x_thread.md",
    "x_thread.txt",
    "reddit_post.md",
    "reddit_post.txt",
)


class RepurposedContent(BaseModel):
    """The only generated fields needed by the publishing workflow."""

    substack_title: str = Field(
        description="A clear, compelling, natural Substack article title."
    )
    substack_content: str = Field(
        description="A detailed, source-grounded long-form Substack article in Markdown."
    )


SYSTEM_PROMPT = """
You are a careful editor and long-form writer. Turn the supplied video
transcript into one detailed, publication-ready Substack article.

The transcript is the sole factual source. Do not add outside knowledge,
facts, examples, quotations, causal claims, or conclusions. Preserve the
speaker's meaning, uncertainty, caveats, names, numbers, and technical terms.
Clean obvious speech-recognition errors only when the intended meaning is
clear; omit uncertain details rather than guessing.
Always write the article title and body in English. Translate Hindi or other
non-English transcript passages faithfully into English; preserve names and
technical terms accurately, and do not include untranslated passages unless
they are proper names or terms used by the speaker.

Write naturally and specifically. Avoid filler, repeated ideas, generic
introductions, clickbait, promotional language, invented personal experience,
and forced conclusions. Use a direct opening and Markdown headings only when
they help readers follow the discussion. Do not mention the transcript,
automatic generation, or the source video.
""".strip()


def target_word_range(transcript_word_count: int) -> tuple[int, int]:
    """Return a proportional article length target for the source length."""

    if transcript_word_count < 300:
        return 250, 400
    if transcript_word_count <= 2_000:
        return 500, 1_000
    if transcript_word_count <= 8_000:
        return 900, 1_800
    return 1_500, 2_500


def build_prompt(transcript: str) -> str:
    word_count = len(transcript.split())
    minimum, maximum = target_word_range(word_count)

    return f"""
Write ONE detailed Substack article based on the complete transcript below.

The transcript contains about {word_count:,} words. Aim for {minimum:,}–{maximum:,}
words in the article body. Treat this as a real target: develop the source's
main arguments and important supporting points, not just its headline ideas.

To make the piece detailed without padding:
- Explain the important ideas and how they relate to one another.
- Preserve meaningful examples, analogies, qualifications, disagreements,
  counterpoints, and unresolved questions that appear in the source.
- Give each major point enough explanation for a reader who has not seen the
  video, while staying within what the transcript supports.
- Do not repeat points or add material simply to reach the target length.
- Start with the strongest supported idea; do not add a generic introduction.
- Use a natural title and a coherent article body. Add useful headings only
  where they improve navigation.

Return the article title and body in the requested structured fields.

FULL TRANSCRIPT
===============
{transcript}
""".strip()


def create_client() -> genai.Client:
    if not os.getenv(GEMINI_API_KEY_ENV):
        raise ConfigurationError(
            f"{GEMINI_API_KEY_ENV} is not set. Add it to "
            f"{PROJECT_ROOT / '.env'} or your environment."
        )
    return genai.Client()


def is_non_retryable_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    patterns = (
        "quota exceeded",
        "generaterequestsperdayperprojectpermodel-freetier",
        "resource_exhausted",
        "exceeded your current quota",
        "daily quota",
        "billing details",
        "api key not valid",
        "invalid api key",
        "permission denied",
        "unauthenticated",
    )
    return any(pattern in error_text for pattern in patterns)


def validate_content(
    data: RepurposedContent,
    transcript_word_count: int,
) -> list[str]:
    issues = []
    if not data.substack_title.strip():
        issues.append("Substack title is empty.")
    if not data.substack_content.strip():
        issues.append("Substack article is empty.")

    minimum, maximum = target_word_range(transcript_word_count)
    article_words = len(data.substack_content.split())
    if article_words < minimum:
        issues.append(
            f"Substack article has {article_words:,} words; "
            f"the target is {minimum:,}–{maximum:,}."
        )
    return issues


def assess_transcript_quality(transcript: str) -> list[tuple[str, str]]:
    """Find clear text-corruption signals before sending a transcript to Gemini."""
    issues = []
    replacement_count = transcript.count("\ufffd")
    if replacement_count:
        issues.append((
            "blocking",
            f"Transcript contains {replacement_count} U+FFFD Unicode replacement character(s), "
            "which indicate text decoding or transcription corruption.",
        ))

    tokens = []
    for raw_token in transcript.split():
        normalized = unicodedata.normalize("NFKC", raw_token).casefold()
        token = "".join(
            char for char in normalized
            if not unicodedata.category(char).startswith(("P", "S"))
        )
        if token:
            tokens.append(token)

    longest_token = ""
    longest_run = 0
    current_token = ""
    current_run = 0
    for token in tokens:
        if token == current_token:
            current_run += 1
        else:
            current_token, current_run = token, 1
        if current_run > longest_run:
            longest_token, longest_run = current_token, current_run

    if longest_run >= REPEATED_TOKEN_BLOCK:
        issues.append((
            "blocking",
            f"Transcript repeats the same token {longest_run} times consecutively "
            f"({longest_token!r}), suggesting an ASR repetition loop.",
        ))
    elif longest_run >= REPEATED_TOKEN_WARNING:
        issues.append((
            "warning",
            f"Transcript repeats the same token {longest_run} times consecutively "
            f"({longest_token!r}); review for a possible ASR loop.",
        ))
    return issues


def build_generation_config(temperature: float) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=RepurposedContent,
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    )


def generate_content(
    transcript: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = REPURPOSER_MAX_RETRIES,
    temperature: float = REPURPOSER_TEMPERATURE,
) -> RepurposedContent:
    client = create_client()
    prompt = build_prompt(transcript)
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            print("[Gemini] Generating Substack article...")
            print(f"[Gemini] Model: {model}")
            start_time = time.perf_counter()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=build_generation_config(temperature),
            )
            print(
                f"[Gemini] Generation completed in "
                f"{time.perf_counter() - start_time:.2f}s"
            )

            if isinstance(response.parsed, RepurposedContent):
                return response.parsed
            if response.parsed is not None:
                return RepurposedContent.model_validate(response.parsed)
            if response.text:
                return RepurposedContent.model_validate_json(response.text)
            raise ValueError("Gemini returned an empty response.")

        except Exception as exc:
            last_error = exc
            print(f"[Gemini] Attempt {attempt}/{max_retries} failed: {exc}")
            if is_non_retryable_error(exc):
                break
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"[Gemini] Retrying in {wait_time}s...")
                time.sleep(wait_time)

    raise ServiceError(f"Gemini generation failed: {last_error}") from last_error


def save_outputs(
    data: RepurposedContent,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "substack_post.txt"
    json_path = output_dir / "repurposed.json"

    text_path.write_text(
        f"# {data.substack_title}\n\n{data.substack_content}\n",
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(data.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for name in LEGACY_OUTPUTS:
        legacy_path = output_dir / name
        try:
            legacy_path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[Output] Could not remove old {name}: {exc}")

    return text_path, json_path


def repurpose_transcript(
    transcript_path: str,
    model: str = DEFAULT_MODEL,
    temperature: float = REPURPOSER_TEMPERATURE,
    max_retries: int = REPURPOSER_MAX_RETRIES,
    allow_low_quality_transcript: bool = False,
) -> RepurposedContent:
    transcript_file = Path(transcript_path)
    if not transcript_file.is_file():
        raise FileNotFoundError(f"Transcript not found: {transcript_file}")
    if transcript_file.suffix.lower() != ".txt":
        print(f"Warning: expected a .txt transcript, got {transcript_file.suffix}")

    transcript = transcript_file.read_text(encoding="utf-8").strip()
    if not transcript:
        raise ValueError("Transcript is empty.")

    character_count = len(transcript)
    word_count = len(transcript.split())
    estimated_tokens = character_count / CHARS_PER_TOKEN_ESTIMATE
    minimum, maximum = target_word_range(word_count)

    print("=" * 60)
    print("AutoTrim - Substack Writer")
    print("=" * 60)
    print(f"Transcript: {transcript_file}")
    print(f"Model:      {model}")
    print(f"Transcript length: {character_count:,} characters")
    print(f"Transcript words:  {word_count:,}")
    print(f"Estimated input tokens: ~{estimated_tokens:,.0f}")
    print(f"Article length target: {minimum:,}–{maximum:,} words")

    quality_issues = assess_transcript_quality(transcript)
    if quality_issues:
        print("[Transcript quality] Review required:")
        for severity, issue in quality_issues:
            print(f"  - {severity.upper()}: {issue}")
        blocking_issues = [issue for severity, issue in quality_issues if severity == "blocking"]
        if blocking_issues and not allow_low_quality_transcript:
            raise TranscriptQualityError(
                "Transcript quality check stopped generation. Inspect transcript.txt, "
                "correct or regenerate it, then rerun. To intentionally bypass the check, "
                "use --allow-low-quality-transcript."
            )
        if allow_low_quality_transcript:
            print("[Transcript quality] Continuing because the override was supplied.")
    else:
        print("[Transcript quality] No obvious text corruption detected.")

    data = generate_content(
        transcript=transcript,
        model=model,
        max_retries=max_retries,
        temperature=temperature,
    )

    issues = validate_content(data, word_count)
    if issues:
        print("[Validation] Warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("[Validation] Article meets the requested length target.")

    text_path, json_path = save_outputs(
        data,
        transcript_file.parent / "content",
    )
    article_words = len(data.substack_content.split())

    print("\nSubstack generation complete")
    print(f"Substack: {text_path}")
    print(f"JSON:     {json_path}")
    print(f"Article words: {article_words:,}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a detailed Substack article from a transcript."
    )
    parser.add_argument("transcript", help="Path to transcript.txt")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=REPURPOSER_TEMPERATURE,
        help=f"Gemini creativity setting from 0 to 1 (default: {REPURPOSER_TEMPERATURE})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=REPURPOSER_MAX_RETRIES,
        help=f"Maximum generation attempts (default: {REPURPOSER_MAX_RETRIES})",
    )
    parser.add_argument(
        "--allow-low-quality-transcript",
        action="store_true",
        help="Continue to Gemini after a transcript quality check failure.",
    )
    args = parser.parse_args()

    if not 0 <= args.temperature <= 1:
        parser.error("--temperature must be between 0 and 1")
    if not 1 <= args.max_retries <= 10:
        parser.error("--max-retries must be between 1 and 10")

    try:
        repurpose_transcript(
            transcript_path=args.transcript,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            allow_low_quality_transcript=args.allow_low_quality_transcript,
        )
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(ExitCode.INTERRUPTED)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        code = getattr(exc, "exit_code", ExitCode.PROCESSING)
        raise SystemExit(code) from exc


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
