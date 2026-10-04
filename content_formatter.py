import argparse
import json
import sys
import time
from pathlib import Path

from llm.ollama_provider import OllamaProvider
from llm.gemini_provider import GeminiProvider


AUTOTRIM_VERSION = "0.4"

DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


SYSTEM_PROMPT = """
You are the Content Intelligence engine of AutoTrim.

Your job is to analyze a video transcript and create a structured,
fact-grounded Content Object that will later be used to generate:

- X posts and threads
- Reddit posts
- Blog articles
- Short-form video candidates
- Captions
- Other derivative content

IMPORTANT RULES:

1. Use ONLY information present in the supplied transcript and source metadata.
2. Do not invent facts, names, claims, statistics, examples, or events.
3. Preserve the meaning of the original speaker.
4. Do not add outside knowledge.
5. If something is unclear, do not guess.
6. Quotes MUST be exact text from the transcript.
7. Timestamps MUST come from the supplied transcript.
8. Candidate segment timestamps must correspond to actual transcript segments.
9. Do not create duplicate candidate segments unnecessarily.
10. Prefer complete thoughts over arbitrary sentence fragments.
11. Return ONLY valid JSON.
12. Do not use markdown code fences.
13. Do not add explanations outside JSON.

CONTENT ANALYSIS

Identify:

- The main topic
- Important secondary topics
- The most important points
- Major claims or explanations made in the video
- Useful sections
- Exact important quotes
- Relevant entities
- Different possible content angles
- Strong candidate clips for short-form content

CANDIDATE SEGMENTS

A candidate segment should:

- Contain a meaningful and relatively self-contained idea.
- Have a clear beginning and ending.
- Be understandable with minimal context.
- Have potential educational, surprising, interesting, useful, or entertaining value.
- Preferably work as a standalone short-form clip.
- Use timestamps from the transcript.
- Prefer 15-60 seconds for short-form candidates.
- Do not simply divide the whole video into large sections.
- Find the strongest specific moments instead.

Score each candidate from 0.0 to 1.0.

Consider:

- Hook strength
- Information value
- Standalone clarity
- Curiosity
- Shareability
- Completeness

CONTENT ANGLES

Examples of angles include:

- Educational explanation
- Surprising fact
- Common misconception
- Question and answer
- Story
- Practical takeaway
- Contrarian idea
- Beginner explanation
- Interesting analogy

Only use angles that actually fit the transcript.

Return EXACTLY this structure:

{
  "summary": "...",
  "main_topic": "...",
  "topics": [
    "..."
  ],
  "key_points": [
    "..."
  ],
  "claims": [
    {
      "claim": "...",
      "supporting_text": "...",
      "start": 0.0,
      "end": 0.0
    }
  ],
  "sections": [
    {
      "title": "...",
      "summary": "...",
      "start": 0.0,
      "end": 0.0
    }
  ],
  "important_quotes": [
    {
      "text": "...",
      "start": 0.0,
      "end": 0.0
    }
  ],
  "entities": [
    {
      "name": "...",
      "type": "person|organization|place|technology|concept|other"
    }
  ],
  "content_angles": [
    {
      "angle": "...",
      "description": "...",
      "best_for": [
        "x",
        "reddit",
        "blog",
        "short"
      ]
    }
  ],
  "candidate_segments": [
    {
      "start": 0.0,
      "end": 0.0,
      "title": "...",
      "hook": "...",
      "reason": "...",
      "score": 0.0
    }
  ]
}

Timestamp rules:

- Use timestamps from the supplied transcript.
- Do not invent timestamps.
- Quote timestamps must correspond to the quoted transcript.
- Candidate segment timestamps must cover actual transcript content.
- The start and end of a candidate should normally align with transcript
  segment boundaries.
"""


def load_transcript(path: Path) -> dict:
    """Load transcript JSON."""

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")

    except OSError as e:
        raise ValueError(f"Could not read file: {e}")


def get_segments(data: dict) -> list:
    """Get WhisperX transcript segments."""

    return data.get(
        "transcription",
        {}
    ).get(
        "segments",
        []
    )


def build_transcript_text(data: dict) -> str:
    """
    Build timestamped transcript for the LLM.
    """

    lines = []

    segments = get_segments(data)

    for segment in segments:

        start = segment.get(
            "start",
            0
        )

        end = segment.get(
            "end",
            0
        )

        text = segment.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        lines.append(
            f"[{start:.3f} - {end:.3f}] {text}"
        )

    return "\n".join(lines)


def build_prompt(data: dict) -> str:
    """Build the content intelligence prompt."""

    source = data.get(
        "source",
        {}
    )

    title = source.get(
        "title",
        ""
    )

    channel = source.get(
        "channel",
        ""
    )

    duration = source.get(
        "duration",
        ""
    )

    video_type = source.get(
        "video_type",
        ""
    )

    transcript = build_transcript_text(
        data
    )

    return f"""
Analyze the following video.

SOURCE METADATA
--------------------------------------------------

Title: {title}
Channel: {channel}
Duration: {duration} seconds
Video type: {video_type}

--------------------------------------------------

TIMESTAMPED TRANSCRIPT
--------------------------------------------------

{transcript}

--------------------------------------------------

Create the AutoTrim Content Object.

Focus on understanding the actual content rather than simply
rewriting the transcript.

For candidate_segments:

- Find the strongest specific moments.
- Prefer complete ideas.
- Prefer approximately 15-60 second clips when possible.
- Do NOT simply split the video into large sections.
- A candidate should have a clear hook or useful takeaway.
- Candidate timestamps should align with actual transcript segment
  boundaries.

For claims, only include claims or explanations actually made
in the transcript.

For important_quotes, copy the transcript wording exactly.

Return ONLY valid JSON.
"""


def clean_json_response(response: str) -> str:
    """
    Clean common formatting added by LLMs before JSON parsing.

    Handles:

        ```json
        {...}
        ```

    as well as accidental surrounding text.
    """

    response = response.strip()

    if not response:
        return response

    # Remove Markdown code fences.
    if response.startswith("```"):

        lines = response.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        response = "\n".join(lines).strip()

    # Recover JSON object if model added text around it.
    if not response.startswith("{"):

        start = response.find("{")
        end = response.rfind("}")

        if start != -1 and end != -1 and end > start:
            response = response[start:end + 1]

    return response.strip()


def validate_content_object(content: dict):
    """Validate required Content Object structure."""

    required_fields = [
        "summary",
        "main_topic",
        "topics",
        "key_points",
        "claims",
        "sections",
        "important_quotes",
        "entities",
        "content_angles",
        "candidate_segments",
    ]

    missing = [
        field
        for field in required_fields
        if field not in content
    ]

    if missing:
        raise ValueError(
            "Missing required fields: "
            + ", ".join(missing)
        )

    list_fields = [
        "topics",
        "key_points",
        "claims",
        "sections",
        "important_quotes",
        "entities",
        "content_angles",
        "candidate_segments",
    ]

    for field in list_fields:

        if not isinstance(
            content[field],
            list
        ):
            raise ValueError(
                f"'{field}' must be a list"
            )


def save_content_object(
    output_path: Path,
    content: dict,
    transcript_data: dict,
    provider_name: str,
    model: str,
    generation_time: float,
):
    """Save Content Object."""

    output = {
        "autotrim_version": AUTOTRIM_VERSION,

        "generated_by": {
            "provider": provider_name,
            "model": model,
        },

        "source": transcript_data.get(
            "source",
            {}
        ),

        "content": content,

        "processing": {
            "generation_time_seconds": round(
                generation_time,
                3
            )
        }
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )


def create_provider(
    provider_name: str,
    model: str | None,
):
    """
    Create the selected LLM provider.
    """

    if provider_name == "ollama":

        selected_model = (
            model
            if model
            else DEFAULT_OLLAMA_MODEL
        )

        return OllamaProvider(
            model=selected_model
        )

    if provider_name == "gemini":

        selected_model = (
            model
            if model
            else DEFAULT_GEMINI_MODEL
        )

        return GeminiProvider(
            model=selected_model
        )

    raise ValueError(
        f"Unsupported provider: {provider_name}"
    )


def get_output_path(
    transcript_path: Path,
    provider_name: str,
    custom_output: str | None,
) -> Path:

    if custom_output:
        return Path(custom_output)

    return (
        transcript_path.parent
        / f"content_{provider_name}.json"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "AutoTrim - Content Intelligence"
        )
    )

    parser.add_argument(
        "transcript",
        help="Path to transcript.json"
    )

    parser.add_argument(
        "--provider",
        choices=[
            "ollama",
            "gemini"
        ],
        default="ollama",
        help=(
            "LLM provider "
            "(default: ollama)"
        )
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override the default model "
            "for the selected provider"
        )
    )

    parser.add_argument(
        "--output",
        help=(
            "Output content.json path. "
            "Defaults to content_<provider>.json "
            "in the transcript directory."
        )
    )

    args = parser.parse_args()

    transcript_path = Path(
        args.transcript
    )

    # ---------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------

    if not transcript_path.exists():

        print(
            f"ERROR: File not found: "
            f"{transcript_path}"
        )

        sys.exit(1)

    if not transcript_path.is_file():

        print(
            f"ERROR: Not a file: "
            f"{transcript_path}"
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # Provider
    # ---------------------------------------------------------

    try:

        provider = create_provider(
            args.provider,
            args.model
        )

    except Exception as e:

        print(
            f"ERROR: Could not initialize "
            f"{args.provider}: {e}"
        )

        sys.exit(1)

    model_name = provider.model

    output_path = get_output_path(
        transcript_path,
        args.provider,
        args.output
    )

    # ---------------------------------------------------------
    # Header
    # ---------------------------------------------------------

    print("=" * 60)
    print("AutoTrim - Content Intelligence")
    print("=" * 60)

    print(
        f"Transcript : {transcript_path}"
    )

    print(
        f"Provider   : {args.provider}"
    )

    print(
        f"Model      : {model_name}"
    )

    print(
        f"Output     : {output_path}"
    )

    # ---------------------------------------------------------
    # Load transcript
    # ---------------------------------------------------------

    print(
        "\n[1/3] Loading transcript..."
    )

    try:

        transcript_data = load_transcript(
            transcript_path
        )

    except ValueError as e:

        print(
            f"ERROR: {e}"
        )

        sys.exit(1)

    segments = get_segments(
        transcript_data
    )

    if not segments:

        print(
            "ERROR: Transcript contains no "
            "transcription segments."
        )

        print(
            "\nExpected structure:"
        )

        print(
            "transcription -> segments -> [...]"
        )

        sys.exit(1)

    transcript_text = build_transcript_text(
        transcript_data
    )

    if not transcript_text.strip():

        print(
            "ERROR: Transcript segments exist "
            "but contain no text."
        )

        sys.exit(1)

    print(
        f"Segments   : {len(segments)}"
    )

    print(
        f"Characters : {len(transcript_text):,}"
    )

    # ---------------------------------------------------------
    # Connection
    # ---------------------------------------------------------

    print(
        f"\n[2/3] Analyzing with "
        f"{args.provider.capitalize()}..."
    )

    if args.provider == "ollama":

        if not provider.check_connection():

            print(
                "ERROR: Ollama is not running."
            )

            print(
                "Start Ollama and try again."
            )

            sys.exit(1)

    else:

        print(
            "Gemini API key loaded."
        )

    # ---------------------------------------------------------
    # Generate
    # ---------------------------------------------------------

    prompt = build_prompt(
        transcript_data
    )

    start_time = time.perf_counter()

    try:

        result = provider.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=0.1
        )

    except Exception as e:

        print(
            f"ERROR: {args.provider.capitalize()} "
            f"generation failed: {e}"
        )

        sys.exit(1)

    generation_time = (
        time.perf_counter()
        - start_time
    )

    raw_response = clean_json_response(
        result.get(
            "response",
            ""
        )
    )

    if not raw_response:

        print(
            f"ERROR: {args.provider.capitalize()} "
            f"returned an empty response."
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # Parse JSON
    # ---------------------------------------------------------

    print(
        "\n[3/3] Validating and saving..."
    )

    try:

        content = json.loads(
            raw_response
        )

    except json.JSONDecodeError as e:

        print(
            f"ERROR: {args.provider.capitalize()} "
            f"did not return valid JSON."
        )

        print(
            "\nCleaned response:"
        )

        print(
            raw_response
        )

        print(
            f"\nJSON error: {e}"
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    try:

        validate_content_object(
            content
        )

    except ValueError as e:

        print(
            f"ERROR: Invalid Content Object: {e}"
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    try:

        save_content_object(
            output_path=output_path,
            content=content,
            transcript_data=transcript_data,
            provider_name=args.provider,
            model=model_name,
            generation_time=generation_time
        )

    except OSError as e:

        print(
            f"ERROR: Could not save output: {e}"
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print(
        "\nContent Object generated successfully."
    )

    print(
        f"Output             : {output_path}"
    )

    print(
        f"Provider           : {args.provider}"
    )

    print(
        f"Model              : {model_name}"
    )

    print(
        f"Generation time    : "
        f"{generation_time:.2f}s"
    )

    print(
        f"Topics             : "
        f"{len(content['topics'])}"
    )

    print(
        f"Key points         : "
        f"{len(content['key_points'])}"
    )

    print(
        f"Claims             : "
        f"{len(content['claims'])}"
    )

    print(
        f"Sections           : "
        f"{len(content['sections'])}"
    )

    print(
        f"Important quotes   : "
        f"{len(content['important_quotes'])}"
    )

    print(
        f"Entities           : "
        f"{len(content['entities'])}"
    )

    print(
        f"Content angles     : "
        f"{len(content['content_angles'])}"
    )

    print(
        f"Candidate segments : "
        f"{len(content['candidate_segments'])}"
    )

    if args.provider == "ollama":

        print(
            "\nOllama model unload requested."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()