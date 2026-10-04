import argparse
import json
import sys
import time
from pathlib import Path

from llm.ollama_provider import OllamaProvider


DEFAULT_MODEL = "qwen2.5-coder:7b"


SYSTEM_PROMPT = """
You are the Content Intelligence engine of AutoTrim.

Your job is to analyze a video transcript and convert it into a
structured, fact-grounded Content Object.

IMPORTANT RULES:

1. Use ONLY information present in the supplied transcript and source metadata.
2. Do not invent facts, names, claims, statistics, examples, or quotes.
3. Preserve the meaning of the original speaker.
4. If something is unclear, do not guess.
5. Extract useful structure that can later be used to create social posts,
   Reddit posts, blog articles, and short-form videos.
6. Quotes MUST be exact text from the transcript.
7. Timestamps MUST come from the supplied transcript.
8. Return ONLY valid JSON.
9. Do not use markdown code fences.
10. Do not add explanations outside the JSON.

Return exactly this structure:

{
  "summary": "A concise factual summary of the entire video",
  "main_topic": "The central topic of the video",
  "topics": [
    "topic 1",
    "topic 2"
  ],
  "key_points": [
    "important point 1",
    "important point 2"
  ],
  "sections": [
    {
      "title": "Section title",
      "summary": "What this section discusses",
      "start": 0.0,
      "end": 10.0
    }
  ],
  "important_quotes": [
    {
      "text": "Exact quote from transcript",
      "start": 0.0,
      "end": 5.0
    }
  ],
  "entities": [
    {
      "name": "Entity name",
      "type": "person|organization|place|technology|concept|other"
    }
  ]
}

Timestamp rules:

- Use timestamps from the transcript.
- Section timestamps must correspond to the relevant transcript content.
- Quote timestamps must correspond to the quoted words.
- Never fabricate timestamps.
"""


def load_transcript(path: Path) -> dict:
    """Load the transcript JSON file."""

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")

    except OSError as e:
        raise ValueError(f"Could not read file: {e}")


def get_segments(data: dict) -> list:
    """
    Get WhisperX transcript segments.

    AutoTrim transcript structure:

    {
        "source": {...},
        "transcription": {
            "segments": [...]
        }
    }
    """

    return data.get("transcription", {}).get("segments", [])


def build_transcript_text(data: dict) -> str:
    """
    Build a timestamped transcript for the LLM.
    """

    lines = []

    segments = get_segments(data)

    for segment in segments:
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "").strip()

        if not text:
            continue

        lines.append(
            f"[{start:.3f} - {end:.3f}] {text}"
        )

    return "\n".join(lines)


def build_prompt(data: dict) -> str:
    """Build the LLM analysis prompt."""

    source = data.get("source", {})

    title = source.get("title", "")
    channel = source.get("channel", "")
    duration = source.get("duration", "")

    transcript = build_transcript_text(data)

    return f"""
Analyze the following video transcript.

SOURCE METADATA
Title: {title}
Channel: {channel}
Duration: {duration} seconds

TRANSCRIPT
--------------------------------------------------
{transcript}
--------------------------------------------------

Create the requested AutoTrim Content Object.

Remember:
- Stay strictly grounded in the transcript.
- Do not invent information.
- Use exact transcript text for quotes.
- Use the supplied timestamps.
"""


def validate_content_object(content: dict):
    """Validate the structure returned by the LLM."""

    required_fields = [
        "summary",
        "main_topic",
        "topics",
        "key_points",
        "sections",
        "important_quotes",
        "entities",
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

    if not isinstance(content["topics"], list):
        raise ValueError("'topics' must be a list")

    if not isinstance(content["key_points"], list):
        raise ValueError("'key_points' must be a list")

    if not isinstance(content["sections"], list):
        raise ValueError("'sections' must be a list")

    if not isinstance(content["important_quotes"], list):
        raise ValueError("'important_quotes' must be a list")

    if not isinstance(content["entities"], list):
        raise ValueError("'entities' must be a list")


def save_content_object(
    output_path: Path,
    content: dict,
    transcript_data: dict,
    model: str,
    generation_time: float,
):
    """Save the final Content Object."""

    output = {
        "autotrim_version": "0.2",

        "generated_by": {
            "provider": "ollama",
            "model": model,
        },

        "source": transcript_data.get("source", {}),

        "content": content,

        "processing": {
            "generation_time_seconds": round(
                generation_time,
                3,
            )
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "AutoTrim Phase 2 - "
            "Transcript to Content Object"
        )
    )

    parser.add_argument(
        "transcript",
        help="Path to transcript.json",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Ollama model "
            f"(default: {DEFAULT_MODEL})"
        ),
    )

    parser.add_argument(
        "--output",
        help=(
            "Output content.json path. "
            "Defaults to the transcript directory."
        ),
    )

    args = parser.parse_args()

    transcript_path = Path(args.transcript)

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
    # Output path
    # ---------------------------------------------------------

    if args.output:

        output_path = Path(args.output)

    else:

        output_path = (
            transcript_path.parent
            / "content.json"
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
        f"Model      : {args.model}"
    )

    print(
        f"Output     : {output_path}"
    )

    # ---------------------------------------------------------
    # Step 1 - Load transcript
    # ---------------------------------------------------------

    print("\n[1/3] Loading transcript...")

    try:

        transcript_data = load_transcript(
            transcript_path
        )

    except ValueError as e:

        print(f"ERROR: {e}")
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
    # Step 2 - Ollama
    # ---------------------------------------------------------

    print("\n[2/3] Analyzing with Ollama...")

    ollama = OllamaProvider(
        model=args.model
    )

    if not ollama.check_connection():

        print(
            "ERROR: Ollama is not running."
        )

        print(
            "Start Ollama and try again."
        )

        sys.exit(1)

    prompt = build_prompt(
        transcript_data
    )

    start_time = time.perf_counter()

    try:

        result = ollama.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=0.1,
        )

    except Exception as e:

        print(
            f"ERROR: Ollama generation failed: {e}"
        )

        sys.exit(1)

    generation_time = (
        time.perf_counter()
        - start_time
    )

    raw_response = result.get(
        "response",
        ""
    ).strip()

    if not raw_response:

        print(
            "ERROR: Ollama returned an empty response."
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # Step 3 - Parse and validate
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
            "ERROR: Ollama did not return valid JSON."
        )

        print(
            "\nRaw response:"
        )

        print(raw_response)

        print(
            f"\nJSON error: {e}"
        )

        sys.exit(1)

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
            model=args.model,
            generation_time=generation_time,
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
        f"Output            : {output_path}"
    )

    print(
        f"Generation time   : "
        f"{generation_time:.2f}s"
    )

    print(
        f"Topics            : "
        f"{len(content['topics'])}"
    )

    print(
        f"Key points        : "
        f"{len(content['key_points'])}"
    )

    print(
        f"Sections          : "
        f"{len(content['sections'])}"
    )

    print(
        f"Important quotes  : "
        f"{len(content['important_quotes'])}"
    )

    print(
        f"Entities          : "
        f"{len(content['entities'])}"
    )

    print(
        "\nOllama model unload requested."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()