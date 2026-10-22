"""
AutoTrim - Resumable Video-to-Content Pipeline

Pipeline:

    Video URL
        â†“
    Existing transcript? â”€â”€ YES â”€â”€â†’ Reuse
        â”‚
        NO
        â†“
    WhisperX transcription
        â†“
    transcript.json
    transcript.txt
        â†“
    Existing content? â”€â”€ YES â”€â”€â†’ Reuse
        â”‚
        NO
        â†“
    Gemini Substack Writer
        â†“
    Substack post

The pipeline is resumable:
- Existing transcription is not repeated.
- Existing repurposed content is not repeated.
- If a later stage fails, the next run resumes from that stage.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from config import (
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL,
    REPURPOSER_MAX_RETRIES,
    REPURPOSER_TEMPERATURE,
    OUTPUT_DIR,
    SUPPORTED_TRANSCRIPTION_LANGUAGES,
    TRANSCRIPTION_BATCH_SIZE,
    TRANSCRIPTION_LANGUAGE,
    TRANSCRIPTION_MODEL,
)
from errors import ExitCode


AUTOTRIM_VERSION = "0.3"


# ============================================================
# Command Runner
# ============================================================

def run_command(command, step_name):
    """
    Run a subprocess and stop the pipeline if it fails.
    """

    print()
    print("=" * 60)
    print(step_name)
    print("=" * 60)

    print("Command:")
    print(" ".join(str(x) for x in command))
    print()

    try:
        result = subprocess.run(
            command,
            cwd=Path(__file__).parent,
            check=False,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: Could not start {step_name}: executable or script not found: {exc}")
        raise SystemExit(ExitCode.DEPENDENCY) from exc
    except OSError as exc:
        print(f"ERROR: Could not start {step_name}: {exc}")
        raise SystemExit(ExitCode.PROCESSING) from exc

    if result.returncode != 0:

        if result.returncode == ExitCode.TRANSCRIPT_QUALITY:
            print()
            print("TRANSCRIPT NEEDS REVIEW")
            print(
                "Transcription finished, but article generation was skipped because "
                "the transcript contains strong corruption signals. Review transcript.txt "
                "in the output folder, correct it or transcribe with a different model, "
                "then rerun. The existing transcript will be reused."
            )
            print(
                "If you intentionally want Gemini to use this transcript, rerun with "
                "--allow-low-quality-transcript."
            )
            raise SystemExit(ExitCode.TRANSCRIPT_QUALITY)

        print()
        print("=" * 60)
        print(f"ERROR: {step_name} failed.")
        print("=" * 60)

        sys.exit(result.returncode)


# ============================================================
# YouTube Video ID
# ============================================================

def extract_video_id(url):
    """
    Extract a YouTube video ID from common URL formats.

    Supports:

        https://youtu.be/VIDEO_ID
        https://www.youtube.com/watch?v=VIDEO_ID
        https://youtube.com/shorts/VIDEO_ID
        https://www.youtube.com/embed/VIDEO_ID
    """

    parsed = urlparse(url)

    hostname = (
        parsed.hostname or ""
    ).lower()

    # --------------------------------------------------------
    # youtu.be/<id>
    # --------------------------------------------------------

    if hostname in {
        "youtu.be",
        "www.youtu.be",
    }:

        video_id = (
            parsed.path
            .strip("/")
            .split("/")[0]
        )

        if video_id:
            return video_id

    # --------------------------------------------------------
    # youtube.com
    # --------------------------------------------------------

    if "youtube.com" in hostname:

        # /watch?v=...
        if parsed.path == "/watch":

            from urllib.parse import parse_qs

            query = parse_qs(
                parsed.query
            )

            video_ids = query.get("v")

            if video_ids:
                return video_ids[0]

        # /shorts/<id>
        # /embed/<id>
        # /live/<id>
        path_parts = [
            part
            for part in parsed.path.split("/")
            if part
        ]

        if len(path_parts) >= 2:

            if path_parts[0] in {
                "shorts",
                "embed",
                "live",
            }:

                return path_parts[1]

    return None


# ============================================================
# Read Transcript Metadata
# ============================================================

def load_json(path):
    """
    Load JSON file safely.
    """

    try:

        with path.open(
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return None


def extract_video_id_from_metadata(data):
    """
    Try to extract video ID from transcript metadata.

    Handles several possible metadata structures so this
    remains compatible with existing transcript.json files.
    """

    if not isinstance(data, dict):
        return None

    # --------------------------------------------------------
    # Direct fields
    # --------------------------------------------------------

    direct_fields = [
        "video_id",
        "videoId",
        "id",
    ]

    for field in direct_fields:

        value = data.get(field)

        if isinstance(value, str) and value:
            return value

    # --------------------------------------------------------
    # Common nested structures
    # --------------------------------------------------------

    nested_objects = [
        data.get("source"),
        data.get("metadata"),
        data.get("video"),
    ]

    for obj in nested_objects:

        if not isinstance(obj, dict):
            continue

        for field in direct_fields:

            value = obj.get(field)

            if isinstance(value, str) and value:
                return value

    return None


# ============================================================
# Find Existing Transcript
# ============================================================

def find_existing_transcript(
    output_dir,
    video_id
):
    """
    Search existing transcript.json files for a matching
    YouTube video ID.

    Returns:

        (transcript_json, transcript_txt)

    or:

        (None, None)
    """

    if not video_id:

        return None, None

    transcript_files = list(
        output_dir.glob(
            "*/transcript.json"
        )
    )

    for transcript_json in transcript_files:

        data = load_json(
            transcript_json
        )

        stored_video_id = (
            extract_video_id_from_metadata(
                data
            )
        )

        if stored_video_id != video_id:
            continue

        transcript_txt = (
            transcript_json.parent /
            "transcript.txt"
        )

        # A transcript.json alone is not enough for our
        # current repurposing stage.
        if not transcript_txt.exists():
            continue

        return (
            transcript_json,
            transcript_txt,
        )

    return None, None


# ============================================================
# Find Latest Transcript
# ============================================================

def find_latest_transcript(output_dir):
    """
    Fallback used after a new transcription.

    Finds the newest transcript.json and its corresponding
    transcript.txt.
    """

    transcript_files = list(
        output_dir.glob(
            "*/transcript.json"
        )
    )

    if not transcript_files:

        return None, None

    transcript_json = max(
        transcript_files,
        key=lambda p: p.stat().st_mtime
    )

    transcript_txt = (
        transcript_json.parent /
        "transcript.txt"
    )

    if not transcript_txt.exists():

        return (
            transcript_json,
            None
        )

    return (
        transcript_json,
        transcript_txt,
    )


# ============================================================
# Check Repurposed Content
# ============================================================

def find_existing_content(
    transcript_txt
):
    """
    Check whether all required repurposed outputs
    already exist.
    """

    content_dir = (
        transcript_txt.parent /
        "content"
    )

    required_files = [
        content_dir / "substack_post.txt",
        content_dir / "repurposed.json",
    ]
    if not all(path.is_file() for path in required_files):
        return None

    data = load_json(required_files[1])
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("substack_title"), str):
        return None
    if not isinstance(data.get("substack_content"), str):
        return None
    if "x_thread" in data or "reddit_title" in data or "reddit_body" in data:
        return None

    return content_dir



# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AutoTrim - Resumable "
            "Video-to-Content Pipeline"
        )
    )

    parser.add_argument(
        "url",
        help="Video URL"
    )

    parser.add_argument(
        "--llm",
        choices=[
            "gemini",
        ],
        default="gemini",
        help="LLM provider (default: gemini)"
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Optional Gemini model override (default: {GEMINI_MODEL})"
        )
    )

    parser.add_argument(
        "--transcription-model",
        default=TRANSCRIPTION_MODEL,
        help=f"WhisperX model (default: {TRANSCRIPTION_MODEL})"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRANSCRIPTION_BATCH_SIZE,
        help=f"WhisperX batch size (default: {TRANSCRIPTION_BATCH_SIZE})"
    )

    parser.add_argument(
        "--transcription-language",
        choices=sorted(SUPPORTED_TRANSCRIPTION_LANGUAGES),
        default=TRANSCRIPTION_LANGUAGE,
        help="Transcription language: auto-detect, Hindi, or mixed Hindi/English",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=REPURPOSER_TEMPERATURE,
        help=f"Gemini creativity setting from 0 to 1 (default: {REPURPOSER_TEMPERATURE})",
    )

    parser.add_argument(
        "--generation-retries",
        type=int,
        default=REPURPOSER_MAX_RETRIES,
        help=f"Maximum Gemini generation attempts (default: {REPURPOSER_MAX_RETRIES})",
    )

    parser.add_argument(
        "--allow-low-quality-transcript",
        action="store_true",
        help="Continue to Gemini when transcript quality checks detect likely corruption.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output root directory for transcripts and content",
    )

    args = parser.parse_args()

    if not 0 <= args.temperature <= 1:
        parser.error("--temperature must be between 0 and 1")
    if not 1 <= args.generation_retries <= 10:
        parser.error("--generation-retries must be between 1 and 10")

    project_dir = Path(__file__).parent
    load_dotenv(project_dir / ".env")

    python_executable = sys.executable

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    output_dir = output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 60)
    print("AutoTrim")
    print("=" * 60)

    print(
        f"Version              : "
        f"{AUTOTRIM_VERSION}"
    )

    print(
        f"LLM                  : "
        f"{args.llm}"
    )

    if args.model:

        print(
            f"LLM Model            : "
            f"{args.model}"
        )

    print(
        f"Transcription Model  : "
        f"{args.transcription_model}"
    )

    print(
        f"Batch Size           : "
        f"{args.batch_size}"
    )

    language_labels = {"auto": "Auto-detect", "hi": "Hindi", "hi-en": "Hindi + English (auto-detect)"}
    print(f"Transcription Language: {language_labels[args.transcription_language]}")

    # ========================================================
    # IDENTIFY VIDEO
    # ========================================================

    video_id = extract_video_id(
        args.url
    )

    print()

    if video_id:

        print(
            f"Video ID             : "
            f"{video_id}"
        )
    else:

        print(
            "Video ID             : "
            "Could not determine"
        )

    # ========================================================
    # STEP 1 - TRANSCRIPTION
    # ========================================================

    transcript_json = None
    transcript_txt = None

    # --------------------------------------------------------
    # Check whether transcription already exists
    # --------------------------------------------------------

    if video_id:

        (
            transcript_json,
            transcript_txt
        ) = find_existing_transcript(
            output_dir,
            video_id
        )

    existing_content = (
        find_existing_content(transcript_txt)
        if transcript_txt is not None
        else None
    )
    if existing_content is None:
        if not os.getenv(GEMINI_API_KEY_ENV):
            print(
                f"ERROR: {GEMINI_API_KEY_ENV} is not set. Add it to "
                f"{project_dir / '.env'} or your environment before generating."
            )
            return ExitCode.CONFIGURATION

    if (
        transcript_json is not None
        and transcript_txt is not None
    ):

        print()
        print("=" * 60)
        print("[1/2] TRANSCRIPTION")
        print("=" * 60)

        print()
        print(
            "âœ“ Existing transcript found."
        )

        print(
            "Skipping transcription."
        )

        print()
        print(
            f"JSON : {transcript_json}"
        )

        print(
            f"TXT  : {transcript_txt}"
        )

    else:

        # ----------------------------------------------------
        # Run transcription
        # ----------------------------------------------------

        if shutil.which("ffmpeg") is None:
            print(
                "ERROR: FFmpeg was not found. Install FFmpeg and add its "
                "bin directory to PATH before transcribing."
            )
            return ExitCode.DEPENDENCY

        transcription_command = [
            python_executable,
            "-u",
            str(
                project_dir /
                "transcribe.py"
            ),
            args.url,
            "--model",
            args.transcription_model,
            "--batch-size",
            str(args.batch_size),
            "--language",
            args.transcription_language,
            "--output-dir",
            str(output_dir),
        ]

        run_command(
            transcription_command,
            "[1/2] TRANSCRIPTION"
        )

        # ----------------------------------------------------
        # Locate newly generated transcript
        # ----------------------------------------------------

        (
            transcript_json,
            transcript_txt
        ) = find_latest_transcript(
            output_dir
        )

        if transcript_json is None:

            print()
            print(
                "ERROR: No transcript.json "
                "was found after transcription."
            )

            sys.exit(ExitCode.PROCESSING)

        if transcript_txt is None:

            print()
            print(
                "ERROR: transcript.txt was not found "
                "after transcription."
            )

            sys.exit(ExitCode.PROCESSING)

        print()
        print(
            "=" * 60
        )

        print(
            "TRANSCRIPT OUTPUT"
        )

        print(
            "=" * 60
        )

        print()
        print(
            f"JSON : {transcript_json}"
        )

        print(
            f"TXT  : {transcript_txt}"
        )

    # ========================================================
    # STEP 2 - CONTENT REPURPOSING
    # ========================================================

    existing_content = (
        find_existing_content(
            transcript_txt
        )
    )

    if existing_content is not None:

        print()
        print("=" * 60)
        print("[2/2] CONTENT REPURPOSING")
        print("=" * 60)

        print()
        print(
            "âœ“ Existing repurposed content found."
        )

        print(
            "Skipping Gemini generation."
        )

        print()
        print(
            f"Content : {existing_content}"
        )

    else:

        # ----------------------------------------------------
        # Run repurposer
        # ----------------------------------------------------

        repurposer_command = [
            python_executable,
            "-u",
            str(
                project_dir /
                "repurposer.py"
            ),
            str(
                transcript_txt
            ),
        ]

        if args.model:

            repurposer_command.extend([
                "--model",
                args.model,
            ])

        repurposer_command.extend([
            "--temperature", str(args.temperature),
            "--max-retries", str(args.generation_retries),
        ])
        if args.allow_low_quality_transcript:
            repurposer_command.append("--allow-low-quality-transcript")

        run_command(
            repurposer_command,
            "[2/2] CONTENT REPURPOSING"
        )

        existing_content = (
            transcript_txt.parent /
            "content"
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    substack_path = (
        existing_content /
        "substack_post.txt"
    )

    json_path = (
        existing_content /
        "repurposed.json"
    )

    print()
    print("=" * 60)
    print("AutoTrim Pipeline Complete")
    print("=" * 60)

    print()
    print(
        f"Transcript JSON : "
        f"{transcript_json}"
    )

    print(
        f"Transcript TXT  : "
        f"{transcript_txt}"
    )

    print()
    print("Generated Content:")

    print(
        f"Substack Post    : "
        f"{substack_path}"
    )

    print(
        f"Structured JSON : "
        f"{json_path}"
    )

    print()
    print(
        f"LLM             : "
        f"{args.llm}"
    )

    if args.model:

        print(
            f"Model           : "
            f"{args.model}"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main() or ExitCode.SUCCESS)
