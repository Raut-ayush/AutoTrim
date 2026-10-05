"""
AutoTrim - Resumable Video-to-Content Pipeline

Pipeline:

    Video URL
        ↓
    Existing transcript? ── YES ──→ Reuse
        │
        NO
        ↓
    WhisperX transcription
        ↓
    transcript.json
    transcript.txt
        ↓
    Existing content? ── YES ──→ Reuse
        │
        NO
        ↓
    Gemini Repurposer
        ↓
    X thread
    Reddit post
    Blog article

The pipeline is resumable:
- Existing transcription is not repeated.
- Existing repurposed content is not repeated.
- If a later stage fails, the next run resumes from that stage.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


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

    result = subprocess.run(
        command,
        cwd=Path(__file__).parent,
    )

    if result.returncode != 0:

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
        content_dir / "x_thread.md",
        content_dir / "reddit_post.md",
        content_dir / "blog_post.md",
        content_dir / "repurposed.json",
    ]

    if all(
        path.exists()
        for path in required_files
    ):

        return content_dir

    return None


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
            "Optional Gemini model override "
            "(default: gemini-3.8-flash)"
        )
    )

    parser.add_argument(
        "--transcription-model",
        default="small",
        help="WhisperX model (default: small)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="WhisperX batch size (default: 4)"
    )

    args = parser.parse_args()

    project_dir = Path(__file__).parent

    python_executable = sys.executable

    output_dir = (
        project_dir /
        "output"
    )

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
            "✓ Existing transcript found."
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

        transcription_command = [
            python_executable,
            str(
                project_dir /
                "transcribe.py"
            ),
            args.url,
            "--model",
            args.transcription_model,
            "--batch-size",
            str(args.batch_size),
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

            sys.exit(1)

        if transcript_txt is None:

            print()
            print(
                "ERROR: transcript.txt was not found "
                "after transcription."
            )

            sys.exit(1)

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
            "✓ Existing repurposed content found."
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

    x_thread_path = (
        existing_content /
        "x_thread.md"
    )

    reddit_path = (
        existing_content /
        "reddit_post.md"
    )

    blog_path = (
        existing_content /
        "blog_post.md"
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
        f"X Thread        : "
        f"{x_thread_path}"
    )

    print(
        f"Reddit Post     : "
        f"{reddit_path}"
    )

    print(
        f"Blog Article    : "
        f"{blog_path}"
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
    main()