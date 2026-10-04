"""
AutoTrim - Complete Video-to-Content Pipeline

Pipeline:

    Video URL
        ↓
    WhisperX transcription
        ↓
    transcript.json
    transcript.txt
        ↓
    Gemini Repurposer
        ↓
    X thread
    Reddit post
    Blog article
"""

import argparse
import subprocess
import sys
from pathlib import Path


AUTOTRIM_VERSION = "0.2"


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


def find_latest_transcript(output_dir):
    """
    Find the newest transcript.json and its corresponding
    transcript.txt.

    transcribe.py creates:

        output/
            <video title>/
                transcript.json
                transcript.txt
    """

    transcript_files = list(
        output_dir.glob("*/transcript.json")
    )

    if not transcript_files:
        print()
        print(
            "ERROR: No transcript.json was found "
            "inside the output directory."
        )
        sys.exit(1)

    # Select the transcript generated most recently.
    transcript_json = max(
        transcript_files,
        key=lambda p: p.stat().st_mtime
    )

    video_output_dir = transcript_json.parent

    transcript_txt = video_output_dir / "transcript.txt"

    if not transcript_txt.exists():
        print()
        print(
            "ERROR: transcript.txt was not found at:\n"
            f"{transcript_txt}"
        )
        sys.exit(1)

    return transcript_json, transcript_txt


def main():

    parser = argparse.ArgumentParser(
        description=(
            "AutoTrim - Video to X, Reddit and Blog "
            "content pipeline"
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
        help="LLM provider to use (default: gemini)"
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
    output_dir = project_dir / "output"

    # =========================================================
    # PIPELINE HEADER
    # =========================================================

    print()
    print("=" * 60)
    print("AutoTrim")
    print("=" * 60)

    print(f"Version              : {AUTOTRIM_VERSION}")
    print(f"LLM                  : {args.llm}")

    if args.model:
        print(f"LLM Model            : {args.model}")

    print(
        f"Transcription Model  : "
        f"{args.transcription_model}"
    )

    print(
        f"Batch Size           : "
        f"{args.batch_size}"
    )

    # =========================================================
    # STEP 1 - TRANSCRIPTION
    # =========================================================

    transcription_command = [
        python_executable,
        str(project_dir / "transcribe.py"),
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

    # =========================================================
    # LOCATE TRANSCRIPTS
    # =========================================================

    transcript_json, transcript_txt = find_latest_transcript(
        output_dir
    )

    print()
    print("=" * 60)
    print("TRANSCRIPT OUTPUT")
    print("=" * 60)

    print()
    print(f"JSON : {transcript_json}")
    print(f"TXT  : {transcript_txt}")

    # =========================================================
    # STEP 2 - CONTENT REPURPOSING
    # =========================================================

    repurposer_command = [
        python_executable,
        str(project_dir / "repurposer.py"),
        str(transcript_txt),
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

    # =========================================================
    # OUTPUT PATHS
    # =========================================================

    content_dir = transcript_txt.parent / "content"

    x_thread_path = content_dir / "x_thread.md"
    reddit_path = content_dir / "reddit_post.md"
    blog_path = content_dir / "blog_post.md"
    json_path = content_dir / "repurposed.json"

    # =========================================================
    # COMPLETE
    # =========================================================

    print()
    print("=" * 60)
    print("AutoTrim Pipeline Complete")
    print("=" * 60)

    print()
    print(f"Transcript JSON : {transcript_json}")
    print(f"Transcript TXT  : {transcript_txt}")

    print()
    print("Generated Content:")

    print(f"X Thread        : {x_thread_path}")
    print(f"Reddit Post     : {reddit_path}")
    print(f"Blog Article    : {blog_path}")
    print(f"Structured JSON : {json_path}")

    print()
    print(f"LLM             : {args.llm}")

    if args.model:
        print(f"Model           : {args.model}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()