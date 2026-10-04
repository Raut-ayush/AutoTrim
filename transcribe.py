import argparse
import json
from pathlib import Path

from yt_dlp import YoutubeDL
import whisperx


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "small"
DEFAULT_BATCH_SIZE = 4


# ============================================================
# YOUTUBE / MEDIA INGESTION
# ============================================================

def download_audio(url: str, output_dir: Path):
    """
    Download the best available audio from a YouTube URL
    and convert it to WAV using FFmpeg through yt-dlp.
    """

    print("\n" + "=" * 60)
    print("[1/4] DOWNLOADING MEDIA")
    print("=" * 60)

    output_template = str(output_dir / "audio.%(ext)s")

    options = {
        "format": "bestaudio/best",

        "outtmpl": output_template,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],

        "noplaylist": True,

        # Keep yt-dlp output visible
        "quiet": False,
    }

    with YoutubeDL(options) as ydl:

        # Get metadata before downloading
        info = ydl.extract_info(url, download=False)

        metadata = {
            "id": info.get("id"),
            "title": info.get("title"),
            "channel": info.get("channel"),
            "uploader": info.get("uploader"),
            "duration": info.get("duration"),
            "webpage_url": info.get("webpage_url"),
            "upload_date": info.get("upload_date"),
            "description": info.get("description"),
        }

        # Save metadata
        metadata_file = output_dir / "metadata.json"

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"\nTitle: {metadata['title']}")
        print(f"Channel: {metadata['channel']}")
        print(f"Duration: {format_duration(metadata['duration'])}")

        # Download
        ydl.download([url])

    audio_file = output_dir / "audio.wav"

    if not audio_file.exists():
        raise FileNotFoundError(
            f"\nAudio download failed.\n"
            f"Expected file: {audio_file}"
        )

    print(f"\nAudio saved to:")
    print(audio_file)

    return audio_file, metadata


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe(
    audio_file: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """
    Transcribe audio using WhisperX on the NVIDIA GPU.
    """

    print("\n" + "=" * 60)
    print("[2/4] TRANSCRIPTION")
    print("=" * 60)

    device = "cuda"
    compute_type = "float16"

    print(f"Model        : {model_name}")
    print(f"Device       : {device}")
    print(f"Compute type : {compute_type}")
    print(f"Batch size   : {batch_size}")

    try:

        model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    except Exception as e:

        print("\nCUDA initialization failed:")
        print(e)

        print("\nFalling back to CPU...")

        device = "cpu"
        compute_type = "int8"

        model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    print(f"\nUsing device: {device}")

    print("\nLoading audio...")

    audio = whisperx.load_audio(
        str(audio_file)
    )

    print("Transcribing...")

    result = model.transcribe(
        audio,
        batch_size=batch_size,
    )

    # Store language detected by WhisperX
    language = result.get("language")

    print(f"\nDetected language: {language}")

    return result, language, device


# ============================================================
# WORD-LEVEL ALIGNMENT
# ============================================================

def align_transcription(
    result,
    audio_file: Path,
    language: str,
    device: str,
):
    """
    Align the transcription to obtain word-level timestamps.
    """

    print("\n" + "=" * 60)
    print("[3/4] WORD-LEVEL ALIGNMENT")
    print("=" * 60)

    print("Loading alignment model...")

    try:

        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )

    except Exception as e:

        print("\nAlignment model failed:")
        print(e)

        print("\nReturning unaligned transcription.")

        return result

    print("Loading audio for alignment...")

    audio = whisperx.load_audio(
        str(audio_file)
    )

    print("Aligning words...")

    aligned_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    print("Word alignment complete.")

    return aligned_result


# ============================================================
# SAVE TXT
# ============================================================

def save_txt(result, output_dir: Path):

    txt_file = output_dir / "transcript.txt"

    with open(
        txt_file,
        "w",
        encoding="utf-8"
    ) as f:

        for segment in result.get("segments", []):

            start = segment.get("start", 0)
            end = segment.get("end", 0)

            text = segment.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            f.write(
                f"[{format_timestamp(start)} --> "
                f"{format_timestamp(end)}]\n"
            )

            f.write(
                text + "\n\n"
            )

    return txt_file


# ============================================================
# SAVE SRT
# ============================================================

def save_srt(result, output_dir: Path):

    srt_file = output_dir / "transcript.srt"

    segments = result.get(
        "segments",
        []
    )

    with open(
        srt_file,
        "w",
        encoding="utf-8"
    ) as f:

        subtitle_number = 1

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

            f.write(
                f"{subtitle_number}\n"
            )

            f.write(
                f"{format_srt_timestamp(start)} --> "
                f"{format_srt_timestamp(end)}\n"
            )

            f.write(
                text + "\n\n"
            )

            subtitle_number += 1

    return srt_file


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    result,
    metadata,
    language,
    device,
    model_name,
    output_dir: Path,
):

    json_file = output_dir / "transcript.json"

    output = {

        "autotrim_version": "0.1",

        "source": {
            "url": metadata.get(
                "webpage_url"
            ),

            "video_id": metadata.get(
                "id"
            ),

            "title": metadata.get(
                "title"
            ),

            "channel": metadata.get(
                "channel"
            ),

            "duration": metadata.get(
                "duration"
            ),
        },

        "transcription": {

            "language": language,

            "model": model_name,

            "device": device,

            "segments": result.get(
                "segments",
                []
            ),
        },
    }

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    return json_file


# ============================================================
# TIME FORMATTING
# ============================================================

def format_timestamp(seconds):

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:06.3f}"
    )


def format_srt_timestamp(seconds):

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(
        seconds % 60
    )

    milliseconds = int(
        round(
            (seconds - int(seconds))
            * 1000
        )
    )

    # Handle rounding to 1000 ms
    if milliseconds >= 1000:

        milliseconds = 0
        secs += 1

        if secs >= 60:

            secs = 0
            minutes += 1

        if minutes >= 60:

            minutes = 0
            hours += 1

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


def format_duration(seconds):

    if not seconds:
        return "Unknown"

    seconds = int(seconds)

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    secs = seconds % 60

    if hours > 0:

        return (
            f"{hours}h "
            f"{minutes}m "
            f"{secs}s"
        )

    return (
        f"{minutes}m "
        f"{secs}s"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AutoTrim v0.1 - "
            "YouTube Video Transcriber"
        )
    )

    parser.add_argument(
        "url",
        help="YouTube video URL",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
        ],
        help=(
            "Whisper model "
            "(default: small)"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "WhisperX batch size "
            "(default: 4)"
        ),
    )

    parser.add_argument(
        "--output",
        default="output",
        help=(
            "Output directory "
            "(default: output)"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n")
    print("=" * 60)
    print("                 AUTOTRIM v0.1")
    print("             YouTube Transcriber")
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # 1. Download
        # ----------------------------------------------------

        audio_file, metadata = (
            download_audio(
                args.url,
                output_dir,
            )
        )

        # ----------------------------------------------------
        # 2. Transcribe
        # ----------------------------------------------------

        result, language, device = (
            transcribe(
                audio_file,
                args.model,
                args.batch_size,
            )
        )

        # ----------------------------------------------------
        # 3. Word alignment
        # ----------------------------------------------------

        result = align_transcription(
            result,
            audio_file,
            language,
            device,
        )

        # ----------------------------------------------------
        # 4. Save results
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("[4/4] SAVING RESULTS")
        print("=" * 60)

        txt_file = save_txt(
            result,
            output_dir,
        )

        srt_file = save_srt(
            result,
            output_dir,
        )

        json_file = save_json(
            result,
            metadata,
            language,
            device,
            args.model,
            output_dir,
        )

        # ----------------------------------------------------
        # Done
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("             AUTOTRIM COMPLETE")
        print("=" * 60)

        print("\nGenerated files:")

        print(
            f"  Audio      : "
            f"{audio_file}"
        )

        print(
            f"  Transcript : "
            f"{txt_file}"
        )

        print(
            f"  Subtitles  : "
            f"{srt_file}"
        )

        print(
            f"  JSON       : "
            f"{json_file}"
        )

        print(
            "\nGPU used: "
            f"{device}"
        )

    except Exception as e:

        print("\n" + "=" * 60)
        print("                 ERROR")
        print("=" * 60)

        print(
            f"\n{type(e).__name__}: {e}"
        )

        raise


if __name__ == "__main__":
    main()