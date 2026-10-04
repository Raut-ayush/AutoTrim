import argparse
import json
import logging
import os
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch
import whisperx
from yt_dlp import YoutubeDL


# ============================================================
# AUTOTRIM CONFIGURATION
# ============================================================

AUTOTRIM_VERSION = "0.2"

DEFAULT_MODEL = "small"
DEFAULT_BATCH_SIZE = 4

BASE_OUTPUT_DIR = Path("output")
LOG_DIR = Path("logs")
BENCHMARK_DIR = Path("benchmark")

SUPPORTED_MODELS = {
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
}


# ============================================================
# TERMINAL UI
# ============================================================

WIDTH = 62


def print_line(char="─"):
    print(char * WIDTH)


def print_header(title):
    print()
    print("╔" + "═" * (WIDTH - 2) + "╗")
    print("║" + title.center(WIDTH - 2) + "║")
    print("╚" + "═" * (WIDTH - 2) + "╝")


def print_section(number, total, title):
    print()
    print(f"[{number}/{total}] {title}")
    print_line()


def print_status(label, value, symbol="✓"):
    print(f"  {symbol} {label:<30} {value}")


def print_key_value(label, value):
    print(f"  {label:<20}: {value}")


# ============================================================
# LOGGING
# ============================================================

def setup_logging(run_id):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / f"{run_id}.log"

    logger = logging.getLogger("autotrim")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )

    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )

    logger.addHandler(file_handler)

    return logger, log_file


# ============================================================
# HELPERS
# ============================================================

def sanitize_filename(name, max_length=120):
    """
    Convert a video title into a Windows-safe folder name.
    """

    name = str(name).strip()

    # Remove Windows-invalid characters
    name = re.sub(r'[<>:"/\\|?*]', "", name)

    # Remove control characters
    name = re.sub(r"[\x00-\x1f]", "", name)

    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)

    # Windows does not like trailing dots/spaces
    name = name.rstrip(". ")

    if not name:
        name = "untitled_video"

    # Avoid Windows reserved names
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }

    if name.upper() in reserved:
        name = f"_{name}"

    return name[:max_length].rstrip(". ")


def format_duration(seconds):
    if seconds is None:
        return "Unknown"

    seconds = int(round(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def format_seconds(seconds):
    return f"{seconds:.2f}s"


def get_video_type(duration):
    if duration is None:
        return "Unknown"

    if duration <= 60:
        return "Short"

    return "Long-form"


def get_system_info():
    gpu_name = None
    gpu_memory_gb = None

    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)

            gpu_memory_gb = round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2,
            )
        except Exception:
            pass

    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": round(
            psutil.virtual_memory().total / (1024 ** 3),
            2
        ),
        "gpu": gpu_name,
        "gpu_vram_gb": gpu_memory_gb,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def get_peak_gpu_memory_gb():
    if not torch.cuda.is_available():
        return None

    try:
        return round(
            torch.cuda.max_memory_allocated(0)
            / (1024 ** 3),
            3
        )
    except Exception:
        return None


def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def normalize_text(text):
    """
    Basic deterministic cleanup.

    Important:
    We do NOT attempt to rewrite the transcript here.
    The raw ASR text should remain untouched in transcript.json.
    """

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def merge_transcript_into_paragraphs(segments):
    """
    Convert timestamped WhisperX segments into readable paragraphs.

    This is intentionally deterministic.

    We don't use an LLM here because transcript formatting
    should not alter the meaning of the ASR output.

    A new paragraph is created when:
      - there is a significant silence
      - the current paragraph becomes reasonably long
      - the text appears to reach a natural sentence boundary
    """

    if not segments:
        return []

    paragraphs = []

    current_text = []
    current_start = None
    current_end = None

    MAX_PARAGRAPH_CHARS = 700
    SILENCE_THRESHOLD = 1.2

    previous_end = None

    for segment in segments:

        text = normalize_text(segment.get("text", ""))

        if not text:
            continue

        start = segment.get("start")
        end = segment.get("end")

        if current_start is None:
            current_start = start

        # Detect meaningful silence
        silence = 0

        if previous_end is not None and start is not None:
            silence = start - previous_end

        current_length = len(" ".join(current_text))

        should_break = (
            current_text
            and (
                silence >= SILENCE_THRESHOLD
                or current_length >= MAX_PARAGRAPH_CHARS
            )
        )

        if should_break:
            paragraphs.append({
                "text": " ".join(current_text).strip(),
                "start": current_start,
                "end": current_end,
            })

            current_text = []
            current_start = start

        current_text.append(text)

        if end is not None:
            current_end = end

        previous_end = end

    if current_text:
        paragraphs.append({
            "text": " ".join(current_text).strip(),
            "start": current_start,
            "end": current_end,
        })

    return paragraphs


# ============================================================
# YOUTUBE DOWNLOAD
# ============================================================

def extract_video_info(url):
    """
    Extract metadata without downloading the media.
    """

    ydl_opts = {
        "quiet": True,
        "no_warnings": False,
        "noplaylist": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return info


def download_audio(url, output_dir, logger):

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting video metadata")

    info = extract_video_info(url)

    video_id = info.get("id", "unknown")
    title = info.get("title", "Untitled")
    channel = (
        info.get("channel")
        or info.get("uploader")
        or "Unknown"
    )
    duration = info.get("duration")

    logger.info(
        "Video: %s | %s | %s",
        video_id,
        title,
        format_duration(duration),
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    print("Downloading audio...")

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_file = output_dir / "audio.wav"

    if not audio_file.exists():
        raise FileNotFoundError(
            f"Audio extraction failed: {audio_file}"
        )

    metadata = {
        "url": url,
        "video_id": video_id,
        "title": title,
        "channel": channel,
        "duration": duration,
        "duration_formatted": format_duration(duration),
        "video_type": get_video_type(duration),
        "thumbnail": info.get("thumbnail"),
        "upload_date": info.get("upload_date"),
        "description": info.get("description"),
    }

    metadata_file = output_dir / "metadata.json"

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "info": info,
        "metadata": metadata,
        "audio_file": audio_file,
        "metadata_file": metadata_file,
    }


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe(
    audio_file,
    model_name,
    batch_size,
    logger,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = (
        "float16"
        if device == "cuda"
        else "int8"
    )

    print("Loading audio...")

    audio = whisperx.load_audio(str(audio_file))

    print("Loading WhisperX model...")

    logger.info(
        "Loading model=%s device=%s compute_type=%s batch_size=%s",
        model_name,
        device,
        compute_type,
        batch_size,
    )

    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
    )

    print("Transcribing...")

    start_time = time.perf_counter()

    result = model.transcribe(
        audio,
        batch_size=batch_size,
    )

    elapsed = time.perf_counter() - start_time

    detected_language = result.get(
        "language",
        "unknown"
    )

    logger.info(
        "Detected language=%s",
        detected_language,
    )

    print(
        f"Detected language: {detected_language}"
    )

    return {
        "result": result,
        "device": device,
        "compute_type": compute_type,
        "model": model_name,
        "time": elapsed,
        "language": detected_language,
    }


# ============================================================
# WORD ALIGNMENT
# ============================================================

def align_transcription(
    transcription_result,
    audio_file,
    logger,
):

    device = transcription_result["device"]
    language = transcription_result["language"]

    print("Loading alignment model...")

    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )

    print("Loading audio for alignment...")

    audio = whisperx.load_audio(str(audio_file))

    print("Aligning words...")

    start_time = time.perf_counter()

    aligned_result = whisperx.align(
        transcription_result["result"]["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Word alignment completed in %.3fs",
        elapsed,
    )

    return aligned_result, elapsed


# ============================================================
# SAVE TRANSCRIPT TXT
# ============================================================

def save_transcript_txt(
    aligned_result,
    output_file,
):
    """
    Save clean paragraph-style transcript.

    No timestamps.

    Timestamps remain available in:
      - transcript.json
      - transcript.srt
    """

    segments = aligned_result.get(
        "segments",
        []
    )

    paragraphs = merge_transcript_into_paragraphs(
        segments
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        for paragraph in paragraphs:
            f.write(paragraph["text"])
            f.write("\n\n")

    return paragraphs


# ============================================================
# SAVE SRT
# ============================================================

def seconds_to_srt_time(seconds):

    if seconds is None:
        seconds = 0

    milliseconds = int(round(seconds * 1000))

    hours = milliseconds // 3_600_000

    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000

    milliseconds %= 60_000

    secs = milliseconds // 1000

    milliseconds %= 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


def save_srt(
    aligned_result,
    output_file,
):

    segments = aligned_result.get(
        "segments",
        []
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        subtitle_index = 1

        for segment in segments:

            start = segment.get("start")
            end = segment.get("end")
            text = normalize_text(
                segment.get("text", "")
            )

            if start is None or end is None:
                continue

            if not text:
                continue

            f.write(
                f"{subtitle_index}\n"
            )

            f.write(
                f"{seconds_to_srt_time(start)} --> "
                f"{seconds_to_srt_time(end)}\n"
            )

            f.write(
                f"{text}\n\n"
            )

            subtitle_index += 1


# ============================================================
# SAVE JSON
# ============================================================

def save_transcript_json(
    metadata,
    aligned_result,
    transcription_info,
    output_file,
    processing_info,
    system_info,
):

    segments = aligned_result.get(
        "segments",
        []
    )

    clean_segments = []

    for index, segment in enumerate(segments):

        clean_segments.append({
            "id": index,
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": normalize_text(
                segment.get("text", "")
            ),
            "words": segment.get(
                "words",
                []
            ),
        })

    transcript_data = {
        "autotrim_version": AUTOTRIM_VERSION,

        "source": {
            "url": metadata.get("url"),
            "video_id": metadata.get("video_id"),
            "title": metadata.get("title"),
            "channel": metadata.get("channel"),
            "duration": metadata.get("duration"),
            "video_type": metadata.get("video_type"),
        },

        "transcription": {
            "language": transcription_info["language"],
            "model": transcription_info["model"],
            "device": transcription_info["device"],
            "compute_type": transcription_info["compute_type"],
            "segments": clean_segments,
        },

        "processing": processing_info,

        "system": system_info,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            transcript_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return transcript_data


# ============================================================
# BENCHMARK DATA
# ============================================================

def save_benchmark_json(
    benchmark_data,
    benchmark_file,
):

    benchmark_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        benchmark_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            benchmark_data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def append_benchmark_csv(
    benchmark_data,
    csv_file,
):

    import csv

    csv_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = csv_file.exists()

    fieldnames = [
        "run_id",
        "timestamp",
        "video_id",
        "title",
        "channel",
        "video_type",
        "duration_seconds",
        "duration_formatted",

        "model",
        "batch_size",
        "language",

        "device",
        "compute_type",
        "gpu",
        "gpu_vram_gb",
        "peak_gpu_memory_gb",

        "download_time",
        "transcription_time",
        "alignment_time",
        "output_time",
        "total_time",

        "rtf",

        "word_count",
        "segment_count",
        "paragraph_count",

        "status",
        "error",
    ]

    with open(
        csv_file,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            key: benchmark_data.get(key)
            for key in fieldnames
        })


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="AutoTrim YouTube Transcriber"
    )

    parser.add_argument(
        "url",
        help="YouTube video or Shorts URL",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=sorted(SUPPORTED_MODELS),
        help=f"WhisperX model (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # RUN IDENTIFICATION
    # --------------------------------------------------------

    run_start = time.perf_counter()

    timestamp = datetime.now()

    run_id = timestamp.strftime(
        "%Y%m%d_%H%M%S"
    )

    logger, log_file = setup_logging(
        run_id
    )

    system_info = get_system_info()

    reset_gpu_memory_stats()

    print_header(
        f"AUTOTRIM v{AUTOTRIM_VERSION}"
    )

    print(
        "YouTube → Audio → Transcript → Alignment"
    )

    # --------------------------------------------------------
    # PREPARE METADATA
    # --------------------------------------------------------

    try:

        print_section(
            1,
            4,
            "ANALYZING SOURCE"
        )

        source_start = time.perf_counter()

        print("Fetching video information...")

        info = extract_video_info(
            args.url
        )

        video_id = info.get(
            "id",
            "unknown"
        )

        title = info.get(
            "title",
            "Untitled"
        )

        channel = (
            info.get("channel")
            or info.get("uploader")
            or "Unknown"
        )

        duration = info.get(
            "duration"
        )

        video_type = get_video_type(
            duration
        )

        # Folder uses title as requested
        folder_name = sanitize_filename(
            title
        )

        output_dir = (
            BASE_OUTPUT_DIR /
            folder_name
        )

        # Handle duplicate title
        if output_dir.exists():

            existing_metadata = (
                output_dir /
                "metadata.json"
            )

            existing_video_id = None

            if existing_metadata.exists():

                try:

                    with open(
                        existing_metadata,
                        "r",
                        encoding="utf-8",
                    ) as f:
                        old_metadata = json.load(f)

                    existing_video_id = old_metadata.get(
                        "video_id"
                    )

                except Exception:
                    pass

            if existing_video_id != video_id:

                output_dir = (
                    BASE_OUTPUT_DIR /
                    f"{folder_name} [{video_id}]"
                )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        source_time = (
            time.perf_counter()
            - source_start
        )

        print()
        print_key_value(
            "Title",
            title
        )
        print_key_value(
            "Channel",
            channel
        )
        print_key_value(
            "Video ID",
            video_id
        )
        print_key_value(
            "Type",
            video_type
        )
        print_key_value(
            "Duration",
            format_duration(duration)
        )
        print_key_value(
            "Output",
            output_dir
        )

        logger.info(
            "Source analyzed in %.3fs",
            source_time
        )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        print_section(
            2,
            4,
            "DOWNLOADING MEDIA"
        )

        download_start = time.perf_counter()

        downloaded = download_audio(
            args.url,
            output_dir,
            logger,
        )

        download_time = (
            time.perf_counter()
            - download_start
        )

        audio_file = downloaded[
            "audio_file"
        ]

        metadata = downloaded[
            "metadata"
        ]

        print_status(
            "Audio downloaded",
            format_seconds(download_time)
        )

        # ----------------------------------------------------
        # TRANSCRIPTION
        # ----------------------------------------------------

        print_section(
            3,
            4,
            "TRANSCRIPTION + ALIGNMENT"
        )

        print_key_value(
            "Model",
            args.model
        )

        print_key_value(
            "Batch size",
            args.batch_size
        )

        print_key_value(
            "Device",
            "CUDA"
            if torch.cuda.is_available()
            else "CPU"
        )

        if torch.cuda.is_available():

            print_key_value(
                "GPU",
                system_info["gpu"]
            )

            print_key_value(
                "VRAM",
                f"{system_info['gpu_vram_gb']} GB"
            )

        # Transcription
        transcription_start = time.perf_counter()

        transcription_info = transcribe(
            audio_file,
            args.model,
            args.batch_size,
            logger,
        )

        transcription_time = (
            time.perf_counter()
            - transcription_start
        )

        print_status(
            "Transcription",
            format_seconds(
                transcription_time
            )
        )

        # Alignment
        alignment_start = time.perf_counter()

        aligned_result, alignment_time = (
            align_transcription(
                transcription_info,
                audio_file,
                logger,
            )
        )

        alignment_time = (
            time.perf_counter()
            - alignment_start
        )

        print_status(
            "Word alignment",
            format_seconds(
                alignment_time
            )
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        print_section(
            4,
            4,
            "GENERATING OUTPUTS"
        )

        output_start = time.perf_counter()

        txt_file = (
            output_dir /
            "transcript.txt"
        )

        srt_file = (
            output_dir /
            "transcript.srt"
        )

        json_file = (
            output_dir /
            "transcript.json"
        )

        paragraphs = save_transcript_txt(
            aligned_result,
            txt_file,
        )

        save_srt(
            aligned_result,
            srt_file,
        )

        # Word count
        all_words = []

        for segment in aligned_result.get(
            "segments",
            []
        ):

            words = segment.get(
                "words",
                []
            )

            for word in words:

                word_text = word.get(
                    "word",
                    ""
                )

                if word_text.strip():
                    all_words.append(
                        word_text
                    )

        word_count = len(
            all_words
        )

        segment_count = len(
            aligned_result.get(
                "segments",
                []
            )
        )

        processing_so_far = (
            time.perf_counter()
            - run_start
        )

        processing_info = {
            "download_seconds": round(
                download_time,
                3
            ),

            "transcription_seconds": round(
                transcription_time,
                3
            ),

            "alignment_seconds": round(
                alignment_time,
                3
            ),
        }

        save_transcript_json(
            metadata,
            aligned_result,
            transcription_info,
            json_file,
            processing_info,
            system_info,
        )

        output_time = (
            time.perf_counter()
            - output_start
        )

        total_time = (
            time.perf_counter()
            - run_start
        )

        # ----------------------------------------------------
        # BENCHMARK
        # ----------------------------------------------------

        duration_seconds = (
            float(duration)
            if duration
            else 0
        )

        rtf = None

        if duration_seconds > 0:
            rtf = round(
                total_time /
                duration_seconds,
                4
            )

        peak_gpu_memory = (
            get_peak_gpu_memory_gb()
        )

        benchmark_data = {

            "run_id": run_id,

            "timestamp": timestamp.isoformat(
                timespec="seconds"
            ),

            "video_id": video_id,
            "title": title,
            "channel": channel,
            "video_type": video_type,

            "duration_seconds": duration,
            "duration_formatted":
                format_duration(duration),

            "model": args.model,
            "batch_size": args.batch_size,
            "language":
                transcription_info["language"],

            "device":
                transcription_info["device"],

            "compute_type":
                transcription_info["compute_type"],

            "gpu":
                system_info["gpu"],

            "gpu_vram_gb":
                system_info["gpu_vram_gb"],

            "peak_gpu_memory_gb":
                peak_gpu_memory,

            "download_time":
                round(download_time, 3),

            "transcription_time":
                round(transcription_time, 3),

            "alignment_time":
                round(alignment_time, 3),

            "output_time":
                round(output_time, 3),

            "total_time":
                round(total_time, 3),

            "rtf": rtf,

            "word_count":
                word_count,

            "segment_count":
                segment_count,

            "paragraph_count":
                len(paragraphs),

            "status":
                "SUCCESS",

            "error":
                None,
        }

        benchmark_json = (
            BENCHMARK_DIR /
            f"{run_id}.json"
        )

        benchmark_csv = (
            BENCHMARK_DIR /
            "runs.csv"
        )

        save_benchmark_json(
            benchmark_data,
            benchmark_json,
        )

        append_benchmark_csv(
            benchmark_data,
            benchmark_csv,
        )

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        print()
        print("╔" + "═" * (WIDTH - 2) + "╗")
        print(
            "║"
            + "AUTOTRIM COMPLETE".center(
                WIDTH - 2
            )
            + "║"
        )
        print("╚" + "═" * (WIDTH - 2) + "╝")

        print()
        print("RESULT")
        print_line()

        print_status(
            "Audio",
            str(audio_file)
        )

        print_status(
            "Transcript TXT",
            str(txt_file)
        )

        print_status(
            "Subtitles SRT",
            str(srt_file)
        )

        print_status(
            "Transcript JSON",
            str(json_file)
        )

        print_status(
            "Benchmark CSV",
            str(benchmark_csv)
        )

        print_status(
            "Run log",
            str(log_file)
        )

        print()
        print("PERFORMANCE")
        print_line()

        print_key_value(
            "Download",
            format_seconds(download_time)
        )

        print_key_value(
            "Transcription",
            format_seconds(
                transcription_time
            )
        )

        print_key_value(
            "Alignment",
            format_seconds(
                alignment_time
            )
        )

        print_key_value(
            "Output",
            format_seconds(
                output_time
            )
        )

        print_key_value(
            "Total",
            format_seconds(
                total_time
            )
        )

        if rtf is not None:

            speed = (
                1 / rtf
                if rtf > 0
                else 0
            )

            print_key_value(
                "RTF",
                f"{rtf:.3f}x "
                f"({speed:.2f}x realtime)"
            )

        print()
        print("TRANSCRIPT")
        print_line()

        print_key_value(
            "Language",
            transcription_info[
                "language"
            ]
        )

        print_key_value(
            "Words",
            word_count
        )

        print_key_value(
            "Segments",
            segment_count
        )

        print_key_value(
            "Paragraphs",
            len(paragraphs)
        )

        print()
        print(
            f"GPU used: "
            f"{transcription_info['device']}"
        )

        logger.info(
            "AutoTrim completed successfully"
        )

        return 0

    except Exception as exc:

        total_time = (
            time.perf_counter()
            - run_start
        )

        logger.exception(
            "AutoTrim failed"
        )

        print()
        print("╔" + "═" * (WIDTH - 2) + "╗")
        print(
            "║"
            + "AUTOTRIM FAILED".center(
                WIDTH - 2
            )
            + "║"
        )
        print("╚" + "═" * (WIDTH - 2) + "╝")

        print()
        print(f"Error: {exc}")

        print(
            f"Time before failure: "
            f"{total_time:.2f}s"
        )

        print(
            f"Detailed log: {log_file}"
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())