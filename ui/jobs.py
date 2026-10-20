"""Persistent, sequential job runner for the local dashboard."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlparse

import psutil

from config import (
    GEMINI_MODEL,
    OUTPUT_DIR,
    PROJECT_ROOT,
    REPURPOSER_MAX_RETRIES,
    REPURPOSER_TEMPERATURE,
    SUPPORTED_TRANSCRIPTION_MODELS,
    SUPPORTED_TRANSCRIPTION_LANGUAGES,
    TRANSCRIPTION_BATCH_SIZE,
    TRANSCRIPTION_LANGUAGE,
    TRANSCRIPTION_MODEL,
)


UI_DATA_DIR = Path(__file__).resolve().parent / "data"
JOBS_DIR = UI_DATA_DIR / "jobs"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def inside_output(path: Path) -> bool:
    try:
        path.resolve().relative_to(OUTPUT_DIR.resolve())
        return True
    except ValueError:
        return False


class JobManager:
    def __init__(self) -> None:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, dict] = {}
        self._active_process: subprocess.Popen | None = None
        self._cancelled: set[str] = set()
        self._load_jobs()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="autotrim-job-worker",
            daemon=True,
        )
        self._worker.start()

    def _job_dir(self, job_id: str) -> Path:
        return JOBS_DIR / job_id

    def _save_locked(self, job: dict) -> None:
        job_dir = self._job_dir(job["id"])
        job_dir.mkdir(parents=True, exist_ok=True)
        temporary = job_dir / "status.tmp"
        temporary.write_text(
            json.dumps(job, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(job_dir / "status.json")

    def _load_jobs(self) -> None:
        for status_file in JOBS_DIR.glob("*/status.json"):
            try:
                job = json.loads(status_file.read_text(encoding="utf-8"))
                if not isinstance(job, dict) or not job.get("id"):
                    continue
            except (OSError, json.JSONDecodeError):
                continue

            if job.get("status") == "running":
                job.update(
                    status="failed",
                    stage="Interrupted",
                    error="Dashboard stopped while this job was running. You can queue it again.",
                    completed_at=now_iso(),
                )
                self._save_locked(job)
            self._jobs[job["id"]] = job
            if job.get("status") == "queued":
                self._queue.put(job["id"])

    @staticmethod
    def _validate_url(value: object) -> str:
        if not isinstance(value, str) or len(value) > 2048:
            raise ValueError("Enter a valid video URL.")
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Enter a full URL beginning with http:// or https://.")
        hostname = (parsed.hostname or "").lower()
        if not (hostname == "youtube.com" or hostname.endswith(".youtube.com") or hostname == "youtu.be"):
            raise ValueError("The dashboard accepts YouTube and youtu.be links.")
        return value.strip()

    def _prepare_video(self, url_value: object, payload: dict) -> tuple[str, dict]:
        url = self._validate_url(url_value)
        model = payload.get("transcription_model", TRANSCRIPTION_MODEL)
        if model not in SUPPORTED_TRANSCRIPTION_MODELS:
            raise ValueError("Choose a supported WhisperX transcription model.")
        language = payload.get("transcription_language", TRANSCRIPTION_LANGUAGE)
        if not isinstance(language, str) or language not in SUPPORTED_TRANSCRIPTION_LANGUAGES:
            raise ValueError("Choose auto-detect, Hindi, or Hindi + English for transcription.")
        try:
            batch_size = int(payload.get("batch_size", TRANSCRIPTION_BATCH_SIZE))
        except (TypeError, ValueError) as exc:
            raise ValueError("Batch size must be a number from 1 to 32.") from exc
        if not 1 <= batch_size <= 32:
            raise ValueError("Batch size must be between 1 and 32.")
        gemini_model = str(payload.get("gemini_model") or GEMINI_MODEL).strip()
        if not gemini_model or len(gemini_model) > 120:
            raise ValueError("Enter a valid Gemini model name.")
        temperature, retries = self._generation_settings(payload)
        allow_low_quality = payload.get("allow_low_quality_transcript", False)
        if not isinstance(allow_low_quality, bool):
            raise ValueError("Transcript quality override must be true or false.")
        output_dir = self._output_dir(payload.get("output_folder", ""))

        return url, {
            "transcription_model": model,
            "transcription_language": language,
            "batch_size": batch_size,
            "gemini_model": gemini_model,
            "temperature": temperature,
            "generation_retries": retries,
            "allow_low_quality_transcript": allow_low_quality,
            "output_dir": str(output_dir),
        }

    @staticmethod
    def _output_dir(value: object) -> Path:
        if not isinstance(value, str) or len(value) > 240:
            raise ValueError("Output folder must be a path under output/.")
        relative = Path(value.strip()) if value.strip() else Path(".")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Choose a relative output folder inside output/.")
        resolved = (OUTPUT_DIR / relative).resolve()
        if not inside_output(resolved):
            raise ValueError("Output folder must stay inside output/.")
        return resolved

    def enqueue_video(self, payload: dict) -> dict:
        url, settings = self._prepare_video(payload.get("url"), payload)
        return self._enqueue(
            mode="video",
            label=url,
            settings=settings,
            url=url,
        )

    def enqueue_videos(self, payload: dict) -> list[dict]:
        urls = payload.get("urls")
        if not isinstance(urls, list) or not 1 <= len(urls) <= 10:
            raise ValueError("Paste between 1 and 10 video links, one per line.")
        prepared = [self._prepare_video(value, payload) for value in urls]
        with self._lock:
            pending = sum(job.get("status") in {"queued", "running"} for job in self._jobs.values())
            if pending + len(prepared) > 25:
                raise ValueError("There is not enough queue space for those links. Wait for jobs to finish or add fewer links.")
        return [
            self._enqueue(mode="video", label=url, settings=settings, url=url)
            for url, settings in prepared
        ]

    def enqueue_regeneration(self, payload: dict) -> dict:
        raw_path = payload.get("transcript_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Choose a transcript to regenerate from.")
        transcript = Path(raw_path).expanduser().resolve()
        if not inside_output(transcript) or transcript.suffix.lower() != ".txt":
            raise ValueError("Regeneration is limited to transcript .txt files inside AutoTrim output.")
        if not transcript.is_file():
            raise ValueError("That transcript file no longer exists.")
        gemini_model = str(payload.get("gemini_model") or GEMINI_MODEL).strip()
        if not gemini_model or len(gemini_model) > 120:
            raise ValueError("Enter a valid Gemini model name.")
        temperature, retries = self._generation_settings(payload)
        allow_low_quality = payload.get("allow_low_quality_transcript", False)
        if not isinstance(allow_low_quality, bool):
            raise ValueError("Transcript quality override must be true or false.")
        return self._enqueue(
            mode="regenerate",
            label=transcript.name,
            settings={"gemini_model": gemini_model, "temperature": temperature, "generation_retries": retries, "allow_low_quality_transcript": allow_low_quality},
            transcript_txt=str(transcript),
        )

    @staticmethod
    def _generation_settings(payload: dict) -> tuple[float, int]:
        try:
            temperature = float(payload.get("temperature", REPURPOSER_TEMPERATURE))
            retries = int(payload.get("generation_retries", REPURPOSER_MAX_RETRIES))
        except (TypeError, ValueError) as exc:
            raise ValueError("Temperature and retry count must be valid numbers.") from exc
        if not 0 <= temperature <= 1:
            raise ValueError("Generation temperature must be between 0 and 1.")
        if not 1 <= retries <= 10:
            raise ValueError("Generation retries must be between 1 and 10.")
        return temperature, retries

    def _enqueue(self, mode: str, label: str, settings: dict, **inputs) -> dict:
        with self._lock:
            pending = sum(job.get("status") in {"queued", "running"} for job in self._jobs.values())
            if pending >= 25:
                raise ValueError("The queue is full. Wait for a few jobs to finish before adding more.")
            job_id = uuid.uuid4().hex[:12]
            job = {
                "id": job_id,
                "mode": mode,
                "label": label,
                "status": "queued",
                "stage": "Waiting in queue",
                "created_at": now_iso(),
                "started_at": None,
                "completed_at": None,
                "settings": settings,
                "inputs": inputs,
                "transcript_txt": inputs.get("transcript_txt"),
                "transcript_json": None,
                "content_dir": None,
                "exit_code": None,
                "error": None,
                "resources": None,
            }
            self._jobs[job_id] = job
            (self._job_dir(job_id) / "log.txt").parent.mkdir(parents=True, exist_ok=True)
            self._save_locked(job)
            self._queue.put(job_id)
            return dict(job)

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.get("created_at", ""), reverse=True)
            return [dict(job) for job in jobs[:80]]

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def get_log(self, job_id: str, max_lines: int = 500) -> str:
        if not self.get_job(job_id):
            raise KeyError(job_id)
        log_file = self._job_dir(job_id) / "log.txt"
        try:
            return "".join(deque(log_file.open(encoding="utf-8", errors="replace"), maxlen=max_lines))
        except OSError:
            return ""

    def get_preview(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        transcript = self._safe_existing(job.get("transcript_txt"))
        transcript_json = self._safe_existing(job.get("transcript_json"))
        content_dir = self._safe_existing(job.get("content_dir"), directory=True)
        transcript_text = ""
        content = None
        if transcript:
            transcript_text = transcript.read_text(encoding="utf-8", errors="replace")[:1_500_000]
        if content_dir:
            content_file = content_dir / "repurposed.json"
            if content_file.is_file():
                try:
                    content = json.loads(content_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    content = None
        return {
            "transcript_txt": str(transcript) if transcript else None,
            "transcript_json": str(transcript_json) if transcript_json else None,
            "transcript": transcript_text,
            "transcript_data": None,
            "content_dir": str(content_dir) if content_dir else None,
            "content": content,
            "files": self.list_output_files(job),
        }

    @staticmethod
    def _safe_existing(raw_path: str | None, directory: bool = False) -> Path | None:
        if not raw_path:
            return None
        path = Path(raw_path).resolve()
        if not inside_output(path) or not path.exists():
            return None
        if directory and not path.is_dir():
            return None
        if not directory and not path.is_file():
            return None
        return path

    def list_output_files(self, job: dict) -> list[dict]:
        files: list[Path] = []
        for key in ("transcript_txt", "transcript_json"):
            candidate = self._safe_existing(job.get(key))
            if candidate:
                files.append(candidate)
        if job.get("transcript_txt"):
            transcript_path = self._safe_existing(job.get("transcript_txt"))
            if transcript_path:
                for candidate in transcript_path.parent.glob("transcript.srt"):
                    files.append(candidate)
        content_dir = self._safe_existing(job.get("content_dir"), directory=True)
        if content_dir:
            files.extend(path for path in content_dir.iterdir() if path.is_file() and path.suffix.lower() in {".md", ".json", ".txt", ".srt"})
        unique = {str(path.resolve()): path.resolve() for path in files if inside_output(path)}
        result = []
        for path in sorted(unique.values(), key=lambda item: item.name.lower()):
            result.append({
                "name": path.name,
                "path": str(path),
                "size": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            })
        return result

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.get("status") == "queued":
                job.update(status="cancelled", stage="Cancelled", completed_at=now_iso())
                self._save_locked(job)
                return True
            process = self._active_process if job.get("status") == "running" else None
        if process and process.poll() is None:
            with self._lock:
                self._cancelled.add(job_id)
            try:
                root = psutil.Process(process.pid)
                children = root.children(recursive=True)
                for child in children:
                    child.terminate()
                root.terminate()
                psutil.wait_procs([*children, root], timeout=3)
            except psutil.Error:
                process.terminate()
            return True
        return False

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.update(changes)
                self._save_locked(job)

    def _append_log(self, job_id: str, line: str) -> None:
        with (self._job_dir(job_id) / "log.txt").open("a", encoding="utf-8") as log_file:
            log_file.write(line)
            if not line.endswith("\n"):
                log_file.write("\n")

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                job = self.get_job(job_id)
                if not job or job.get("status") != "queued":
                    continue
                self._run(job)
            finally:
                self._queue.task_done()

    @staticmethod
    def _command(job: dict) -> list[str]:
        python = os.fspath(Path(sys.executable))
        if job["mode"] == "regenerate":
            command = [
                python, "-u", str(PROJECT_ROOT / "repurposer.py"),
                job["inputs"]["transcript_txt"], "--model", job["settings"]["gemini_model"],
                "--temperature", str(job["settings"].get("temperature", REPURPOSER_TEMPERATURE)),
                "--max-retries", str(job["settings"].get("generation_retries", REPURPOSER_MAX_RETRIES)),
            ]
            if job["settings"].get("allow_low_quality_transcript"):
                command.append("--allow-low-quality-transcript")
            return command
        command = [
            python, "-u", str(PROJECT_ROOT / "autotrim.py"),
            job["inputs"]["url"],
            "--transcription-model", job["settings"]["transcription_model"],
            "--transcription-language", job["settings"].get("transcription_language", TRANSCRIPTION_LANGUAGE),
            "--batch-size", str(job["settings"]["batch_size"]),
            "--output-dir", job["settings"].get("output_dir", str(OUTPUT_DIR)),
            "--model", job["settings"]["gemini_model"],
            "--temperature", str(job["settings"].get("temperature", REPURPOSER_TEMPERATURE)),
            "--generation-retries", str(job["settings"].get("generation_retries", REPURPOSER_MAX_RETRIES)),
        ]
        if job["settings"].get("allow_low_quality_transcript"):
            command.append("--allow-low-quality-transcript")
        return command

    def _set_paths_from_log(self, job_id: str, line: str) -> None:
        cleaned = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        changes: dict[str, str] = {}
        if cleaned.startswith("JSON :"):
            value = cleaned.partition(":")[2].strip()
            candidate = Path(value)
            if inside_output(candidate):
                changes["transcript_json"] = str(candidate.resolve())
        elif cleaned.startswith("TXT  :"):
            value = cleaned.partition(":")[2].strip()
            candidate = Path(value)
            if inside_output(candidate):
                changes["transcript_txt"] = str(candidate.resolve())
                changes["content_dir"] = str(candidate.resolve().parent / "content")
        if changes:
            self._update(job_id, **changes)

    @staticmethod
    def _stage_for_line(line: str) -> str | None:
        value = line.lower()
        if "[1/2] transcription" in value:
            return "Transcribing"
        if "existing transcript found" in value:
            return "Reusing transcript"
        if "[2/2] content repurposing" in value or "gemini] generating content" in value:
            return "Generating content"
        if "skipping gemini generation" in value:
            return "Reusing generated content"
        if "pipeline complete" in value:
            return "Finishing"
        if "analyzing source" in value or "fetching video information" in value:
            return "Fetching video details"
        return None

    def _sample_resources(self, process: subprocess.Popen, stop: threading.Event, job_id: str) -> None:
        try:
            root_process = psutil.Process(process.pid)
            root_process.cpu_percent(None)
        except (psutil.Error, OSError):
            root_process = None
        gpu_checked = 0
        while not stop.wait(1):
            cpu = memory_mb = None
            memory_percent = None
            try:
                if root_process:
                    processes = [root_process, *root_process.children(recursive=True)]
                    cpu = round(sum(proc.cpu_percent(None) for proc in processes if proc.is_running()), 1)
                    memory_mb = round(sum(proc.memory_info().rss for proc in processes if proc.is_running()) / (1024 * 1024))
                    memory_percent = round(memory_mb * 1024 * 1024 / psutil.virtual_memory().total * 100, 1)
            except (psutil.Error, OSError):
                pass
            resources = {"process_cpu_percent": cpu, "process_memory_mb": memory_mb, "process_memory_percent": memory_percent}
            gpu_checked += 1
            if gpu_checked >= 4:
                gpu_checked = 0
                nvidia_smi = shutil.which("nvidia-smi")
                if nvidia_smi:
                    try:
                        gpu = subprocess.run(
                            [nvidia_smi, "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=2, check=True,
                        ).stdout.strip().splitlines()[0].split(",")
                        resources.update({"gpu_percent": int(gpu[0].strip()), "gpu_memory_mb": int(gpu[1].strip()), "gpu_memory_total_mb": int(gpu[2].strip())})
                    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
                        resources["gpu_percent"] = None
            self._update(job_id, resources=resources)

    def _run(self, job: dict) -> None:
        job_id = job["id"]
        self._update(job_id, status="running", stage="Starting", started_at=now_iso(), error=None)
        command = self._command(job)
        self._append_log(job_id, "$ " + subprocess.list2cmdline(command))
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self._lock:
                self._active_process = process
            stop_sampler = threading.Event()
            sampler = threading.Thread(target=self._sample_resources, args=(process, stop_sampler, job_id), daemon=True)
            sampler.start()
            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(job_id, line.rstrip("\r\n"))
                self._set_paths_from_log(job_id, line)
                stage = self._stage_for_line(line)
                if stage:
                    self._update(job_id, stage=stage)
            exit_code = process.wait()
            stop_sampler.set()
            sampler.join(timeout=2)
            with self._lock:
                self._active_process = None
            final = self.get_job(job_id) or job
            with self._lock:
                was_cancelled = job_id in self._cancelled
                self._cancelled.discard(job_id)
            if was_cancelled:
                self._update(job_id, status="cancelled", stage="Cancelled", exit_code=exit_code, completed_at=now_iso())
                return
            if exit_code == 0:
                if final.get("transcript_txt") and not final.get("content_dir"):
                    self._update(job_id, content_dir=str(Path(final["transcript_txt"]).parent / "content"))
                self._update(job_id, status="completed", stage="Complete", exit_code=0, completed_at=now_iso())
            else:
                cancelled = process.returncode is not None and process.returncode < 0
                transcript_needs_review = exit_code == 7
                if transcript_needs_review:
                    self._update(
                        job_id,
                        status="failed",
                        stage="Transcript needs review",
                        exit_code=exit_code,
                        error=(
                            "Article generation was stopped because the transcript looks corrupted. "
                            "Open transcript.txt from this job, correct it or choose a different "
                            "transcription model, then regenerate."
                        ),
                        completed_at=now_iso(),
                    )
                    return
                self._update(
                    job_id,
                    status="cancelled" if cancelled else "failed",
                    stage="Cancelled" if cancelled else "Failed",
                    exit_code=exit_code,
                    error=None if cancelled else f"AutoTrim exited with code {exit_code}. See the job log for details.",
                    completed_at=now_iso(),
                )
        except Exception as exc:
            with self._lock:
                self._active_process = None
            self._append_log(job_id, f"Dashboard runner error: {exc}")
            self._update(job_id, status="failed", stage="Failed", error=str(exc), completed_at=now_iso())


manager = JobManager()
