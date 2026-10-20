"""Loopback-only HTTP server for the AutoTrim dashboard."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import sys
from urllib.parse import parse_qs, unquote, urlparse

from dotenv import load_dotenv, set_key, unset_key
from config import (
    GEMINI_MODEL,
    GEMINI_API_KEY_ENV,
    OUTPUT_DIR,
    PROJECT_ROOT,
    REPURPOSER_MAX_RETRIES,
    REPURPOSER_TEMPERATURE,
    SUPPORTED_TRANSCRIPTION_MODELS,
    TRANSCRIPTION_BATCH_SIZE,
    TRANSCRIPTION_MODEL,
)
from ui.jobs import inside_output, manager


ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


def save_gemini_api_key(value: object, env_file: Path = ENV_FILE) -> None:
    if not isinstance(value, str):
        raise ValueError("Enter a valid Gemini API key.")
    key = value.strip()
    if not key or len(key) > 8192 or "\n" in key or "\r" in key:
        raise ValueError("Enter a valid Gemini API key (1–8192 characters).")
    success, _ = set_key(str(env_file), GEMINI_API_KEY_ENV, key, quote_mode="always")
    if not success:
        raise OSError("Could not save the Gemini API key to the local .env file.")
    os.environ[GEMINI_API_KEY_ENV] = key


def remove_gemini_api_key(env_file: Path = ENV_FILE) -> None:
    if env_file.is_file():
        success, _ = unset_key(str(env_file), GEMINI_API_KEY_ENV)
        if success is False:
            raise OSError("Could not remove the Gemini API key from the local .env file.")
    os.environ.pop(GEMINI_API_KEY_ENV, None)


UI_DIR = Path(__file__).resolve().parent
STATIC_DIR = UI_DIR / "static"
INDEX_FILE = UI_DIR / "index.html"


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "AutoTrimDashboard/1.0"

    def log_message(self, format_string: str, *args) -> None:
        print(f"[dashboard] {self.address_string()} {format_string % args}")

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        super().end_headers()

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: int, data: object) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _body_json(self) -> dict:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid request size.") from exc
        if size > 32_000:
            raise ValueError("Request is too large.")
        try:
            value = json.loads(self.rfile.read(size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request must contain valid JSON.") from exc
        if not isinstance(value, dict):
            raise ValueError("Request must be a JSON object.")
        return value

    def _library(self) -> list[dict]:
        if not OUTPUT_DIR.exists():
            return []
        items = []
        for transcript in OUTPUT_DIR.rglob("transcript.txt"):
            if not transcript.is_file():
                continue
            content_dir = transcript.parent / "content"
            content_files = []
            if content_dir.is_dir():
                content_files = [
                    {"name": path.name, "path": str(path.resolve()), "size": path.stat().st_size}
                    for path in content_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".md", ".json", ".txt"}
                ]
            items.append({
                "title": transcript.parent.name,
                "transcript_path": str(transcript.resolve()),
                "transcript_json": str((transcript.parent / "transcript.json").resolve()) if (transcript.parent / "transcript.json").is_file() else None,
                "content_files": content_files,
                "has_content": bool(content_files),
                "modified_at": transcript.stat().st_mtime,
            })
        return sorted(items, key=lambda item: item["modified_at"], reverse=True)[:200]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in {"/", "/index.html"}:
                return self._send(200, INDEX_FILE.read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/static/"):
                relative = Path(unquote(path.removeprefix("/static/")))
                asset = (STATIC_DIR / relative).resolve()
                if not asset.is_relative_to(STATIC_DIR.resolve()) or not asset.is_file():
                    return self._json(404, {"error": "Asset not found."})
                media_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                return self._send(200, asset.read_bytes(), f"{media_type}; charset=utf-8")
            if path == "/api/state":
                jobs = manager.list_jobs()
                active = next((job for job in jobs if job.get("status") == "running"), None)
                return self._json(200, {
                    "jobs": jobs,
                    "active_job_id": active.get("id") if active else None,
                    "queued_count": sum(job.get("status") == "queued" for job in jobs),
                    "library": self._library(),
                    "settings": {
                        "transcription_models": sorted(SUPPORTED_TRANSCRIPTION_MODELS),
                        "transcription_model": TRANSCRIPTION_MODEL,
                        "batch_size": TRANSCRIPTION_BATCH_SIZE,
                        "gemini_model": GEMINI_MODEL,
                        "temperature": REPURPOSER_TEMPERATURE,
                        "generation_retries": REPURPOSER_MAX_RETRIES,
                        "output_dir": str(OUTPUT_DIR),
                        "gemini_api_key_configured": bool(os.getenv(GEMINI_API_KEY_ENV)),
                    },
                })
            if path == "/api/library/preview":
                raw = parse_qs(parsed.query).get("path", [""])[0]
                transcript = Path(unquote(raw)).resolve()
                if not inside_output(transcript) or transcript.suffix.lower() != ".txt" or not transcript.is_file():
                    return self._json(404, {"error": "Transcript not found."})
                content_dir = transcript.parent / "content"
                content = None
                content_json = content_dir / "repurposed.json"
                if content_json.is_file():
                    try:
                        content = json.loads(content_json.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        content = None
                files = []
                for candidate in [transcript, transcript.parent / "transcript.json", transcript.parent / "transcript.srt"]:
                    if candidate.is_file():
                        files.append({"name": candidate.name, "path": str(candidate.resolve()), "size": candidate.stat().st_size})
                if content_dir.is_dir():
                    files.extend({"name": item.name, "path": str(item.resolve()), "size": item.stat().st_size} for item in content_dir.iterdir() if item.is_file() and item.suffix.lower() in {".md", ".json", ".txt", ".srt"})
                transcript_json = transcript.parent / "transcript.json"
                return self._json(200, {
                    "transcript_txt": str(transcript),
                    "transcript_json": str(transcript_json) if transcript_json.is_file() else None,
                    "transcript": transcript.read_text(encoding="utf-8", errors="replace")[:1_500_000],
                    "transcript_data": None,
                    "content_dir": str(content_dir) if content_dir.is_dir() else None,
                    "content": content,
                    "files": files,
                })
            if path.startswith("/api/jobs/"):
                parts = path.strip("/").split("/")
                if len(parts) < 3:
                    return self._json(404, {"error": "Job not found."})
                job_id = parts[2]
                if len(parts) == 4 and parts[3] == "log":
                    try:
                        return self._json(200, {"log": manager.get_log(job_id)})
                    except KeyError:
                        return self._json(404, {"error": "Job not found."})
                if len(parts) == 4 and parts[3] == "preview":
                    try:
                        return self._json(200, manager.get_preview(job_id))
                    except KeyError:
                        return self._json(404, {"error": "Job not found."})
                job = manager.get_job(job_id)
                return self._json(200, job) if job else self._json(404, {"error": "Job not found."})
            if path == "/api/download":
                raw = parse_qs(parsed.query).get("path", [""])[0]
                target = Path(unquote(raw)).resolve()
                if not inside_output(target) or target.suffix.lower() not in {".txt", ".srt", ".json", ".md"} or not target.is_file():
                    return self._json(404, {"error": "Output file not found."})
                data = target.read_bytes()
                safe_name = target.name.replace('"', "").replace("\r", "").replace("\n", "")
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
                self.end_headers()
                self.wfile.write(data)
                return
            return self._json(404, {"error": "Not found."})
        except (OSError, ValueError) as exc:
            self._json(500, {"error": str(exc)})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._body_json()
            if path == "/api/settings/gemini-key":
                save_gemini_api_key(payload.get("api_key"))
                return self._json(200, {"saved": True, "configured": True})
            if path == "/api/settings/gemini-key/remove":
                remove_gemini_api_key()
                return self._json(200, {"saved": True, "configured": False})
            if path == "/api/jobs/batch":
                return self._json(202, {"jobs": manager.enqueue_videos(payload)})
            if path == "/api/jobs":
                return self._json(202, manager.enqueue_video(payload))
            if path == "/api/regenerate":
                return self._json(202, manager.enqueue_regeneration(payload))
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[3]
                return self._json(200, {"cancelled": manager.cancel(job_id)})
            return self._json(404, {"error": "Not found."})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except KeyError:
            self._json(404, {"error": "Job not found."})
        except OSError:
            # Avoid returning file or credential details to the browser.
            self._json(500, {"error": "Could not update the local Gemini API key."})

    def _host_allowed(self) -> bool:
        host = self.headers.get("Host", "").split(":", 1)[0].strip("[]").lower()
        return host in {"127.0.0.1", "localhost"}

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        # Reject Host-header rebinding attempts; the service listens only on loopback.
        if not self._host_allowed():
            self.send_error(400, "Invalid Host header")
            return False
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoTrim local dashboard")
    parser.add_argument("--port", type=int, default=8765, help="Local port (default: 8765)")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print("AutoTrim dashboard is running locally.")
    print(f"Open http://127.0.0.1:{args.port} in your browser. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
