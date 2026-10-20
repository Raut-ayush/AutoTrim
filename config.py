"""Shared defaults and paths for the AutoTrim command-line tools."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Transcription
TRANSCRIPTION_MODEL = "small"
TRANSCRIPTION_BATCH_SIZE = 4
TRANSCRIPTION_LANGUAGE = "auto"
SUPPORTED_TRANSCRIPTION_LANGUAGES = frozenset({"auto", "hi", "hi-en"})
SUPPORTED_TRANSCRIPTION_MODELS = frozenset(
    {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
)

# Content generation
GEMINI_MODEL = "gemini-3.8-flash"
OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_HOST = "http://localhost:11434"
GENERATION_TEMPERATURE = 0.2
REPURPOSER_TEMPERATURE = 0.3
CONTENT_FORMATTER_TEMPERATURE = 0.1
GEMINI_MAX_RETRIES = 4
REPURPOSER_MAX_RETRIES = 5

# Relative paths resolve from the project, not the caller's working directory.
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
