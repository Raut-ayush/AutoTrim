"""Shared CLI exit codes and user-facing error types."""


class ExitCode:
    SUCCESS = 0
    USAGE = 2
    CONFIGURATION = 3
    DEPENDENCY = 4
    SERVICE = 5
    PROCESSING = 6
    TRANSCRIPT_QUALITY = 7
    INTERRUPTED = 130


class AutoTrimError(Exception):
    """An expected failure that can be explained directly to the user."""

    exit_code = ExitCode.PROCESSING


class ConfigurationError(AutoTrimError):
    exit_code = ExitCode.CONFIGURATION


class DependencyError(AutoTrimError):
    exit_code = ExitCode.DEPENDENCY


class ServiceError(AutoTrimError):
    exit_code = ExitCode.SERVICE


class TranscriptQualityError(AutoTrimError):
    """Transcript has corruption signals that need review before generation."""

    exit_code = ExitCode.TRANSCRIPT_QUALITY
