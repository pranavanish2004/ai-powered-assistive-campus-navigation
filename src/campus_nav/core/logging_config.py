"""
Structured logging setup.

Design rationale
-----------------
This is a voice-only, often headless system — when something goes wrong in
the field, logs are the *only* debugging tool available (there's no UI to
screenshot). We therefore:

- Log to both console (development) and a rotating file (always), so nothing
  is lost when a terminal is closed.
- Support two output formats, chosen via `Settings.log_format`:
    * "text" — human-readable, for local development.
    * "json" — one JSON object per line, for production log aggregation
      (CloudWatch/Loki/ELK can index structured fields directly instead of
      regex-parsing free text).
- Include a `session_id` field on every record via a logging filter, so logs
  from concurrent user sessions can be separated during debugging even
  before true multi-tenancy lands (see architecture docs, Scalability).
- Configure the root logger exactly once per process (`configure_logging`
  is idempotent), from a single place, at process startup.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Any

from campus_nav.config.settings import get_settings

_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | session=%(session_id)s | %(message)s"

_RESERVED_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class _SessionContextFilter(logging.Filter):
    """Injects a `session_id` field into every log record, defaulting to
    'system' for logs emitted outside of a user session context."""

    def __init__(self) -> None:
        super().__init__()
        self.session_id = "system"

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "session_id"):
            record.session_id = self.session_id
        return True


class _JSONFormatter(logging.Formatter):
    """Renders each log record as a single JSON line. Any extra fields
    passed via `logger.info(msg, extra={...})` are included verbatim,
    which is what makes this useful for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "session_id": getattr(record, "session_id", "system"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS and key not in payload:
                payload[key] = value

        return json.dumps(payload, default=str)


_session_filter = _SessionContextFilter()


def set_current_session_id(session_id: str) -> None:
    """Update the session id attached to subsequent log records on this
    process. Call this when a navigation session starts."""
    _session_filter.session_id = session_id


def _build_formatter() -> logging.Formatter:
    settings = get_settings()
    if settings.log_format == "json":
        return _JSONFormatter()
    return logging.Formatter(_TEXT_FORMAT)


def configure_logging() -> None:
    """Configure the root logger. Call once at application startup, before
    any module-level logger is used. Safe to call multiple times (e.g. from
    multiple test modules) — subsequent calls are a no-op."""
    settings = get_settings()
    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    if root_logger.handlers:
        return  # already configured

    formatter = _build_formatter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_session_filter)
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "campus_nav.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_session_filter)
    root_logger.addHandler(file_handler)


def reset_logging() -> None:
    """Remove all handlers from the root logger. Intended for tests that
    need to reconfigure logging (e.g. to switch log_format) mid-suite;
    production code should not need this."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Use `__name__` as `name` at each
    call site so log lines are traceable to their source module."""
    return logging.getLogger(name)
