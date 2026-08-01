"""Tests for core.logging_config."""

import logging

from campus_nav.core.logging_config import (
    configure_logging,
    get_logger,
    set_current_session_id,
)


def test_configure_logging_is_idempotent():
    """Calling configure_logging() multiple times must not duplicate
    handlers (which would otherwise cause every log line to print N times)."""
    configure_logging()
    handlers_after_first_call = len(logging.getLogger().handlers)
    configure_logging()
    handlers_after_second_call = len(logging.getLogger().handlers)
    assert handlers_after_first_call == handlers_after_second_call
    assert handlers_after_first_call > 0


def test_get_logger_returns_named_logger():
    logger = get_logger("campus_nav.test_module")
    assert logger.name == "campus_nav.test_module"


def test_session_id_filter_injects_session_id_into_records():
    """caplog installs its own pytest-managed handler which bypasses our
    custom filter, so we exercise the filter directly rather than through
    the logging pipeline — this is what actually determines correctness."""
    from campus_nav.core.logging_config import _SessionContextFilter

    session_filter = _SessionContextFilter()
    session_filter.session_id = "session-abc-123"

    record = logging.LogRecord(
        name="campus_nav.test_module",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )

    session_filter.filter(record)
    assert record.session_id == "session-abc-123"


def test_session_id_defaults_to_system_when_unset():
    from campus_nav.core.logging_config import _SessionContextFilter

    session_filter = _SessionContextFilter()
    record = logging.LogRecord(
        name="campus_nav.test_module",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )

    session_filter.filter(record)
    assert record.session_id == "system"
