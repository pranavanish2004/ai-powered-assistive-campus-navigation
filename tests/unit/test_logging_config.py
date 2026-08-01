"""Tests for core.logging_config."""

import json
import logging

import pytest

from campus_nav.core.logging_config import (
    _JSONFormatter,
    _SessionContextFilter,
    configure_logging,
    get_logger,
    reset_logging,
    set_current_session_id,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _cleanup_logging():
    yield
    reset_logging()


def test_configure_logging_is_idempotent():
    configure_logging()
    handlers_after_first_call = len(logging.getLogger().handlers)
    configure_logging()
    handlers_after_second_call = len(logging.getLogger().handlers)
    assert handlers_after_first_call == handlers_after_second_call
    assert handlers_after_first_call > 0


def test_get_logger_returns_named_logger():
    logger = get_logger("campus_nav.test_module")
    assert logger.name == "campus_nav.test_module"


def _make_record(msg: str = "test message") -> logging.LogRecord:
    return logging.LogRecord(
        name="campus_nav.test_module",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_session_id_filter_injects_session_id_into_records():
    session_filter = _SessionContextFilter()
    session_filter.session_id = "session-abc-123"
    record = _make_record()
    session_filter.filter(record)
    assert record.session_id == "session-abc-123"


def test_session_id_defaults_to_system_when_unset():
    session_filter = _SessionContextFilter()
    record = _make_record()
    session_filter.filter(record)
    assert record.session_id == "system"


def test_session_id_filter_does_not_overwrite_existing_session_id():
    session_filter = _SessionContextFilter()
    session_filter.session_id = "process-default"
    record = _make_record()
    record.session_id = "explicitly-set"
    session_filter.filter(record)
    assert record.session_id == "explicitly-set"


def test_set_current_session_id_updates_module_level_filter():
    from campus_nav.core.logging_config import _session_filter

    set_current_session_id("my-session")
    assert _session_filter.session_id == "my-session"
    set_current_session_id("system")  # reset for other tests


def test_json_formatter_produces_valid_json_with_expected_fields():
    formatter = _JSONFormatter()
    record = _make_record("hello world")
    record.session_id = "session-xyz"
    output = formatter.format(record)

    parsed = json.loads(output)
    assert parsed["message"] == "hello world"
    assert parsed["session_id"] == "session-xyz"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "campus_nav.test_module"


def test_json_formatter_includes_extra_fields():
    formatter = _JSONFormatter()
    record = _make_record("hello")
    record.session_id = "system"
    record.custom_field = "custom_value"
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["custom_field"] == "custom_value"


def test_json_formatter_includes_exception_info():
    formatter = _JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert "exception" in parsed
    assert "ValueError" in parsed["exception"]
