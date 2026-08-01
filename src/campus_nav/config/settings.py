"""
Centralized, type-safe application configuration.

Design rationale
-----------------
All configuration is loaded and *validated* at process startup through a
single Settings object. This gives us two properties that matter for a
safety-adjacent system:

1. Fail-fast: a missing/malformed environment variable raises immediately on
   boot, rather than surfacing as a mysterious None deep inside the
   Navigation Engine at runtime (e.g. mid-navigation).
2. Single source of truth: every module reads configuration through
   `get_settings()` instead of calling `os.environ` directly, so there is
   exactly one place that knows how config is sourced. Swapping the source
   later (e.g. to a secrets manager in production) touches this file only.

`get_settings()` is memoized (lru_cache) so settings are parsed once per
process. `reload_settings()` clears the cache — used by tests that need to
re-read environment variables mutated mid-test.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration, sourced from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- General ---
    app_env: Literal["development", "testing", "production"] = Field(
        default="development",
        description="Deployment environment. Controls logging verbosity and "
        "which infrastructure implementations are wired in.",
    )
    app_name: str = "campus-nav-assistant"

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_dir: Path = Path("logs")
    log_format: Literal["text", "json"] = Field(
        default="text",
        description="'text' is human-readable for local development; 'json' "
        "emits structured, machine-parseable lines suited to log aggregation "
        "in production (e.g. CloudWatch, Loki).",
    )

    # --- Database ---
    database_url: str = Field(
        default="sqlite:///./data/campus_nav.db",
        description="SQLAlchemy connection string. SQLite for local/pilot "
        "use; swap to a postgresql:// URL for multi-user production "
        "deployment without changing application code.",
    )

    # --- Camera ---
    camera_index: int = Field(default=0, ge=0)
    camera_frame_width: int = Field(default=640, gt=0)
    camera_frame_height: int = Field(default=480, gt=0)

    # --- GPS / Localization ---
    gps_provider: Literal["mock", "serial_nmea"] = Field(
        default="mock",
        description="'mock' returns a simulated coordinate for indoor "
        "development. Real hardware providers are added when campus "
        "deployment hardware is selected.",
    )
    gps_snap_max_distance_meters: float = Field(
        default=30.0,
        gt=0,
        description="Maximum distance a raw GPS fix may be from the nearest "
        "graph node/edge and still be 'snapped' to the campus graph. Beyond "
        "this, the fix is treated as unreliable (e.g. GPS drift, indoors).",
    )

    # --- Speech Recognition ---
    # NOTE: Whisper itself is an optional dependency (see pyproject.toml
    # `[speech]` extra) and is not installed by default in Phase 1.1/1.2.
    # These settings are read only by the speech module once it is added.
    whisper_model_size: Literal["tiny", "base", "small", "medium", "large"] = "base"
    whisper_language: str = "en"

    # --- Text-to-Speech ---
    tts_engine: Literal["pyttsx3"] = Field(
        default="pyttsx3",
        description="pyttsx3 is fully offline, which matters for the "
        "safety-critical alert lane: it must never depend on network "
        "availability.",
    )
    tts_rate_wpm: int = Field(default=175, gt=0)

    # --- Detection / Safety thresholds ---
    min_detection_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detections below this confidence are treated as "
        "'uncertain', not ignored silently.",
    )
    off_route_warning_meters: float = Field(
        default=15.0,
        gt=0,
        description="If the user's snapped position is farther than this "
        "from the active route, the Safety Engine raises an off-route "
        "warning and the Navigation Engine may trigger rerouting.",
    )

    @field_validator("log_dir")
    @classmethod
    def _ensure_log_dir_exists(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton (parsed once, cached)."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the memoized Settings and re-read from the environment.
    Intended for tests that mutate environment variables mid-run; production
    code should not need this."""
    get_settings.cache_clear()
    return get_settings()
