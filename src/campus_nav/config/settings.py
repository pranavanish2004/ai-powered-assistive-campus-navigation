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

    # --- Phase 1.3: Camera source selection ---
    camera_source_type: Literal["usb", "ip", "file"] = Field(
        default="usb",
        description="'usb' uses OpenCVCamera against a local device index. "
        "'ip' connects to a network camera stream (e.g. DroidCam, IP "
        "Webcam) via camera_ip_url. 'file' replays a video file via "
        "camera_video_file_path — useful for deterministic development "
        "and demos without physical hardware.",
    )
    camera_ip_url: str | None = Field(
        default=None,
        description="Stream URL for camera_source_type='ip', e.g. "
        "'http://192.168.1.5:4747/video' for a typical DroidCam setup.",
    )
    camera_video_file_path: str | None = Field(
        default=None,
        description="Path to a video file for camera_source_type='file'.",
    )
    camera_loop_video_file: bool = Field(
        default=True,
        description="If True, VideoFileCamera restarts from the beginning "
        "when it reaches the end of the file, rather than reporting "
        "unavailable — convenient for long-running demos/tests.",
    )
    frame_buffer_size: int = Field(
        default=5,
        gt=0,
        description="Maximum number of frames CameraManager holds in its "
        "ring buffer. Small on purpose: for navigation guidance, a stale "
        "frame is actively misleading, so the buffer favors recency over "
        "throughput and drops old frames rather than growing unbounded.",
    )
    vision_processing_fps: float = Field(
        default=5.0,
        gt=0,
        description="Target rate at which the vision pipeline processes "
        "frames, independent of the camera's native capture rate. Object "
        "detection/OCR/depth are all far more expensive than frame "
        "capture, so processing deliberately runs slower than capture.",
    )

    # --- Phase 1.3: Object detection ---
    detector_backend: Literal["yolov8", "mock"] = Field(
        default="mock",
        description="'mock' requires no model weights or ML dependencies "
        "and is the safe default so the application boots without the "
        "optional [vision] extra installed. 'yolov8' requires "
        "`pip install -e '.[vision]'` and a model weights file.",
    )
    yolo_model_path: str = Field(
        default="yolov8n.pt",
        description="Path or Ultralytics model name for YOLOv8 weights. "
        "Only read when detector_backend='yolov8'.",
    )

    # --- Phase 1.3: OCR ---
    ocr_backend: Literal["easyocr", "mock"] = Field(
        default="mock",
        description="'mock' is the safe default requiring no ML "
        "dependencies. 'easyocr' requires `pip install -e '.[vision]'`.",
    )
    ocr_confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    # --- Phase 1.3: Depth estimation ---
    depth_backend: Literal["midas", "heuristic"] = Field(
        default="heuristic",
        description="'heuristic' estimates distance from bounding-box size "
        "with no ML dependencies and no model download — a reasonable "
        "default for a pilot without a depth model deployed yet. 'midas' "
        "requires `pip install -e '.[vision]'` and downloads pretrained "
        "MiDaS weights via torch.hub on first use.",
    )
    depth_safe_distance_meters: float = Field(
        default=1.5,
        gt=0,
        description="An obstacle closer than this is considered near "
        "enough to warrant a safety announcement if it is also blocking "
        "the path (see vision.safety.obstacle_events).",
    )

    # --- Phase 1.3: Object tracking ---
    tracking_iou_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum IoU (intersection-over-union) between a new "
        "detection and an existing track for them to be matched as the "
        "same physical object across frames.",
    )
    tracking_max_missed_frames: int = Field(
        default=10,
        ge=0,
        description="A track survives this many consecutive frames with no "
        "matching detection before being dropped — handling brief "
        "occlusions (e.g. someone briefly walking between the camera and "
        "the tracked obstacle) without losing its persistent id.",
    )

    # --- Phase 1.4b: Dashboard integration ---
    dashboard_backend: Literal["iith_live", "mock"] = Field(
        default="mock",
        description="'mock' is the safe default requiring no network "
        "access. 'iith_live' calls the real, confirmed-public endpoints "
        "of the community-run IITH Dashboard (bus schedule, city bus "
        "schedule, mess menu) — see infrastructure/dashboard/"
        "iith_dashboard_repository.py for exactly which endpoints and why "
        "only those three. The dashboard is a replaceable enhancement, "
        "never a dependency the rest of the assistant relies on.",
    )
    dashboard_base_url: str = Field(
        default="https://api.iith.dev",
        description="Base URL for the IITH Dashboard API. Overridable for "
        "self-hosted forks or local testing against a fake server.",
    )
    dashboard_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Per-request timeout for dashboard calls. Kept short "
        "deliberately: this is a no-SLA, community-run service, and a slow "
        "dashboard response must never noticeably delay the assistant.",
    )

    # --- Phase 1.4 refinement: campus data ingestion ---
    campus_data_directory: str | None = Field(
        default=None,
        description="Optional directory scanned at startup for additional "
        "campus data files (.geojson, .csv, .yaml, .md) via "
        "CampusDataIngestionPipeline. Nodes/edges are merged into the "
        "loaded graph and knowledge entries into the RAG knowledge base, "
        "in addition to (never replacing) the base pilot graph and static "
        "knowledge file. None (default) means no additional ingestion — "
        "existing behavior is completely unchanged.",
    )

    # --- Phase 1.4a: RAG / knowledge base ---
    rag_backend: Literal["keyword", "tfidf", "chroma", "faiss"] = Field(
        default="tfidf",
        description="'tfidf' (TF-IDF cosine similarity) is the default: "
        "materially better ranking than plain 'keyword' overlap, at zero "
        "added dependency cost (uses only numpy, already core). 'chroma' "
        "and 'faiss' use real embedding models (optional [rag] extra) "
        "for genuine semantic search. All four implement the same "
        "RAGRepositoryInterface.",
    )
    rag_knowledge_path: str = Field(
        default="data/campus_knowledge.json",
        description="Path to the static supplementary knowledge base "
        "(accessibility info, emergency contacts, campus rules) merged "
        "with facts derived from the campus graph's landmark descriptions.",
    )
    rag_top_k: int = Field(
        default=3, gt=0, description="Number of knowledge documents retrieved per query."
    )

    # --- Phase 1.4c: Pluggable MLLM provider ---
    mllm_provider: Literal["mock", "anthropic", "openai", "gemini", "local_llama"] = Field(
        default="mock",
        description="'mock' requires no API key or network access and is "
        "the safe default. Real providers require the optional [llm] "
        "extra (except 'local_llama', which talks to a local Ollama "
        "server over HTTP using the already-core httpx dependency).",
    )
    mllm_model: str | None = Field(
        default=None,
        description="Model name/id passed to the selected provider. None "
        "means 'use that provider's own default', since each provider's "
        "reasonable default changes independently of this codebase.",
    )
    mllm_max_tokens: int = Field(default=512, gt=0)
    mllm_api_key: str | None = Field(
        default=None,
        description="API key passed explicitly to the provider. If None, "
        "each provider SDK falls back to its own standard environment "
        "variable (e.g. ANTHROPIC_API_KEY) — this field is a convenience "
        "override, not a requirement.",
    )
    mllm_base_url: str | None = Field(
        default=None,
        description="Override base URL for the selected provider. Used by "
        "'local_llama' (defaults to http://localhost:11434 if unset) and "
        "optionally by other providers pointing at a compatible proxy.",
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
