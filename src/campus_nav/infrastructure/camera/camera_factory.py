"""
Camera source factory.

Centralizes "which CameraInterface implementation should we use", driven by
`Settings.camera_source_type`, so callers (the vision API startup code)
don't need their own if/elif chain over concrete camera classes -- exactly
the same role `JSONGraphRepository` plays for graph loading in Phase 1.2.
"""

from __future__ import annotations

from campus_nav.config.settings import Settings
from campus_nav.core.exceptions import ConfigurationError
from campus_nav.infrastructure.camera.ip_camera import IPCameraSource
from campus_nav.infrastructure.camera.opencv_camera import OpenCVCamera
from campus_nav.infrastructure.camera.video_file_camera import VideoFileCamera
from campus_nav.interfaces.camera_interface import CameraInterface


def create_camera(settings: Settings) -> CameraInterface:
    """Build the CameraInterface implementation selected by
    `settings.camera_source_type`. Raises ConfigurationError immediately
    (fail-fast, matching the Phase 1.1 configuration philosophy) if the
    selected source type is missing its required settings."""
    if settings.camera_source_type == "usb":
        return OpenCVCamera(
            device_index=settings.camera_index,
            frame_width=settings.camera_frame_width,
            frame_height=settings.camera_frame_height,
        )

    if settings.camera_source_type == "ip":
        if not settings.camera_ip_url:
            raise ConfigurationError("camera_source_type='ip' requires camera_ip_url to be set.")
        return IPCameraSource(stream_url=settings.camera_ip_url)

    if settings.camera_source_type == "file":
        if not settings.camera_video_file_path:
            raise ConfigurationError(
                "camera_source_type='file' requires camera_video_file_path to be set."
            )
        return VideoFileCamera(
            file_path=settings.camera_video_file_path,
            loop=settings.camera_loop_video_file,
        )

    # Unreachable given Settings' Literal type validates this at parse
    # time, but kept explicit rather than falling through silently.
    raise ConfigurationError(f"Unknown camera_source_type: {settings.camera_source_type!r}")
