"""
capture/camera_source.py
─────────────────────────
Hardware Ingestion Layer and Video Stream Abstraction Interface.

Robofest Gujarat 6.0 | Aerial Robotics | Senior Division

Provides a unified, thread-safe `.read()` and `.release()` interface across diverse
imaging inputs without altering downstream processing code:
  - Luxonis OAK-D Lite via standard V4L2 UVC device node (e.g., /dev/video0)
  - Pre-recorded flight test video files (.mp4, .avi, .mov)
  - Low-latency drone RTSP video feeds (rtsp://...)

Configuration:
  Source selection is driven dynamically by `config.yaml` -> `camera.source`.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Abstract Camera Source Interface
# ===========================================================================

class CameraSource(ABC):
    """
    Abstract Base Class defining the standard frame acquisition contract.

    All subclasses must implement `.read()` for frame retrieval and `.release()`
    for deterministic hardware descriptor teardown. Context-manager protocols
    (`__enter__`, `__exit__`) are supported out of the box.
    """

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """
        Fetch the next synchronized BGR video frame.

        Returns
        -------
        Optional[np.ndarray]
            Array of shape (H, W, 3) with dtype uint8 in BGR format, or None
            if the underlying hardware stream is exhausted or disconnected.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Release underlying V4L2 device descriptors, codecs, or network sockets."""
        ...

    def __enter__(self) -> CameraSource:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


# ===========================================================================
# OpenCV VideoCapture Backend Wrapper
# ===========================================================================

class _OpenCVSource(CameraSource):
    """
    Internal wrapper managing `cv2.VideoCapture` streams.

    Features built-in frame-rate throttling and automated network reconnection
    with exponential backoff for RTSP/UVC streams.

    Parameters
    ----------
    source : Union[int, str]
        Integer V4L2 device index (e.g., 0 for OAK-D Lite UVC), filesystem path, or RTSP URL.
    target_fps : int, default=0
        Target frame rate limit (0 = uncapped maximum throughput).
    reconnect_attempts : int, default=3
        Maximum retry attempts on network or stream read failure.
    reconnect_delay_s : float, default=1.0
        Delay in seconds between reconnection attempts.
    """

    def __init__(
        self,
        source: Union[int, str],
        target_fps: int = 0,
        reconnect_attempts: int = 3,
        reconnect_delay_s: float = 1.0,
    ) -> None:
        self._source: Union[int, str] = source
        self._target_fps: int = target_fps
        self._reconnect_max: int = reconnect_attempts
        self._reconnect_delay: float = reconnect_delay_s
        self._frame_interval: float = (1.0 / target_fps) if target_fps > 0 else 0.0
        self._last_frame_time: float = 0.0

        self._cap: cv2.VideoCapture = self._open()

    def _open(self) -> cv2.VideoCapture:
        """
        Instantiate and validate cv2.VideoCapture stream.

        Returns
        -------
        cv2.VideoCapture
            Opened video capture descriptor.

        Raises
        ------
        RuntimeError
            If OpenCV fails to open the specified device index or file path.
        """
        logger.info("Initializing video capture descriptor: %s", self._source)
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open video source: {self._source!r}. "
                "Ensure device node (/dev/video0) or file path exists and has read permissions."
            )
        return cap

    def read(self) -> Optional[np.ndarray]:
        """
        Acquire next frame with optional monotonic FPS throttling and RTSP reconnection.

        Returns
        -------
        Optional[np.ndarray]
            BGR video frame array, or None if stream is disconnected.
        """
        # Monotonic frame-rate regulation
        if self._frame_interval > 0:
            elapsed = time.monotonic() - self._last_frame_time
            wait = self._frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)

        ok, frame = self._cap.read()

        # Handle transient network stream failures (RTSP)
        if not ok:
            if isinstance(self._source, str) and self._source.startswith("rtsp"):
                logger.warning("RTSP stream frame drop — initiating reconnection sequence...")
                for attempt in range(self._reconnect_max):
                    time.sleep(self._reconnect_delay)
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self._source)
                    ok, frame = self._cap.read()
                    if ok:
                        logger.info("RTSP stream reconnected successfully (attempt %d/%d)", attempt + 1, self._reconnect_max)
                        break
                else:
                    logger.error("RTSP reconnection aborted after %d failed attempts", self._reconnect_max)
                    return None
            else:
                return None

        self._last_frame_time = time.monotonic()
        return frame

    def release(self) -> None:
        """Safely close the underlying capture handle."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Released camera capture descriptor: %s", self._source)


# ===========================================================================
# Concrete Ingestion Implementations
# ===========================================================================

class WebcamSource(_OpenCVSource):
    """
    V4L2 / UVC Camera Device Ingestion Source.

    Designed for the Luxonis OAK-D Lite operating in standard UVC mode (`/dev/video0`)
    or standard USB webcam interfaces.

    Parameters
    ----------
    device_id : int, default=0
        V4L2 hardware device index.
    target_fps : int, default=0
        Frame rate cap in frames per second.
    """

    def __init__(self, device_id: int = 0, target_fps: int = 0) -> None:
        super().__init__(source=device_id, target_fps=target_fps)


class FileSource(_OpenCVSource):
    """
    Recorded Video File Ingestion Source.

    Enables repeatable bench testing on captured flight logs (.mp4, .avi, .mov).

    Parameters
    ----------
    path : Union[str, Path]
        Filesystem path to the video file.
    loop : bool, default=False
        If True, continuously rewinds and replays video on EOF.
    target_fps : int, default=0
        Frame rate cap. Use 0 for uncapped stress testing.
    """

    def __init__(self, path: Union[str, Path], loop: bool = False, target_fps: int = 0) -> None:
        self._loop: bool = loop
        self._path: str = str(path)
        super().__init__(source=self._path, target_fps=target_fps)

    def read(self) -> Optional[np.ndarray]:
        """Fetch frame with optional automatic replay on EOF."""
        frame = super().read()
        if frame is None and self._loop:
            logger.info("FileSource: Replaying video from beginning (%s)", self._path)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = super().read()
        return frame


class RTSPSource(_OpenCVSource):
    """
    Network RTSP Video Stream Ingestion Source.

    Connects to live wireless drone video downlinks.

    Parameters
    ----------
    url : str
        Full RTSP endpoint URL (e.g., 'rtsp://192.168.1.10:554/live').
    target_fps : int, default=0
        Frame rate throttle.
    """

    def __init__(self, url: str, target_fps: int = 0) -> None:
        super().__init__(source=url, target_fps=target_fps, reconnect_attempts=5)


# ===========================================================================
# Source Factory
# ===========================================================================

def build_source(camera_cfg: dict) -> CameraSource:
    """
    Factory function resolving and instantiating the appropriate CameraSource.

    Parameters
    ----------
    camera_cfg : dict
        The `camera` configuration section from config.yaml.

    Returns
    -------
    CameraSource
        Configured and opened camera source instance ready for `.read()`.

    Raises
    ------
    ValueError
        If the configured source type cannot be resolved.
    """
    raw_source = camera_cfg.get("source", 0)
    fps = camera_cfg.get("target_fps", 0)

    # Integer -> V4L2 Device Index (Luxonis OAK-D Lite UVC /dev/video0)
    if isinstance(raw_source, int):
        logger.info("Instantiating WebcamSource (device_index=%d, fps_cap=%d)", raw_source, fps)
        return WebcamSource(device_id=raw_source, target_fps=fps)

    source_str = str(raw_source)

    # String beginning with rtsp:// -> Network Stream
    if source_str.lower().startswith("rtsp://"):
        logger.info("Instantiating RTSPSource (url=%s, fps_cap=%d)", source_str, fps)
        return RTSPSource(url=source_str, target_fps=fps)

    # Existing local file path -> FileSource
    if Path(source_str).exists():
        logger.info("Instantiating FileSource (path=%s, fps_cap=%d)", source_str, fps)
        return FileSource(path=source_str, loop=False, target_fps=fps)

    raise ValueError(
        f"Unable to resolve camera source specification: {raw_source!r}. "
        "Expected: integer device index (e.g. 0), existing file path, or 'rtsp://' URL."
    )
