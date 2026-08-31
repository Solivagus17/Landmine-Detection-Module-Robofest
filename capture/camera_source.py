"""
capture/camera_source.py
─────────────────────────
Abstracted camera source interface for the Landmine Detection Module.

Provides a common `.read()` interface regardless of whether the frame comes
from a webcam, a video file, or an RTSP drone camera stream. Swap sources
by changing `camera.source` in config.yaml — no code changes needed.

Usage
-----
    source = build_source(cfg["camera"])
    while True:
        frame = source.read()
        if frame is None:
            break
        # process frame ...
    source.release()

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class CameraSource(ABC):
    """
    Abstract camera source.

    All subclasses must implement `read()` and `release()`.
    """

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """
        Read the next frame.

        Returns
        -------
        np.ndarray or None
            BGR frame array (H, W, 3), or None if the source is exhausted /
            unavailable.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Release underlying resources (camera handle, file handle, etc.)."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


# ---------------------------------------------------------------------------
# OpenCV VideoCapture backend (webcam + file + RTSP)
# ---------------------------------------------------------------------------

class _OpenCVSource(CameraSource):
    """
    Internal wrapper around cv2.VideoCapture that handles webcam, video file,
    and RTSP URL sources with a unified interface.
    """

    def __init__(
        self,
        source,                         # int (device) | str (file/URL)
        target_fps: int = 0,
        reconnect_attempts: int = 3,    # RTSP reconnect tries
        reconnect_delay_s: float = 1.0,
    ):
        self._source           = source
        self._target_fps       = target_fps
        self._reconnect_max    = reconnect_attempts
        self._reconnect_delay  = reconnect_delay_s
        self._frame_interval   = (1.0 / target_fps) if target_fps > 0 else 0.0
        self._last_frame_time  = 0.0

        self._cap = self._open()

    def _open(self) -> cv2.VideoCapture:
        logger.info("Opening camera source: %s", self._source)
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera source: {self._source!r}. "
                "Check that the device/file/URL exists and is accessible."
            )
        return cap

    def read(self) -> Optional[np.ndarray]:
        # FPS throttle — skip sleep if frame_interval is 0 (uncapped)
        if self._frame_interval > 0:
            elapsed = time.monotonic() - self._last_frame_time
            wait    = self._frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)

        ok, frame = self._cap.read()

        # RTSP reconnect on transient failure
        if not ok:
            if isinstance(self._source, str) and self._source.startswith("rtsp"):
                logger.warning("RTSP read failed — attempting reconnect...")
                for attempt in range(self._reconnect_max):
                    time.sleep(self._reconnect_delay)
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self._source)
                    ok, frame = self._cap.read()
                    if ok:
                        logger.info("RTSP reconnect succeeded (attempt %d)", attempt + 1)
                        break
                else:
                    logger.error("RTSP reconnect failed after %d attempts", self._reconnect_max)
                    return None
            else:
                # End of video file or webcam error
                return None

        self._last_frame_time = time.monotonic()
        return frame

    def release(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera source released: %s", self._source)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

class WebcamSource(_OpenCVSource):
    """
    Live webcam source.

    Parameters
    ----------
    device_id : int
        OpenCV device index (0 = default webcam, 1 = second webcam, etc.)
    target_fps : int
        FPS cap (0 = uncapped).
    """

    def __init__(self, device_id: int = 0, target_fps: int = 0):
        super().__init__(source=device_id, target_fps=target_fps)


class FileSource(_OpenCVSource):
    """
    Video file or image sequence source.

    Parameters
    ----------
    path : str or Path
        Path to .mp4, .avi, .mov, etc.
    loop : bool
        Whether to loop the video file endlessly (useful for testing).
    target_fps : int
        FPS cap — use 0 to play as fast as possible (stress-test mode).
    """

    def __init__(self, path, loop: bool = False, target_fps: int = 0):
        self._loop = loop
        self._path = str(path)
        super().__init__(source=self._path, target_fps=target_fps)

    def read(self) -> Optional[np.ndarray]:
        frame = super().read()
        if frame is None and self._loop:
            # Rewind to the beginning
            logger.info("FileSource: looping video %s", self._path)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = super().read()
        return frame


class RTSPSource(_OpenCVSource):
    """
    RTSP camera stream (future drone camera).

    Parameters
    ----------
    url : str
        Full RTSP URL, e.g. "rtsp://192.168.1.10:554/stream"
    target_fps : int
        FPS cap (0 = uncapped — use the stream's native rate).
    """

    def __init__(self, url: str, target_fps: int = 0):
        super().__init__(source=url, target_fps=target_fps, reconnect_attempts=5)


# ---------------------------------------------------------------------------
# Factory function — called by main.py
# ---------------------------------------------------------------------------

def build_source(camera_cfg: dict) -> CameraSource:
    """
    Build the appropriate CameraSource from the `camera` config block.

    Parameters
    ----------
    camera_cfg : dict
        The `camera` section of config.yaml, e.g.:
        {
            "source": 0,
            "target_fps": 30,
            ...
        }

    Returns
    -------
    CameraSource
        Ready-to-use source. Call .read() to get frames, .release() when done.
    """
    raw_source = camera_cfg.get("source", 0)
    fps        = camera_cfg.get("target_fps", 0)

    # Integer → webcam device index
    if isinstance(raw_source, int):
        logger.info("Building WebcamSource (device=%d, fps=%d)", raw_source, fps)
        return WebcamSource(device_id=raw_source, target_fps=fps)

    source_str = str(raw_source)

    # RTSP URL
    if source_str.lower().startswith("rtsp://"):
        logger.info("Building RTSPSource: %s", source_str)
        return RTSPSource(url=source_str, target_fps=fps)

    # File path
    if Path(source_str).exists():
        logger.info("Building FileSource: %s", source_str)
        return FileSource(path=source_str, loop=False, target_fps=fps)

    raise ValueError(
        f"Cannot determine source type for: {raw_source!r}. "
        "Expected: int (webcam index), existing file path, or rtsp:// URL."
    )
