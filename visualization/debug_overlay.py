"""
visualization/debug_overlay.py
────────────────────────────────
Diagnostic Visualization HUD and Real-Time Feedback Overlay Engine.

Robofest Gujarat 6.0 | Aerial Robotics | Senior Division

Renders live diagnostic overlays on conditioned video frames for interactive bench tuning:
  - Semi-transparent filled contours (alpha=0.25) and bounding boxes per detection.
  - Semantic class labels and calibrated confidence pills.
  - Rolling 30-frame average FPS counter (top-left).
  - Active detection backend provenance indicator (top-right).
  - Real-time detection count and per-frame compute latency summary (bottom-left).

Display Auto-Scaling:
  Detection algorithms run strictly at `working_resolution` (e.g., 480x480 or 320x320) to
  minimize latency on Raspberry Pi Zero 2 W. The debug overlay dynamically upscales the
  output canvas to fill the host workstation monitor without introducing distortion.

Deployment Note:
  In production flight on the drone, disable this module (`debug.enabled: false` or `--no-debug`)
  to eliminate X11/GUI thread overhead and conserve 512 MB RAM.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
import platform
import time
from collections import deque
from typing import Deque, List, Optional, Tuple, Union

import cv2
import numpy as np

from output.detection_result import Detection, FrameResult, MineClass

logger = logging.getLogger(__name__)


# ===========================================================================
# Display Dimension Resolution Helper
# ===========================================================================

def _get_screen_size() -> Tuple[int, int]:
    """
    Detect primary display resolution in pixels (width, height).

    Employs Windows user32 DPI-aware probe or cross-platform OpenCV probe,
    falling back to 1280x720 if windowing context is unavailable.

    Returns
    -------
    Tuple[int, int]
        Physical display dimensions (width, height).
    """
    try:
        if platform.system() == "Windows":
            import ctypes
            user32 = ctypes.windll.user32
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                user32.SetProcessDPIAware()
            w = user32.GetSystemMetrics(0)
            h = user32.GetSystemMetrics(1)
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass

    try:
        tmp_win = "__screen_probe__"
        cv2.namedWindow(tmp_win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(tmp_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(1)
        rect = cv2.getWindowImageRect(tmp_win)
        cv2.destroyWindow(tmp_win)
        if rect[2] > 0 and rect[3] > 0:
            return rect[2], rect[3]
    except Exception:
        pass

    logger.warning("Unable to detect display resolution — defaulting to 1280x720")
    return 1280, 720


# ===========================================================================
# Diagnostic Overlay Engine
# ===========================================================================

class DebugOverlay:
    """
    Renders diagnostic HUD elements, bounding geometries, and telemetry overlays.

    Parameters
    ----------
    debug_cfg : dict
        The `debug` section from config.yaml.
    """

    def __init__(self, debug_cfg: dict) -> None:
        self._enabled: bool = bool(debug_cfg.get("enabled", True))
        self._draw_fill: bool = bool(debug_cfg.get("draw_contour_fill", True))
        self._fill_alpha: float = float(debug_cfg.get("contour_fill_alpha", 0.25))
        self._draw_bbox: bool = bool(debug_cfg.get("draw_bbox", True))
        self._show_fps: bool = bool(debug_cfg.get("show_fps", True))
        self._show_backend: bool = bool(debug_cfg.get("show_backend_label", True))
        self._window_title: str = str(debug_cfg.get("window_title", "Landmine Detection - Debug View"))

        def _to_bgr(lst: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
            if isinstance(lst, (list, tuple)) and len(lst) == 3:
                return (int(lst[0]), int(lst[1]), int(lst[2]))
            return default

        self._color_mine: Tuple[int, int, int]   = _to_bgr(debug_cfg.get("color_surface_mine"),  (50,  220,  50))
        self._color_marker: Tuple[int, int, int] = _to_bgr(debug_cfg.get("color_buried_marker"), (220, 120,  50))
        self._color_unk: Tuple[int, int, int]    = _to_bgr(debug_cfg.get("color_unknown"),       (200, 200, 200))

        # Rolling 30-frame timing buffer for FPS estimation
        self._frame_times: Deque[float] = deque(maxlen=30)

        # Typography configuration
        self._font: int = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale: float = 0.55
        self._font_thick: int = 1

        # Display window scaling setup
        screen_w, screen_h = _get_screen_size()
        frac: float = float(debug_cfg.get("display_screen_fraction", 0.95))
        self._display_w: int = int(screen_w * frac)
        self._display_h: int = int(screen_h * frac)
        self._window_created: bool = False

        logger.info(
            "DebugOverlay initialized (window=%dx%d, contour_fill=%s, fps_hud=%s)",
            self._display_w, self._display_h, self._draw_fill, self._show_fps,
        )

    # -----------------------------------------------------------------------
    # Public Rendering & Display Methods
    # -----------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        result: FrameResult,
        backend_name: str = "classical",
    ) -> np.ndarray:
        """
        Draw all diagnostic annotations and telemetry metrics onto a frame copy.

        Parameters
        ----------
        frame : np.ndarray
            BGR image at working resolution.
        result : FrameResult
            Perception records for the current frame.
        backend_name : str, default="classical"
            Active detector backend tag.

        Returns
        -------
        np.ndarray
            Annotated BGR image.
        """
        if not self._enabled:
            return frame

        out: np.ndarray = frame.copy()
        self._frame_times.append(time.monotonic())

        # Render individual detection contours and bounding boxes
        for det in result.detections:
            self._draw_detection(out, det)

        # Render telemetry HUD elements
        if self._show_fps:
            self._draw_fps(out)

        if self._show_backend:
            self._draw_backend_label(out, backend_name)

        self._draw_detection_summary(out, result)

        return out

    def show(self, frame: np.ndarray) -> bool:
        """
        Display annotated frame in an OpenCV window scaled to monitor geometry.

        Returns
        -------
        bool
            False if 'q' or Escape key was pressed, True otherwise.
        """
        if not self._enabled:
            return True

        if not self._window_created:
            cv2.namedWindow(self._window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_title, self._display_w, self._display_h)
            cv2.moveWindow(self._window_title, 0, 0)
            self._window_created = True

        # Upscale working canvas to fit display window without aspect distortion
        fh, fw = frame.shape[:2]
        scale: float = min(self._display_w / fw, self._display_h / fh)
        scaled_w: int = int(fw * scale)
        scaled_h: int = int(fh * scale)
        display: np.ndarray = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # Pad canvas with dark grey borders
        canvas: np.ndarray = np.full((self._display_h, self._display_w, 3), 30, dtype=np.uint8)
        pad_x: int = (self._display_w - scaled_w) // 2
        pad_y: int = (self._display_h - scaled_h) // 2
        canvas[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = display

        cv2.imshow(self._window_title, canvas)
        key: int = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        """Close all OpenCV display windows."""
        if self._enabled:
            cv2.destroyAllWindows()

    # -----------------------------------------------------------------------
    # Internal Annotation Helpers
    # -----------------------------------------------------------------------

    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Resolve BGR color for semantic class."""
        if class_name == MineClass.SURFACE_MINE:
            return self._color_mine
        if class_name == MineClass.BURIED_MARKER:
            return self._color_marker
        return self._color_unk

    def _draw_detection(self, out: np.ndarray, det: Detection) -> None:
        """Draw alpha-blended contour fill, outer stroke, and label pill."""
        color = self._get_color(det.class_name)
        x, y, w, h = det.bbox

        # Alpha-blended filled contour
        if self._draw_fill and det.contour is not None:
            overlay = out.copy()
            cv2.drawContours(overlay, [det.contour], -1, color, thickness=cv2.FILLED)
            cv2.addWeighted(overlay, self._fill_alpha, out, 1 - self._fill_alpha, 0, out)
            cv2.drawContours(out, [det.contour], -1, color, thickness=2)

        # Bounding rectangle
        if self._draw_bbox:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness=2)

        # Semantic label and confidence score
        label: str = f"{det.class_name}  {det.confidence:.2f}"
        text_size = cv2.getTextSize(label, self._font, self._font_scale, self._font_thick)[0]
        text_x: int = x
        text_y: int = max(y - 6, text_size[1] + 4)

        # Contrast pill background
        bg_x1: int = text_x - 2
        bg_y1: int = text_y - text_size[1] - 4
        bg_x2: int = text_x + text_size[0] + 4
        bg_y2: int = text_y + 2
        cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), thickness=-1)

        cv2.putText(
            out, label,
            (text_x, text_y),
            self._font, self._font_scale,
            color, self._font_thick, cv2.LINE_AA,
        )

    def _draw_fps(self, out: np.ndarray) -> None:
        """Render rolling average FPS counter."""
        fps: float = self._compute_fps()
        label: str = f"FPS: {fps:.1f}"
        self._put_hud_text(out, label, row=0, col="left")

    def _draw_backend_label(self, out: np.ndarray, backend_name: str) -> None:
        """Render active detector backend indicator."""
        label: str = f"BACKEND: {backend_name.upper()}"
        self._put_hud_text(out, label, row=0, col="right")

    def _draw_detection_summary(self, out: np.ndarray, result: FrameResult) -> None:
        """Render detection counts and per-frame compute latency."""
        n_mines: int = len(result.surface_mines)
        n_markers: int = len(result.buried_markers)
        label: str = f"Mines: {n_mines}  |  Markers: {n_markers}  |  [{result.processing_time_ms:.1f} ms]"
        h = out.shape[0]
        cv2.putText(
            out, label,
            (8, h - 10),
            self._font, 0.48,
            (220, 220, 220), 1, cv2.LINE_AA,
        )

    def _put_hud_text(
        self,
        out: np.ndarray,
        text: str,
        row: int = 0,
        col: str = "left",
    ) -> None:
        """Place text with drop shadow at a structured HUD grid position."""
        line_h: int = 22
        margin: int = 8
        text_y: int = margin + line_h * (row + 1)
        text_size = cv2.getTextSize(text, self._font, 0.50, 1)[0]

        if col == "right":
            text_x = out.shape[1] - text_size[0] - margin
        else:
            text_x = margin

        # Shadow stroke
        cv2.putText(out, text, (text_x + 1, text_y + 1), self._font, 0.50, (0, 0, 0), 2, cv2.LINE_AA)
        # Foreground text
        cv2.putText(out, text, (text_x, text_y), self._font, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

    def _compute_fps(self) -> float:
        """Compute rolling average FPS across recent frame timestamps."""
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._frame_times) - 1) / elapsed
