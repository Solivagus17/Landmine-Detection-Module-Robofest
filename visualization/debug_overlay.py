"""
visualization/debug_overlay.py
────────────────────────────────
Debug visualization overlay for the Landmine Detection Module.

Renders detection results on the live frame for real-time tuning on a laptop.
Controlled by the `debug` section of config.yaml — set `enabled: false`
for headless/deployment mode (e.g., on Pi 5 during the actual flight).

Overlay contents:
  - Semi-transparent filled contour OR bounding box per detection
  - Class label + confidence text
  - FPS counter (top-left)
  - Active detector backend name (top-right)
  - Detection count per class (bottom-left)

Color coding (configurable in config.yaml):
  - surface_mine   → Green by default
  - buried_marker  → Teal/Blue by default
  - unknown        → Grey

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import time
import logging
import platform
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from output.detection_result import Detection, FrameResult, MineClass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Screen resolution helper
# ---------------------------------------------------------------------------

def _get_screen_size() -> Tuple[int, int]:
    """
    Return (width, height) of the primary display.
    Uses ctypes on Windows (no extra dependency).
    Falls back to a sensible default on other platforms.
    """
    try:
        if platform.system() == "Windows":
            import ctypes
            user32 = ctypes.windll.user32
            # SetProcessDPIAware so we get physical pixels, not scaled ones
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
    # Cross-platform fallback via OpenCV (creates a tiny temp window)
    try:
        tmp = "__screen_probe__"
        cv2.namedWindow(tmp, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(tmp, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Give it one event loop tick
        cv2.waitKey(1)
        rect = cv2.getWindowImageRect(tmp)
        cv2.destroyWindow(tmp)
        if rect[2] > 0 and rect[3] > 0:
            return rect[2], rect[3]
    except Exception:
        pass
    logger.warning("Could not detect screen size — defaulting to 1280x720")
    return 1280, 720


# ---------------------------------------------------------------------------
# DebugOverlay
# ---------------------------------------------------------------------------

class DebugOverlay:
    """
    Draws detection results on a frame for interactive tuning.

    Parameters
    ----------
    debug_cfg : dict
        The `debug` section of config.yaml.
    """

    def __init__(self, debug_cfg: dict):
        self._enabled      = bool(debug_cfg.get("enabled", True))
        self._draw_fill    = bool(debug_cfg.get("draw_contour_fill", True))
        self._fill_alpha   = float(debug_cfg.get("contour_fill_alpha", 0.25))
        self._draw_bbox    = bool(debug_cfg.get("draw_bbox", True))
        self._show_fps     = bool(debug_cfg.get("show_fps", True))
        self._show_backend = bool(debug_cfg.get("show_backend_label", True))
        self._window_title = str(debug_cfg.get("window_title", "Landmine Detection"))

        def _to_bgr(lst, default) -> Tuple[int, int, int]:
            if isinstance(lst, (list, tuple)) and len(lst) == 3:
                return (int(lst[0]), int(lst[1]), int(lst[2]))
            return default

        self._color_mine   = _to_bgr(debug_cfg.get("color_surface_mine"),  (50,  220,  50))
        self._color_marker = _to_bgr(debug_cfg.get("color_buried_marker"), (220, 120,  50))
        self._color_unk    = _to_bgr(debug_cfg.get("color_unknown"),       (200, 200, 200))

        # Rolling FPS buffer (last 30 frame times)
        self._frame_times: deque = deque(maxlen=30)

        # OpenCV font settings
        self._font       = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.55
        self._font_thick = 1

        # --- Display / window sizing ---
        # Detect screen resolution and compute the display frame size.
        # Detection always runs at working_resolution (e.g. 480x480);
        # we upscale ONLY for display so the window fills the screen.
        screen_w, screen_h = _get_screen_size()
        # Use a configurable fraction of the screen (default 95%)
        frac = float(debug_cfg.get("display_screen_fraction", 0.95))
        self._display_w = int(screen_w * frac)
        self._display_h = int(screen_h * frac)
        self._screen_w  = screen_w
        self._screen_h  = screen_h
        self._window_created = False   # lazy window creation on first show()

        logger.info(
            "DebugOverlay ready — screen: %dx%d, display window: %dx%d, "
            "fill: %s, fps: %s",
            screen_w, screen_h, self._display_w, self._display_h,
            self._draw_fill, self._show_fps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        result: FrameResult,
        backend_name: str = "classical",
    ) -> np.ndarray:
        """
        Draw debug overlay on a frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame at working resolution (will be drawn on in-place copy).
        result : FrameResult
            Detection results for this frame.
        backend_name : str
            Detector backend label for overlay display.

        Returns
        -------
        np.ndarray
            Annotated BGR frame (new array — original is not modified).
        """
        if not self._enabled:
            return frame

        # Work on a copy — do not mutate the original frame
        out = frame.copy()

        # Track frame times for FPS calculation
        self._frame_times.append(time.monotonic())

        # Draw each detection
        for det in result.detections:
            self._draw_detection(out, det)

        # Overlay text elements
        if self._show_fps:
            self._draw_fps(out)

        if self._show_backend:
            self._draw_backend_label(out, backend_name)

        self._draw_detection_summary(out, result)

        return out

    def show(self, frame: np.ndarray) -> bool:
        """
        Display the frame in an OpenCV window.

        Returns
        -------
        bool
            False if the user pressed 'q' or Escape (quit signal), True otherwise.
        """
        if not self._enabled:
            return True

        # --- Create window once, sized to screen ---
        if not self._window_created:
            cv2.namedWindow(self._window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_title, self._display_w, self._display_h)
            # Move to top-left corner so it doesn't hide behind the taskbar
            cv2.moveWindow(self._window_title, 0, 0)
            self._window_created = True

        # --- Upscale display frame to fill the window ---
        # The input frame is at working resolution (e.g. 480x480).
        # We letterbox it again into display_w x display_h so aspect ratio
        # is preserved but the window is fully sized to the screen.
        fh, fw = frame.shape[:2]
        scale    = min(self._display_w / fw, self._display_h / fh)
        scaled_w = int(fw * scale)
        scaled_h = int(fh * scale)
        display  = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # Pad to exact display_w x display_h with a dark grey border
        canvas = np.full((self._display_h, self._display_w, 3), 30, dtype=np.uint8)
        pad_x  = (self._display_w - scaled_w) // 2
        pad_y  = (self._display_h - scaled_h) // 2
        canvas[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = display

        cv2.imshow(self._window_title, canvas)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)   # 27 = Escape

    def close(self) -> None:
        """Destroy all OpenCV windows."""
        if self._enabled:
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        if class_name == MineClass.SURFACE_MINE:
            return self._color_mine
        if class_name == MineClass.BURIED_MARKER:
            return self._color_marker
        return self._color_unk

    def _draw_detection(self, out: np.ndarray, det: Detection) -> None:
        """Draw contour fill + bounding box + label for one detection."""
        color = self._get_color(det.class_name)
        x, y, w, h = det.bbox

        # --- Semi-transparent contour fill ---
        if self._draw_fill and det.contour is not None:
            overlay = out.copy()
            cv2.drawContours(overlay, [det.contour], -1, color, thickness=cv2.FILLED)
            cv2.addWeighted(overlay, self._fill_alpha, out, 1 - self._fill_alpha, 0, out)
            # Solid contour border
            cv2.drawContours(out, [det.contour], -1, color, thickness=2)

        # --- Bounding box ---
        if self._draw_bbox:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness=2)

        # --- Label: "class  0.73" ---
        label      = f"{det.class_name}  {det.confidence:.2f}"
        text_size  = cv2.getTextSize(label, self._font, self._font_scale, self._font_thick)[0]
        text_x     = x
        text_y     = max(y - 6, text_size[1] + 4)

        # Dark background pill for readability
        bg_x1 = text_x - 2
        bg_y1 = text_y - text_size[1] - 4
        bg_x2 = text_x + text_size[0] + 4
        bg_y2 = text_y + 2
        cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), thickness=-1)

        cv2.putText(
            out, label,
            (text_x, text_y),
            self._font, self._font_scale,
            color, self._font_thick, cv2.LINE_AA,
        )

    def _draw_fps(self, out: np.ndarray) -> None:
        """Draw FPS counter top-left."""
        fps = self._compute_fps()
        label = f"FPS: {fps:.1f}"
        self._put_hud_text(out, label, row=0, col="left")

    def _draw_backend_label(self, out: np.ndarray, backend_name: str) -> None:
        """Draw backend name top-right."""
        label = f"MODE: {backend_name.upper()}"
        self._put_hud_text(out, label, row=0, col="right")

    def _draw_detection_summary(self, out: np.ndarray, result: FrameResult) -> None:
        """Draw detection counts bottom-left."""
        n_mines   = len(result.surface_mines)
        n_markers = len(result.buried_markers)
        label = (
            f"Mines: {n_mines}  |  Markers: {n_markers}  |  "
            f"[{result.processing_time_ms:.1f} ms]"
        )
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
        """
        Place a HUD text element at a grid position.
        row=0 → near top, col="left"|"right"
        """
        line_h  = 22
        margin  = 8
        text_y  = margin + line_h * (row + 1)

        text_size = cv2.getTextSize(text, self._font, 0.50, 1)[0]

        if col == "right":
            text_x = out.shape[1] - text_size[0] - margin
        else:
            text_x = margin

        # Shadow
        cv2.putText(out, text, (text_x + 1, text_y + 1),
                    self._font, 0.50, (0, 0, 0), 2, cv2.LINE_AA)
        # Foreground
        cv2.putText(out, text, (text_x, text_y),
                    self._font, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

    def _compute_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / elapsed
