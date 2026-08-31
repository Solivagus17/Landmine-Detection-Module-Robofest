"""
preprocess/frame_prep.py
─────────────────────────
Frame preprocessing pipeline for the Landmine Detection Module.

Responsibilities
----------------
1. Resize native camera frame to working resolution (letterbox — preserves
   aspect ratio, pads with black) so downstream detection always sees a
   consistent pixel grid.
2. Optionally apply CLAHE for contrast enhancement (useful on overcast days
   or shadowed terrain).
3. Optionally apply Gaussian blur to reduce sensor noise before edge detection.
4. Return both the preprocessed frame AND the scale/offset metadata needed to
   map detection coordinates back to the original frame if required.

All parameters are driven by the `preprocessing` and `camera` sections of
config.yaml — no magic numbers here.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Letterbox metadata
# ---------------------------------------------------------------------------

@dataclass
class LetterboxInfo:
    """
    Metadata describing how the original frame was transformed to the working
    resolution. Use these values to map coordinates back to the original.

    Example (map a detection bbox back to native resolution):
        native_x = int((bbox_x - pad_x) / scale)
        native_y = int((bbox_y - pad_y) / scale)
        native_w = int(bbox_w / scale)
        native_h = int(bbox_h / scale)
    """
    orig_w:       int     # Native frame width
    orig_h:       int     # Native frame height
    target_w:     int     # Working-resolution width
    target_h:     int     # Working-resolution height
    scale:        float   # Single scale factor (same for both axes — uniform scaling)
    pad_x:        int     # Left padding in px (black border added by letterbox)
    pad_y:        int     # Top padding in px
    scale_x:      float   # Convenience: target_w / orig_w (without padding)
    scale_y:      float   # Convenience: target_h / orig_h (without padding)


# ---------------------------------------------------------------------------
# FramePreprocessor
# ---------------------------------------------------------------------------

class FramePreprocessor:
    """
    Stateful preprocessor — built once from config, called per frame.

    Parameters
    ----------
    camera_cfg : dict
        The `camera` section of config.yaml.
    preprocessing_cfg : dict
        The `preprocessing` section of config.yaml.
    """

    def __init__(self, camera_cfg: dict, preprocessing_cfg: dict):
        self._target_w = int(camera_cfg.get("working_width",  480))
        self._target_h = int(camera_cfg.get("working_height", 480))

        pp = preprocessing_cfg
        self._use_clahe         = bool(pp.get("use_clahe", True))
        self._clahe_clip        = float(pp.get("clahe_clip_limit", 2.0))
        self._clahe_tile        = int(pp.get("clahe_tile_size", 8))
        self._blur_k            = int(pp.get("blur_kernel_size", 3))
        self._blur_sigma        = float(pp.get("blur_sigma", 0))

        # Validate blur kernel is odd and positive
        if self._blur_k > 0 and self._blur_k % 2 == 0:
            self._blur_k += 1
            logger.warning(
                "blur_kernel_size must be odd — bumped to %d", self._blur_k
            )

        # Pre-build CLAHE object (avoid re-creation per frame)
        if self._use_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self._clahe_clip,
                tileGridSize=(self._clahe_tile, self._clahe_tile),
            )
        else:
            self._clahe = None

        logger.info(
            "FramePreprocessor ready — target: %dx%d, CLAHE: %s, blur_k: %d",
            self._target_w, self._target_h, self._use_clahe, self._blur_k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, LetterboxInfo]:
        """
        Preprocess a raw camera frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from CameraSource.read(), shape (H, W, 3).

        Returns
        -------
        processed : np.ndarray
            BGR preprocessed frame at working resolution, shape (target_h, target_w, 3).
        info : LetterboxInfo
            Coordinate mapping metadata.
        """
        # 1. Letterbox resize
        processed, info = self._letterbox(frame)

        # 2. CLAHE contrast enhancement (applied to L channel in LAB colorspace)
        if self._use_clahe and self._clahe is not None:
            processed = self._apply_clahe(processed)

        # 3. Gaussian blur
        if self._blur_k > 0:
            processed = cv2.GaussianBlur(
                processed,
                (self._blur_k, self._blur_k),
                self._blur_sigma,
            )

        return processed, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _letterbox(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, LetterboxInfo]:
        """
        Resize frame to target resolution while preserving aspect ratio.
        Pads with black (128, 128, 128 grey) to fill the target canvas.

        This is the standard letterbox approach used by YOLO preprocessing —
        consistent with how we'd feed frames to the YOLO upgrade path later.
        """
        orig_h, orig_w = frame.shape[:2]
        target_w, target_h = self._target_w, self._target_h

        # Compute uniform scale to fit within target without cropping
        scale = min(target_w / orig_w, target_h / orig_h)

        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        # Padding to center the resized image
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create grey canvas and place resized frame in center
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        info = LetterboxInfo(
            orig_w  = orig_w,
            orig_h  = orig_h,
            target_w = target_w,
            target_h = target_h,
            scale   = scale,
            pad_x   = pad_x,
            pad_y   = pad_y,
            scale_x = new_w / orig_w,
            scale_y = new_h / orig_h,
        )

        return canvas, info

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the L (luminance) channel of the LAB colorspace.

        This improves local contrast without oversaturating colors, which is
        important for detecting low-contrast mine discs on similar-tone terrain.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        lab  = cv2.merge([l_ch, a_ch, b_ch])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
