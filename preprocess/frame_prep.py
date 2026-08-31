"""
preprocess/frame_prep.py
─────────────────────────
Photometric and Geometric Conditioning Pipeline.

Robofest Gujarat 6.0 | Aerial Robotics | Senior Division

Responsibilities:
  1. Metric Letterboxing: Uniformly scales incoming native camera frames (from Luxonis OAK-D Lite)
     into a fixed square working canvas (e.g., 480x480 or 320x320) with neutral grey padding (114),
     preserving the metric aspect ratio of circular ground targets without non-affine distortion.
  2. Luminance-Isolated CLAHE: Transforms BGR to CIE L*a*b* color space and equalizes only the
     lightness L* channel (clipLimit=2.0, tileGridSize=8x8), enhancing contrast on shadowed or
     overcast terrain without altering chromatic ratios.
  3. Gaussian Noise Filtering: Attenuates high-frequency sensor noise and Bayer artifacts prior to
     spatial gradient extraction.
  4. Spatial Inversion Metadata: Generates and returns a `LetterboxInfo` dataclass instance containing
     exact scale factors and offsets for re-projecting bounding coordinates back to native space.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Coordinate Transformation Metadata Dataclass
# ===========================================================================

@dataclass(frozen=True)
class LetterboxInfo:
    """
    Encapsulates affine coordinate mapping parameters between native and working resolutions.

    Fields
    ------
    orig_w : int
        Native sensor frame width in pixels (e.g. 1920 or 1280).
    orig_h : int
        Native sensor frame height in pixels (e.g. 1080 or 720).
    target_w : int
        Working canvas width in pixels (e.g. 480 or 320).
    target_h : int
        Working canvas height in pixels (e.g. 480 or 320).
    scale : float
        Uniform isotropic scaling factor: min(target_w / orig_w, target_h / orig_h).
    pad_x : int
        Horizontal padding offset in pixels applied to each side.
    pad_y : int
        Vertical padding offset in pixels applied to each side.
    scale_x : float
        Effective horizontal scale factor: (orig_w * scale) / orig_w.
    scale_y : float
        Effective vertical scale factor: (orig_h * scale) / orig_h.

    Coordinate Inversion Formula:
        native_x = (working_x - pad_x) / scale
        native_y = (working_y - pad_y) / scale
    """
    orig_w:   int
    orig_h:   int
    target_w: int
    target_h: int
    scale:    float
    pad_x:    int
    pad_y:    int
    scale_x:  float
    scale_y:  float


# ===========================================================================
# Frame Preprocessor Engine
# ===========================================================================

class FramePreprocessor:
    """
    Stateful image preprocessor managing spatial rescaling and photometric equalization.

    Parameters
    ----------
    camera_cfg : dict
        The `camera` section from config.yaml containing working resolution targets.
    preprocessing_cfg : dict
        The `preprocessing` section from config.yaml controlling CLAHE and Gaussian blur.
    """

    def __init__(self, camera_cfg: dict, preprocessing_cfg: dict) -> None:
        self._target_w: int = int(camera_cfg.get("working_width", 480))
        self._target_h: int = int(camera_cfg.get("working_height", 480))

        pp = preprocessing_cfg
        self._use_clahe: bool = bool(pp.get("use_clahe", True))
        self._clahe_clip: float = float(pp.get("clahe_clip_limit", 2.0))
        self._clahe_tile: int = int(pp.get("clahe_tile_size", 8))
        self._blur_k: int = int(pp.get("blur_kernel_size", 3))
        self._blur_sigma: float = float(pp.get("blur_sigma", 0))

        # Enforce odd aperture size for Gaussian convolution
        if self._blur_k > 0 and self._blur_k % 2 == 0:
            self._blur_k += 1
            logger.warning("blur_kernel_size must be odd — adjusted to %d", self._blur_k)

        # Pre-allocate CLAHE operator descriptor
        self._clahe: Optional[cv2.CLAHE] = None
        if self._use_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self._clahe_clip,
                tileGridSize=(self._clahe_tile, self._clahe_tile),
            )

        logger.info(
            "FramePreprocessor initialized (canvas=%dx%d, CLAHE=%s [clip=%.1f, grid=%d], blur_k=%d)",
            self._target_w, self._target_h, self._use_clahe, self._clahe_clip, self._clahe_tile, self._blur_k,
        )

    # -----------------------------------------------------------------------
    # Public Execution Method
    # -----------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, LetterboxInfo]:
        """
        Execute the full geometric and photometric preprocessing sequence.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR input frame from CameraSource.read() of shape (H, W, 3).

        Returns
        -------
        processed : np.ndarray
            Conditioned BGR image of shape (target_h, target_w, 3) ready for candidate proposal.
        info : LetterboxInfo
            Coordinate mapping metadata for native frame projection.
        """
        # Step 1: Metric aspect-ratio preserving letterbox resize
        processed, info = self._letterbox(frame)

        # Step 2: CIE L*a*b* luminance-channel CLAHE equalization
        if self._use_clahe and self._clahe is not None:
            processed = self._apply_clahe(processed)

        # Step 3: High-frequency Gaussian spatial smoothing
        if self._blur_k > 0:
            processed = cv2.GaussianBlur(
                processed,
                (self._blur_k, self._blur_k),
                self._blur_sigma,
            )

        return processed, info

    # -----------------------------------------------------------------------
    # Internal Transformation Helpers
    # -----------------------------------------------------------------------

    def _letterbox(self, frame: np.ndarray) -> Tuple[np.ndarray, LetterboxInfo]:
        """
        Resize image to square target dimensions while preserving isotropic aspect ratio.

        Pads margins with neutral grey (114, 114, 114) consistent with neural model standards.
        """
        orig_h, orig_w = frame.shape[:2]
        target_w, target_h = self._target_w, self._target_h

        # Compute uniform scale to fit within canvas bounds
        scale: float = min(target_w / orig_w, target_h / orig_h)
        new_w: int = int(round(orig_w * scale))
        new_h: int = int(round(orig_h * scale))

        # Centering offsets
        pad_x: int = (target_w - new_w) // 2
        pad_y: int = (target_h - new_h) // 2

        # Bilinear interpolation resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Initialize grey canvas and embed resized frame
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        info = LetterboxInfo(
            orig_w   = orig_w,
            orig_h   = orig_h,
            target_w = target_w,
            target_h = target_h,
            scale    = scale,
            pad_x    = pad_x,
            pad_y    = pad_y,
            scale_x  = new_w / orig_w,
            scale_y  = new_h / orig_h,
        )

        return canvas, info

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization exclusively to luminance.

        Decouples chromaticity from intensity, preventing chromatic artifacting.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
