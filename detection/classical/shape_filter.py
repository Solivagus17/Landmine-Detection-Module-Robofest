"""
detection/classical/shape_filter.py
────────────────────────────────────
Shape-based filtering and classification of raw candidate contours.

Given the raw contour list from CandidateProposer, this module applies a
layered rejection pipeline specifically tuned to discriminate mine discs from
organic/scene false positives (hands, skin, fabric, shadows, floor patterns).

Filter pipeline (in order of cheapness, applied early-to-late):
  1.  Area gate               — size range for mine vs scene objects
  2.  Aspect ratio gate       — disc is not very elongated
  3.  Circularity gate        — roundness: disc ≈ 0.7-1.0, jagged blob ≈ 0.05-0.3
  4.  Solidity gate           — area / convex_hull (hand gaps → low solidity)
  5.  Convexity defect gate   — max depth of hull concavities in px
                                (finger gaps create deep defects; disc has none)
  6.  Texture uniformity gate — std dev of grayscale pixels inside contour
                                (flat plastic = low; skin/fabric = high)
  7.  Color consistency gate  — std dev of pixel values: mine is one solid color
  8.  Confidence scoring      — heuristic blend of shape quality metrics

Key design choice: compute ALL metrics in the filter() loop from the grayscale
frame, then pass them into evaluators. Evaluators are pure logic, not I/O.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from output.detection_result import Detection, MineClass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intermediate candidate (internal, before confidence scoring)
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    contour:       np.ndarray
    bbox:          Tuple[int, int, int, int]   # (x, y, w, h)
    area:          float
    solidity:      float
    aspect_ratio:  float
    class_name:    str
    confidence:    float


# ---------------------------------------------------------------------------
# ShapeFilter
# ---------------------------------------------------------------------------

class ShapeFilter:
    """
    Filters and classifies candidate contours from CandidateProposer.

    Parameters
    ----------
    mine_cfg : dict
        The `surface_mine` section of config.yaml.
    marker_cfg : dict
        The `buried_marker` section of config.yaml.
    """

    def __init__(self, mine_cfg: dict, marker_cfg: dict):
        # --- Surface mine parameters ---
        self._mine_min_area   = float(mine_cfg.get("min_area_px",         150))
        self._mine_max_area   = float(mine_cfg.get("max_area_px",       20000))
        self._mine_min_ar     = float(mine_cfg.get("min_aspect_ratio",    0.40))
        self._mine_max_ar     = float(mine_cfg.get("max_aspect_ratio",    2.50))
        self._mine_min_sol    = float(mine_cfg.get("min_solidity",         0.65))
        self._mine_min_ext    = float(mine_cfg.get("min_extent",           0.0))

        # Circularity = 4π·area/perimeter²  (perfect circle = 1.0)
        # Ellipse 2:1 ≈ 0.78 | jagged hand blob ≈ 0.05-0.25
        self._mine_min_circ   = float(mine_cfg.get("min_circularity",     0.50))

        # Polygon approximation vertex count gate.
        # cv2.approxPolyDP simplifies the contour with epsilon=2% of perimeter.
        # Rectangle / painting / frame  → ~4 vertices   (REJECTED)
        # Triangle, diamond             → 3-5 vertices  (REJECTED)
        # Ellipse / disc                → 8-15 vertices (ACCEPTED)
        # Raise epsilon_frac in config to get fewer vertices from noisy contours.
        self._mine_min_verts  = int(mine_cfg.get("min_approx_vertices",   7))
        self._mine_eps_frac   = float(mine_cfg.get("approx_epsilon_frac", 0.02))

        # Convexity defect gate:
        # Depth of deepest concavity in the contour's convex hull (pixels).
        # A disc has near-zero defect depth; finger gaps create 10-40px defects.
        # Unit: pixels at working resolution (e.g. 480px wide).
        self._mine_max_defect = float(mine_cfg.get("max_convexity_defect_px", 10.0))

        # Texture uniformity gate:
        # Std dev of grayscale pixel values INSIDE the contour mask.
        # Flat plastic disc: 5-25 (very uniform tone).
        # Human skin / fabric: 30-70 (texture, creases, shadows).
        self._mine_max_tex    = float(mine_cfg.get("max_texture_std",     32.0))

        # Color consistency gate:
        # Std dev of BGR channel values inside the contour mask, averaged
        # across channels. One-solid-color mine: low; multi-tone skin: high.
        # Set to 0 to disable (same info as texture_std for greyscale).
        self._mine_max_color  = float(mine_cfg.get("max_color_std",       40.0))

        self._mine_base_conf  = float(mine_cfg.get("base_confidence",      0.50))

        # --- Buried marker parameters ---
        self._mrk_enabled     = bool(marker_cfg.get("enabled", True))
        self._mrk_min_area    = float(marker_cfg.get("min_area_px",         30))
        self._mrk_max_area    = float(marker_cfg.get("max_area_px",       3000))
        self._mrk_min_ar      = float(marker_cfg.get("min_aspect_ratio",   0.15))
        self._mrk_max_ar      = float(marker_cfg.get("max_aspect_ratio",   6.0))
        self._mrk_min_sol     = float(marker_cfg.get("min_solidity",        0.30))
        self._mrk_min_circ    = float(marker_cfg.get("min_circularity",    0.25))
        self._mrk_base_conf   = float(marker_cfg.get("base_confidence",    0.40))

        # --- Marker color gate ---
        self._color_gate_on   = bool(marker_cfg.get("use_color_gate", False))
        self._color_low1  = np.array(marker_cfg.get("marker_color_hsv_low",  [0,   100, 80]),  dtype=np.uint8)
        self._color_high1 = np.array(marker_cfg.get("marker_color_hsv_high", [10,  255, 255]), dtype=np.uint8)
        self._use_hue_wrap    = bool(marker_cfg.get("use_hue_wrap", False))
        self._color_low2  = np.array(marker_cfg.get("marker_color_hsv_low2",  [170, 100, 80]),  dtype=np.uint8)
        self._color_high2 = np.array(marker_cfg.get("marker_color_hsv_high2", [179, 255, 255]), dtype=np.uint8)
        self._color_min_fill  = float(marker_cfg.get("color_gate_min_fill", 0.30))

        logger.info(
            "ShapeFilter ready — mine area: [%.0f, %.0f] px², circ≥%.2f, "
            "tex_std≤%.0f, defect≤%.0fpx | marker: %s",
            self._mine_min_area, self._mine_max_area,
            self._mine_min_circ, self._mine_max_tex, self._mine_max_defect,
            "ENABLED" if self._mrk_enabled else "DISABLED",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        contours: list,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
        detector_backend: str = "classical",
    ) -> List[Detection]:
        """
        Filter raw candidate contours and produce Detection objects.

        Parameters
        ----------
        contours : list of np.ndarray
            Raw contours from CandidateProposer.propose().
        frame : np.ndarray
            Preprocessed BGR frame — used for texture/color analysis.
        frame_id : int
            Current frame counter (passed through to Detection).
        timestamp_ms : float
            Wall-clock time of frame capture in ms.
        detector_backend : str
            Backend label passed through to Detection.

        Returns
        -------
        List[Detection]
            Confirmed detections, sorted by confidence descending.
        """
        # Pre-compute grayscale once — reused for texture analysis per contour
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pre-compute HSV frame once (only needed if color gate is on)
        hsv_frame: Optional[np.ndarray] = None
        if self._color_gate_on:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        fh, fw = frame.shape[:2]
        detections: List[Detection] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1:
                continue   # degenerate contour

            # ----------------------------------------------------------
            # Step 1 — area gate
            # ----------------------------------------------------------
            is_mine_size   = self._mine_min_area <= area <= self._mine_max_area
            is_marker_size = self._mrk_min_area  <= area <= self._mrk_max_area

            if not is_mine_size and not is_marker_size:
                continue

            # ----------------------------------------------------------
            # Step 2 — cheap geometric metrics (no pixel access yet)
            # ----------------------------------------------------------
            bbox         = cv2.boundingRect(contour)   # (x, y, w, h)
            bx, by, bw, bh = bbox
            aspect_ratio = bw / bh if bh > 0 else 0.0

            hull      = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity  = area / hull_area if hull_area > 0 else 0.0

            extent = area / (bw * bh) if (bw * bh) > 0 else 0.0

            # Circularity = 4π·area/perimeter²
            perimeter   = cv2.arcLength(contour, closed=True)
            circularity = (4.0 * 3.14159265 * area / (perimeter * perimeter)
                          ) if perimeter > 0 else 0.0

            # Polygon vertex count — RECTANGLE REJECTION
            # approxPolyDP with epsilon=2% of perimeter.
            # Rectangle → 4 vertices | Circle/ellipse → 8-15 vertices.
            epsilon = self._mine_eps_frac * perimeter
            approx  = cv2.approxPolyDP(contour, epsilon, closed=True)
            n_verts = len(approx)

            # ----------------------------------------------------------
            # Step 3 — pixel-level metrics (only if cheap gates passed)
            # ----------------------------------------------------------
            # Guard: skip pixel analysis early if geometric gates already fail
            # (avoids wasting time on doomed candidates)
            geom_ok_mine = (
                is_mine_size
                and n_verts >= self._mine_min_verts          # not a rectangle/polygon
                and circularity >= self._mine_min_circ
                and self._mine_min_ar <= aspect_ratio <= self._mine_max_ar
                and solidity >= self._mine_min_sol
            )

            texture_std    = 999.0   # high default → rejected unless computed
            max_defect_px  = 999.0
            color_std      = 999.0

            if geom_ok_mine:
                texture_std, color_std = self._compute_pixel_stats(
                    contour, bx, by, bw, bh, fh, fw, gray, frame
                )
                max_defect_px = self._compute_max_defect(contour)

            # ----------------------------------------------------------
            # Step 4 — classify as mine or marker
            # ----------------------------------------------------------
            candidate = None

            if is_mine_size:
                candidate = self._evaluate_mine(
                    contour, bbox, area, solidity, aspect_ratio,
                    extent, circularity, n_verts, texture_std, color_std, max_defect_px,
                )

            if candidate is None and is_marker_size and self._mrk_enabled:
                candidate = self._evaluate_marker(
                    contour, bbox, area, solidity, aspect_ratio,
                    circularity, hsv_frame,
                )

            if candidate is None:
                continue

            # ----------------------------------------------------------
            # Step 5 — produce Detection
            # ----------------------------------------------------------
            detections.append(Detection(
                class_name       = candidate.class_name,
                bbox             = candidate.bbox,
                contour          = candidate.contour,
                confidence       = candidate.confidence,
                frame_id         = frame_id,
                timestamp_ms     = timestamp_ms,
                detector_backend = detector_backend,
            ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.debug(
            "ShapeFilter: %d contours → %d detections (%d mines, %d markers)",
            len(contours),
            len(detections),
            sum(1 for d in detections if d.class_name == MineClass.SURFACE_MINE),
            sum(1 for d in detections if d.class_name == MineClass.BURIED_MARKER),
        )

        return detections

    # ------------------------------------------------------------------
    # Internal classifiers
    # ------------------------------------------------------------------

    def _evaluate_mine(
        self,
        contour, bbox, area: float,
        solidity: float, aspect_ratio: float, extent: float,
        circularity: float,
        n_verts: int,
        texture_std: float,
        color_std: float,
        max_defect_px: float,
    ) -> Optional[_Candidate]:
        """
        Apply all surface_mine filters.

        Filter order: cheapest (geometry) first, most expensive (pixel) last.
        """

        # 1. VERTEX COUNT — reject rectangles, frames, paintings, screens
        # A circle/ellipse needs 8-15 poly segments; a rectangle needs only 4.
        if n_verts < self._mine_min_verts:
            return None

        # 2. CIRCULARITY — disc roundness
        if circularity < self._mine_min_circ:
            return None

        # 3. ASPECT RATIO — disc is not very elongated
        if not (self._mine_min_ar <= aspect_ratio <= self._mine_max_ar):
            return None

        # 4. SOLIDITY — hand gaps / concave shapes rejected
        if solidity < self._mine_min_sol:
            return None

        # 5. CONVEXITY DEFECTS — finger / finger-gap depth
        if max_defect_px > self._mine_max_defect:
            return None

        # 6. TEXTURE UNIFORMITY
        if texture_std > self._mine_max_tex:
            return None

        # 7. COLOR CONSISTENCY
        if self._mine_max_color > 0 and color_std > self._mine_max_color:
            return None

        # Optional extent gate
        if self._mine_min_ext > 0 and extent < self._mine_min_ext:
            return None

        # Heuristic confidence
        circ_bonus = max(0.0, (circularity - self._mine_min_circ) * 0.20)
        sol_bonus  = max(0.0, (solidity    - self._mine_min_sol)  * 0.10)
        tex_bonus  = max(0.0, (self._mine_max_tex - texture_std)  / self._mine_max_tex * 0.10)
        confidence = min(1.0, self._mine_base_conf + circ_bonus + sol_bonus + tex_bonus)

        return _Candidate(
            contour      = contour,
            bbox         = bbox,
            area         = area,
            solidity     = solidity,
            aspect_ratio = aspect_ratio,
            class_name   = MineClass.SURFACE_MINE,
            confidence   = confidence,
        )

    def _evaluate_marker(
        self,
        contour, bbox, area: float,
        solidity: float, aspect_ratio: float,
        circularity: float,
        hsv_frame: Optional[np.ndarray],
    ) -> Optional[_Candidate]:
        """Apply buried_marker filters and compute heuristic confidence."""

        # Circularity gate (looser than mine — marker may be peg/ring shape)
        if circularity < self._mrk_min_circ:
            return None

        # Aspect ratio gate
        if not (self._mrk_min_ar <= aspect_ratio <= self._mrk_max_ar):
            return None

        # Solidity gate
        if solidity < self._mrk_min_sol:
            return None

        confidence = self._mrk_base_conf

        # Optional color gate — boosts confidence significantly if color matches
        if self._color_gate_on and hsv_frame is not None:
            color_fill = self._compute_color_fill(contour, hsv_frame)
            if color_fill < self._color_min_fill:
                return None
            color_bonus = min(0.30, color_fill * 0.30)
            confidence  = min(1.0, confidence + color_bonus)

        return _Candidate(
            contour      = contour,
            bbox         = bbox,
            area         = area,
            solidity     = solidity,
            aspect_ratio = aspect_ratio,
            class_name   = MineClass.BURIED_MARKER,
            confidence   = confidence,
        )

    # ------------------------------------------------------------------
    # Pixel-level analysis helpers
    # ------------------------------------------------------------------

    def _compute_pixel_stats(
        self,
        contour: np.ndarray,
        bx: int, by: int, bw: int, bh: int,
        fh: int, fw: int,
        gray: np.ndarray,
        frame_bgr: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute texture_std and color_std for pixels inside the contour.

        Returns
        -------
        texture_std : float
            Std dev of grayscale values inside the contour mask.
            Low → uniform flat disc. High → skin texture / fabric / shadow.

        color_std : float
            Mean of per-channel BGR std devs inside the contour mask.
            Low → one solid colour. High → mixed tones (skin, gradients).
        """
        # Clamp bbox to frame bounds
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(fw, bx + bw)
        y2 = min(fh, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 999.0, 999.0

        # Build a local mask (contour shifted to ROI coordinates)
        roi_h, roi_w = y2 - y1, x2 - x1
        local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        shifted = contour - np.array([x1, y1])
        cv2.drawContours(local_mask, [shifted], -1, 255, thickness=cv2.FILLED)

        # Extract interior pixels
        gray_roi   = gray[y1:y2, x1:x2]
        gray_pix   = gray_roi[local_mask > 0]

        if len(gray_pix) == 0:
            return 999.0, 999.0

        texture_std = float(np.std(gray_pix))

        # Color std — per-channel, then averaged
        bgr_roi = frame_bgr[y1:y2, x1:x2]
        ch_stds = []
        for c in range(3):
            ch_pix = bgr_roi[:, :, c][local_mask > 0]
            ch_stds.append(float(np.std(ch_pix)))
        color_std = float(np.mean(ch_stds))

        return texture_std, color_std

    def _compute_max_defect(self, contour: np.ndarray) -> float:
        """
        Compute the depth (in pixels) of the deepest convexity defect.

        A convexity defect is a concavity where the contour dips inward from
        the convex hull. For a smooth disc → near 0px. For a hand →
        finger-gap defects of 10-50px at working resolution.

        Returns
        -------
        float
            Maximum defect depth in pixels. Returns 0.0 if no defects found.
        """
        try:
            # returnPoints=False needed for convexityDefects
            hull_idx = cv2.convexHull(contour, returnPoints=False)
            if hull_idx is None or len(hull_idx) < 3:
                return 0.0
            defects = cv2.convexityDefects(contour, hull_idx)
            if defects is None or len(defects) == 0:
                return 0.0
            # cv2.convexityDefects returns (N, 1, 4) in most builds but
            # (N, 4) in some — reshape to (-1, 4) to handle both safely.
            # Columns: [start_idx, end_idx, far_idx, depth*256]
            defects_flat = defects.reshape(-1, 4)
            return float(defects_flat[:, 3].max()) / 256.0
        except cv2.error:
            # Can fail on degenerate/self-intersecting contours — treat as no defects
            return 0.0

    # ------------------------------------------------------------------
    # Color gate helper (marker only)
    # ------------------------------------------------------------------

    def _compute_color_fill(
        self,
        contour: np.ndarray,
        hsv_frame: np.ndarray,
    ) -> float:
        """
        Fraction of contour interior pixels matching the configured HSV range.
        Returns float in [0.0, 1.0].
        """
        h, w = hsv_frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        color_mask = cv2.inRange(hsv_frame, self._color_low1, self._color_high1)
        if self._use_hue_wrap:
            color_mask2 = cv2.inRange(hsv_frame, self._color_low2, self._color_high2)
            color_mask  = cv2.bitwise_or(color_mask, color_mask2)

        interior_pixels = cv2.countNonZero(mask)
        if interior_pixels == 0:
            return 0.0

        matching_pixels = cv2.countNonZero(cv2.bitwise_and(color_mask, mask))
        return matching_pixels / interior_pixels
