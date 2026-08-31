"""
detection/classical/shape_filter.py
────────────────────────────────────
9-Stage Geometric, Topological, and Photometric Shape Filtering Engine.

Robofest Gujarat 6.0 | Aerial Robotics | Senior Division

Evaluates raw candidate contours produced by CandidateProposer against a sequential
hierarchy of discriminative gates engineered to differentiate planar landmine discs
and tactical subsurface markers from scene clutter (floor tiles, wires, human hands,
fabric, gravel, and organic ground shadows):

Sequential Filter Pipeline (Ordered by Computational Complexity):
  1. Area Range Gate: Bounded by metric flight altitude projection table [min_area_px, max_area_px].
  2. Bounding-Box Aspect Ratio Gate: Rejects elongated linear artifacts (wires, ruts, cracks).
  3. Circularity / Isoperimetric Quotient Gate: Q = (4 * pi * Area) / Perimeter^2 >= 0.50.
  4. Convex Hull Solidity Gate: S = Area / Hull_Area >= 0.65.
  5. Polygonal Vertex Count Gate (RDP Rectangle Rejection): N_v >= 7 (approx_epsilon_frac=0.02).
  6. Maximum Convexity Defect Depth Gate: Scans boundary concavity depth (D_max <= 10.0 px).
  7. Interior Grayscale Texture Dispersion Gate: Masked pixel std dev (sigma_gray <= 65.0).
  8. Interior BGR Color Dispersion Gate: Mean per-channel standard deviation (sigma_BGR <= 70.0).
  9. Heuristic Confidence Fusion: Blends circularity, solidity, and photometric uniformity metrics.

Buried Marker Perception:
  Evaluated through dedicated geometric gates and optional race-day calibrated HSV color manifold
  boundaries with full hue wrap-around support for red pigments ([0, 10] U [170, 179]).

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


# ===========================================================================
# Internal Candidate Representation
# ===========================================================================

@dataclass(frozen=True)
class _Candidate:
    """Internal candidate structure prior to final Detection contract generation."""
    contour:       np.ndarray
    bbox:          Tuple[int, int, int, int]   # (x, y, w, h) in working resolution
    area:          float
    solidity:      float
    aspect_ratio:  float
    class_name:    str
    confidence:    float


# ===========================================================================
# Shape Filter Engine
# ===========================================================================

class ShapeFilter:
    """
    Stateful geometry and texture classification engine.

    Parameters
    ----------
    mine_cfg : dict
        The `surface_mine` section from config.yaml.
    marker_cfg : dict
        The `buried_marker` section from config.yaml.
    """

    def __init__(self, mine_cfg: dict, marker_cfg: dict) -> None:
        # --- Surface Mine Filter Parameters ---
        self._mine_min_area: float = float(mine_cfg.get("min_area_px", 400.0))
        self._mine_max_area: float = float(mine_cfg.get("max_area_px", 20000.0))
        self._mine_min_ar: float = float(mine_cfg.get("min_aspect_ratio", 0.40))
        self._mine_max_ar: float = float(mine_cfg.get("max_aspect_ratio", 2.50))
        self._mine_min_sol: float = float(mine_cfg.get("min_solidity", 0.65))
        self._mine_min_ext: float = float(mine_cfg.get("min_extent", 0.0))

        # Circularity (Isoperimetric Quotient: 4*pi*Area / Perimeter^2)
        self._mine_min_circ: float = float(mine_cfg.get("min_circularity", 0.50))

        # Ramer-Douglas-Peucker (RDP) Polygon Simplification Vertex Gate
        self._mine_min_verts: int = int(mine_cfg.get("min_approx_vertices", 7))
        self._mine_eps_frac: float = float(mine_cfg.get("approx_epsilon_frac", 0.02))

        # Convexity Defect Depth Gate (pixels at working canvas resolution)
        self._mine_max_defect: float = float(mine_cfg.get("max_convexity_defect_px", 10.0))

        # Interior Grayscale and Color Variance Limits
        self._mine_max_tex: float = float(mine_cfg.get("max_texture_std", 65.0))
        self._mine_max_color: float = float(mine_cfg.get("max_color_std", 70.0))
        self._mine_base_conf: float = float(mine_cfg.get("base_confidence", 0.50))

        # --- Buried Marker Filter Parameters ---
        self._mrk_enabled: bool = bool(marker_cfg.get("enabled", False))
        self._mrk_min_area: float = float(marker_cfg.get("min_area_px", 30.0))
        self._mrk_max_area: float = float(marker_cfg.get("max_area_px", 3000.0))
        self._mrk_min_ar: float = float(marker_cfg.get("min_aspect_ratio", 0.15))
        self._mrk_max_ar: float = float(marker_cfg.get("max_aspect_ratio", 6.0))
        self._mrk_min_sol: float = float(marker_cfg.get("min_solidity", 0.30))
        self._mrk_min_circ: float = float(marker_cfg.get("min_circularity", 0.25))
        self._mrk_base_conf: float = float(marker_cfg.get("base_confidence", 0.40))

        # --- Race-Day Marker HSV Color Manifold Gating ---
        self._color_gate_on: bool = bool(marker_cfg.get("use_color_gate", False))
        self._color_low1: np.ndarray = np.array(marker_cfg.get("marker_color_hsv_low", [0, 100, 80]), dtype=np.uint8)
        self._color_high1: np.ndarray = np.array(marker_cfg.get("marker_color_hsv_high", [10, 255, 255]), dtype=np.uint8)
        self._use_hue_wrap: bool = bool(marker_cfg.get("use_hue_wrap", False))
        self._color_low2: np.ndarray = np.array(marker_cfg.get("marker_color_hsv_low2", [170, 100, 80]), dtype=np.uint8)
        self._color_high2: np.ndarray = np.array(marker_cfg.get("marker_color_hsv_high2", [179, 255, 255]), dtype=np.uint8)
        self._color_min_fill: float = float(marker_cfg.get("color_gate_min_fill", 0.30))

        logger.info(
            "ShapeFilter initialized (mine_area=[%.0f, %.0f], circ>=%.2f, tex_std<=%.0f, defect<=%.1fpx | marker=%s)",
            self._mine_min_area, self._mine_max_area,
            self._mine_min_circ, self._mine_max_tex, self._mine_max_defect,
            "ENABLED" if self._mrk_enabled else "DISABLED",
        )

    # -----------------------------------------------------------------------
    # Public Filtering Method
    # -----------------------------------------------------------------------

    def filter(
        self,
        contours: list,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
        detector_backend: str = "classical",
    ) -> List[Detection]:
        """
        Evaluate raw contour populations against geometric, topological, and photometric discriminators.

        Parameters
        ----------
        contours : list of np.ndarray
            Raw topological contours from CandidateProposer.propose().
        frame : np.ndarray
            Preprocessed BGR working frame used for interior pixel extraction.
        frame_id : int
            Monotonically increasing integer frame counter.
        timestamp_ms : float
            Capture epoch timestamp in milliseconds.
        detector_backend : str, default="classical"
            Backend provenance tag.

        Returns
        -------
        List[Detection]
            Confirmed detection records, sorted by confidence score in descending order.
        """
        # Pre-compute color representations once per frame
        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_frame: Optional[np.ndarray] = None
        if self._color_gate_on:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        fh, fw = frame.shape[:2]
        detections: List[Detection] = []

        for contour in contours:
            area: float = cv2.contourArea(contour)
            if area < 1.0:
                continue

            # ---------------------------------------------------------------
            # Stage 1: Area Boundary Gate
            # ---------------------------------------------------------------
            is_mine_size: bool = self._mine_min_area <= area <= self._mine_max_area
            is_marker_size: bool = self._mrk_min_area <= area <= self._mrk_max_area

            if not is_mine_size and not is_marker_size:
                continue

            # ---------------------------------------------------------------
            # Stage 2: Geometric & Topological Metrics (Pure Mathematics)
            # ---------------------------------------------------------------
            bbox: Tuple[int, int, int, int] = cv2.boundingRect(contour)
            bx, by, bw, bh = bbox
            aspect_ratio: float = bw / bh if bh > 0 else 0.0

            hull: np.ndarray = cv2.convexHull(contour)
            hull_area: float = cv2.contourArea(hull)
            solidity: float = area / hull_area if hull_area > 0 else 0.0
            extent: float = area / (bw * bh) if (bw * bh) > 0 else 0.0

            # Isoperimetric Quotient / Circularity: 4*pi*Area / Perimeter^2
            perimeter: float = cv2.arcLength(contour, closed=True)
            circularity: float = (
                (4.0 * 3.141592653589793 * area / (perimeter * perimeter))
                if perimeter > 0 else 0.0
            )

            # RDP Polygon Simplification (Rectangle / Frame Rejection)
            epsilon: float = self._mine_eps_frac * perimeter
            approx: np.ndarray = cv2.approxPolyDP(contour, epsilon, closed=True)
            n_verts: int = len(approx)

            # ---------------------------------------------------------------
            # Stage 3: Photometric & Defect Metrics (Guarded Execution)
            # ---------------------------------------------------------------
            geom_ok_mine: bool = (
                is_mine_size
                and n_verts >= self._mine_min_verts
                and circularity >= self._mine_min_circ
                and self._mine_min_ar <= aspect_ratio <= self._mine_max_ar
                and solidity >= self._mine_min_sol
            )

            texture_std: float = 999.0
            color_std: float = 999.0
            max_defect_px: float = 999.0

            if geom_ok_mine:
                texture_std, color_std = self._compute_pixel_stats(
                    contour, bx, by, bw, bh, fh, fw, gray, frame
                )
                max_defect_px = self._compute_max_defect(contour)

            # ---------------------------------------------------------------
            # Stage 4: Target Classification
            # ---------------------------------------------------------------
            candidate: Optional[_Candidate] = None

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

            # ---------------------------------------------------------------
            # Stage 5: Detection Contract Instantiation
            # ---------------------------------------------------------------
            detections.append(Detection(
                class_name       = candidate.class_name,
                bbox             = candidate.bbox,
                contour          = candidate.contour,
                confidence       = candidate.confidence,
                frame_id         = frame_id,
                timestamp_ms     = timestamp_ms,
                detector_backend = detector_backend,
            ))

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.debug(
            "ShapeFilter: %d contours -> %d detections (%d mines, %d markers)",
            len(contours), len(detections),
            sum(1 for d in detections if d.class_name == MineClass.SURFACE_MINE),
            sum(1 for d in detections if d.class_name == MineClass.BURIED_MARKER),
        )

        return detections

    # -----------------------------------------------------------------------
    # Internal Discriminator Classifiers
    # -----------------------------------------------------------------------

    def _evaluate_mine(
        self,
        contour: np.ndarray,
        bbox: Tuple[int, int, int, int],
        area: float,
        solidity: float,
        aspect_ratio: float,
        extent: float,
        circularity: float,
        n_verts: int,
        texture_std: float,
        color_std: float,
        max_defect_px: float,
    ) -> Optional[_Candidate]:
        """
        Evaluate surface landmine discriminators and compute heuristic confidence.
        """
        # Gate 1: RDP Vertex Count Gate (Rejects 4-vertex quadrilaterals / floor tiles)
        if n_verts < self._mine_min_verts:
            return None

        # Gate 2: Circularity Gate
        if circularity < self._mine_min_circ:
            return None

        # Gate 3: Aspect Ratio Bounding
        if not (self._mine_min_ar <= aspect_ratio <= self._mine_max_ar):
            return None

        # Gate 4: Convex Hull Solidity
        if solidity < self._mine_min_sol:
            return None

        # Gate 5: Convexity Defect Maximum Depth Gate (Rejects finger notches)
        if max_defect_px > self._mine_max_defect:
            return None

        # Gate 6: Grayscale Interior Texture Standard Deviation
        if texture_std > self._mine_max_tex:
            return None

        # Gate 7: BGR Color Dispersion
        if self._mine_max_color > 0 and color_std > self._mine_max_color:
            return None

        # Gate 8: Optional Extent Bounding
        if self._mine_min_ext > 0 and extent < self._mine_min_ext:
            return None

        # Stage 9: Heuristic Confidence Fusion
        circ_bonus: float = max(0.0, (circularity - self._mine_min_circ) * 0.20)
        sol_bonus: float = max(0.0, (solidity - self._mine_min_sol) * 0.10)
        tex_bonus: float = max(0.0, (self._mine_max_tex - texture_std) / self._mine_max_tex * 0.10)
        confidence: float = min(1.0, self._mine_base_conf + circ_bonus + sol_bonus + tex_bonus)

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
        contour: np.ndarray,
        bbox: Tuple[int, int, int, int],
        area: float,
        solidity: float,
        aspect_ratio: float,
        circularity: float,
        hsv_frame: Optional[np.ndarray],
    ) -> Optional[_Candidate]:
        """
        Evaluate buried mine marker discriminators and color manifold fill.
        """
        if circularity < self._mrk_min_circ:
            return None

        if not (self._mrk_min_ar <= aspect_ratio <= self._mrk_max_ar):
            return None

        if solidity < self._mrk_min_sol:
            return None

        confidence: float = self._mrk_base_conf

        # Optional race-day calibrated color manifold gating
        if self._color_gate_on and hsv_frame is not None:
            color_fill: float = self._compute_color_fill(contour, hsv_frame)
            if color_fill < self._color_min_fill:
                return None
            color_bonus: float = min(0.30, color_fill * 0.30)
            confidence = min(1.0, confidence + color_bonus)

        return _Candidate(
            contour      = contour,
            bbox         = bbox,
            area         = area,
            solidity     = solidity,
            aspect_ratio = aspect_ratio,
            class_name   = MineClass.BURIED_MARKER,
            confidence   = confidence,
        )

    # -----------------------------------------------------------------------
    # Pixel-Level Statistics & Convexity Analysis Helpers
    # -----------------------------------------------------------------------

    def _compute_pixel_stats(
        self,
        contour: np.ndarray,
        bx: int, by: int, bw: int, bh: int,
        fh: int, fw: int,
        gray: np.ndarray,
        frame_bgr: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Extract masked pixels inside the candidate contour and compute texture/color standard deviation.

        Returns
        -------
        texture_std : float
            Standard deviation of grayscale values inside contour.
        color_std : float
            Mean of per-channel BGR standard deviations inside contour.
        """
        x1: int = max(0, bx)
        y1: int = max(0, by)
        x2: int = min(fw, bx + bw)
        y2: int = min(fh, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 999.0, 999.0

        roi_h, roi_w = y2 - y1, x2 - x1
        local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        shifted = contour - np.array([x1, y1])
        cv2.drawContours(local_mask, [shifted], -1, 255, thickness=cv2.FILLED)

        gray_roi = gray[y1:y2, x1:x2]
        gray_pix = gray_roi[local_mask > 0]

        if len(gray_pix) == 0:
            return 999.0, 999.0

        texture_std: float = float(np.std(gray_pix))

        bgr_roi = frame_bgr[y1:y2, x1:x2]
        ch_stds = [float(np.std(bgr_roi[:, :, c][local_mask > 0])) for c in range(3)]
        color_std: float = float(np.mean(ch_stds))

        return texture_std, color_std

    def _compute_max_defect(self, contour: np.ndarray) -> float:
        """
        Compute maximum orthogonal convexity defect depth in pixels.

        Smooth circular discs produce near-zero defect depths, whereas branched
        structures (e.g. human fingers, weeds) produce large depths (> 15 px).
        """
        try:
            hull_idx = cv2.convexHull(contour, returnPoints=False)
            if hull_idx is None or len(hull_idx) < 3:
                return 0.0
            defects = cv2.convexityDefects(contour, hull_idx)
            if defects is None or len(defects) == 0:
                return 0.0
            defects_flat = defects.reshape(-1, 4)
            return float(defects_flat[:, 3].max()) / 256.0
        except cv2.error:
            return 0.0

    def _compute_color_fill(
        self,
        contour: np.ndarray,
        hsv_frame: np.ndarray,
    ) -> float:
        """
        Compute fraction of interior contour pixels matching the calibrated HSV color manifold.
        """
        h, w = hsv_frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        color_mask = cv2.inRange(hsv_frame, self._color_low1, self._color_high1)
        if self._use_hue_wrap:
            color_mask2 = cv2.inRange(hsv_frame, self._color_low2, self._color_high2)
            color_mask = cv2.bitwise_or(color_mask, color_mask2)

        interior_pixels = cv2.countNonZero(mask)
        if interior_pixels == 0:
            return 0.0

        matching_pixels = cv2.countNonZero(cv2.bitwise_and(color_mask, mask))
        return matching_pixels / interior_pixels
