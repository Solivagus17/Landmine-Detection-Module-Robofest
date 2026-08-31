"""
detection/classical/candidate_proposal.py
──────────────────────────────────────────
Candidate region proposal for the classical CV detection pipeline.

This module generates a list of raw candidate contours from a preprocessed
frame using two complementary methods:

  Method 1 — Canny edges (primary):
    Detects the boundary of a mine disc via edge gradients. Works well when
    the mine has a distinct color/texture difference from the ground.

  Method 2 — Adaptive thresholding (secondary):
    Finds flat uniform blobs that Canny misses (e.g., low-contrast mine on
    similar-tone terrain). Produces filled binary regions rather than edges.

When config sets method = "both", both methods run and their contour sets are
merged + deduplicated by IoU before being passed to the shape filter.

Key design choices
------------------
- NO hardcoded color thresholds here — color is not used as a primary cue.
- Contour extraction, NOT ellipse fitting — handles irregular shapes.
- All thresholds are driven by config.yaml — zero magic numbers in code.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Type alias: a contour is an ndarray of shape (N, 1, 2), dtype int32
Contour = np.ndarray


# ---------------------------------------------------------------------------
# CandidateProposer
# ---------------------------------------------------------------------------

class CandidateProposer:
    """
    Generates raw candidate contours from a preprocessed BGR frame.

    Parameters
    ----------
    cfg : dict
        The `candidate_proposal` section of config.yaml.
    """

    def __init__(self, cfg: dict):
        self._method        = cfg.get("method", "both").lower()

        # Canny params
        self._canny_low     = int(cfg.get("canny_low",  30))
        self._canny_high    = int(cfg.get("canny_high", 90))
        self._dil_kernel_sz = int(cfg.get("canny_dilation_kernel", 5))
        self._dil_iters     = int(cfg.get("canny_dilation_iters",  2))

        # Adaptive threshold params
        self._adap_block    = int(cfg.get("adaptive_block_size", 31))
        self._adap_c        = int(cfg.get("adaptive_c", 4))

        # Deduplication
        self._dedup_iou     = float(cfg.get("dedup_iou_threshold", 0.3))

        # Pre-build dilation kernel (avoid re-creating per frame)
        self._dil_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self._dil_kernel_sz, self._dil_kernel_sz),
        )

        # Validate adaptive block size — must be odd and > 1
        if self._adap_block % 2 == 0:
            self._adap_block += 1
        if self._adap_block < 3:
            self._adap_block = 3

        logger.info(
            "CandidateProposer ready — method: %s, canny: [%d, %d], "
            "adaptive_block: %d, dedup_iou: %.2f",
            self._method, self._canny_low, self._canny_high,
            self._adap_block, self._dedup_iou,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(self, frame: np.ndarray) -> List[Contour]:
        """
        Generate candidate contours from a preprocessed BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Preprocessed BGR frame at working resolution, shape (H, W, 3).

        Returns
        -------
        List[np.ndarray]
            List of raw contours, each shape (N, 1, 2) dtype int32.
            These are NOT yet filtered by size/shape — pass to ShapeFilter next.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        contours_canny = []
        contours_adap  = []

        if self._method in ("canny", "both"):
            contours_canny = self._propose_canny(gray)

        if self._method in ("adaptive_thresh", "both"):
            contours_adap = self._propose_adaptive(gray)

        if self._method == "both":
            # Merge and deduplicate
            all_contours = self._merge_and_dedup(contours_canny, contours_adap)
        elif self._method == "canny":
            all_contours = contours_canny
        else:
            all_contours = contours_adap

        logger.debug(
            "CandidateProposer: canny=%d, adaptive=%d → merged=%d candidates",
            len(contours_canny), len(contours_adap), len(all_contours),
        )

        return all_contours

    # ------------------------------------------------------------------
    # Method 1: Canny edges
    # ------------------------------------------------------------------

    def _propose_canny(self, gray: np.ndarray) -> List[Contour]:
        """
        Find candidate contours via Canny edge detection.

        Pipeline:
          grayscale → Canny → dilate (close edge gaps) → findContours
        """
        edges = cv2.Canny(gray, self._canny_low, self._canny_high)

        # Dilate edges to close gaps around mine boundaries.
        # Without this, broken edges produce many small arc contours instead of
        # one closed contour around the mine.
        dilated = cv2.dilate(edges, self._dil_kernel, iterations=self._dil_iters)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours)

    # ------------------------------------------------------------------
    # Method 2: Adaptive thresholding
    # ------------------------------------------------------------------

    def _propose_adaptive(self, gray: np.ndarray) -> List[Contour]:
        """
        Find candidate contours via adaptive Gaussian thresholding.

        Detects locally-bright or locally-dark blobs against the surrounding
        ground texture — effective for low-contrast mine discs.

        Pipeline:
          grayscale → adaptiveThreshold (inverted) → morphological opening
          (noise removal) → findContours
        """
        # ADAPTIVE_THRESH_GAUSSIAN_C with THRESH_BINARY_INV:
        # pixels darker than local neighborhood mean → white (foreground)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self._adap_block,
            self._adap_c,
        )

        # Morphological opening: remove single-pixel noise dots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _merge_and_dedup(
        self,
        contours_a: List[Contour],
        contours_b: List[Contour],
    ) -> List[Contour]:
        """
        Merge two contour lists and remove duplicates using bounding-box IoU.

        Strategy: start with contours_a (Canny), then for each contour in
        contours_b (adaptive), add it only if it doesn't overlap significantly
        with any already-added contour. This prioritises Canny contours (which
        tend to be tighter around the mine boundary) when both methods fire on
        the same object.
        """
        if not contours_a:
            return contours_b
        if not contours_b:
            return contours_a

        merged = list(contours_a)
        bboxes_merged = [cv2.boundingRect(c) for c in merged]

        for c_new in contours_b:
            bbox_new = cv2.boundingRect(c_new)
            is_dup   = any(
                _bbox_iou(bbox_new, bm) >= self._dedup_iou
                for bm in bboxes_merged
            )
            if not is_dup:
                merged.append(c_new)
                bboxes_merged.append(bbox_new)

        return merged


# ---------------------------------------------------------------------------
# Helper: bounding-box IoU
# ---------------------------------------------------------------------------

def _bbox_iou(
    bbox_a: Tuple[int, int, int, int],
    bbox_b: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection over Union of two (x, y, w, h) bounding boxes.

    Returns a float in [0.0, 1.0].
    """
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    # Convert to (x1, y1, x2, y2)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    # Intersection
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h

    if inter == 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
