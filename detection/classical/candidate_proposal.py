"""
detection/classical/candidate_proposal.py
──────────────────────────────────────────
Dual-Stream Candidate Region Proposal and Spatial IoU Deduplication Engine.

Robofest Gujarat 6.0 | Aerial Robotics | Senior Division

Generates a candidate contour population from a preprocessed BGR frame using
two complementary feature extraction streams:

  Stream A — Canny Gradient Hysteresis (Primary Edge Stream):
    Extracts high-frequency boundary gradients via Sobel operators. Employs dual-threshold
    hysteresis (T_low=50, T_high=130) followed by a 5x5 square morphological dilation to bridge
    discontinuous boundary arcs along the outer rim of embedded mine discs.

  Stream B — Local Adaptive Gaussian Thresholding (Secondary Flat-Blob Stream):
    Computes a locally adaptive threshold across a 51x51 Gaussian window to extract flat,
    uniform planar targets with low boundary contrast against soil/gravel terrain. Followed
    by morphological opening and closing with a 3x3 elliptical kernel.

  Spatial IoU Deduplication:
    When `method: "both"` is configured, both streams run in parallel. Candidate contours from
    Stream B that spatially overlap Stream A candidates with Bounding-Box IoU >= 0.30 are
    suppressed, preserving high-precision gradient edge geometries.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Type alias: an OpenCV contour is an ndarray of shape (N, 1, 2) with dtype int32
Contour = np.ndarray


# ===========================================================================
# Candidate Region Proposal Engine
# ===========================================================================

class CandidateProposer:
    """
    Stateful candidate region proposer generating raw topological contour sets.

    Parameters
    ----------
    cfg : dict
        The `candidate_proposal` section from config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._method: str = cfg.get("method", "both").lower()

        # Stream A (Canny) Parameters
        self._canny_low: int = int(cfg.get("canny_low", 50))
        self._canny_high: int = int(cfg.get("canny_high", 130))
        self._dil_kernel_sz: int = int(cfg.get("canny_dilation_kernel", 5))
        self._dil_iters: int = int(cfg.get("canny_dilation_iters", 2))

        # Stream B (Adaptive Threshold) Parameters
        self._adap_block: int = int(cfg.get("adaptive_block_size", 51))
        self._adap_c: int = int(cfg.get("adaptive_c", 8))

        # Spatial Deduplication Parameter
        self._dedup_iou: float = float(cfg.get("dedup_iou_threshold", 0.3))

        # Pre-allocate morphological structuring elements
        self._dil_kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self._dil_kernel_sz, self._dil_kernel_sz),
        )
        self._morph_kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        # Enforce odd window size > 1 for adaptive Gaussian thresholding
        if self._adap_block % 2 == 0:
            self._adap_block += 1
        if self._adap_block < 3:
            self._adap_block = 3

        logger.info(
            "CandidateProposer initialized (method=%s, canny=[%d, %d], adap_block=%d, dedup_iou=%.2f)",
            self._method, self._canny_low, self._canny_high, self._adap_block, self._dedup_iou,
        )

    # -----------------------------------------------------------------------
    # Public Proposal Method
    # -----------------------------------------------------------------------

    def propose(self, frame: np.ndarray) -> List[Contour]:
        """
        Generate raw candidate contours from a conditioned BGR working frame.

        Parameters
        ----------
        frame : np.ndarray
            Preprocessed BGR image of shape (target_h, target_w, 3) from FramePreprocessor.

        Returns
        -------
        List[Contour]
            List of raw contour ndarrays of shape (N, 1, 2) dtype int32.
            These candidates are unclassified and must be passed to ShapeFilter.
        """
        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        contours_canny: List[Contour] = []
        contours_adap: List[Contour] = []

        if self._method in ("canny", "both"):
            contours_canny = self._propose_canny(gray)

        if self._method in ("adaptive_thresh", "both"):
            contours_adap = self._propose_adaptive(gray)

        if self._method == "both":
            all_contours: List[Contour] = self._merge_and_dedup(contours_canny, contours_adap)
        elif self._method == "canny":
            all_contours = contours_canny
        else:
            all_contours = contours_adap

        logger.debug(
            "CandidateProposer: canny=%d, adaptive=%d -> merged=%d candidates",
            len(contours_canny), len(contours_adap), len(all_contours),
        )

        return all_contours

    # -----------------------------------------------------------------------
    # Stream A: Canny Gradient Edge Proposal
    # -----------------------------------------------------------------------

    def _propose_canny(self, gray: np.ndarray) -> List[Contour]:
        """
        Extract candidate boundary contours via Sobel gradient magnitude and hysteresis.

        Sequence:
          Grayscale -> Canny Hysteresis -> Morphological Dilation -> findContours
        """
        edges = cv2.Canny(gray, self._canny_low, self._canny_high)

        # Dilation bridges discontinuous boundary arcs
        dilated = cv2.dilate(edges, self._dil_kernel, iterations=self._dil_iters)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours)

    # -----------------------------------------------------------------------
    # Stream B: Adaptive Gaussian Thresholding Proposal
    # -----------------------------------------------------------------------

    def _propose_adaptive(self, gray: np.ndarray) -> List[Contour]:
        """
        Extract candidate contours for low-contrast planar blobs.

        Sequence:
          Grayscale -> Inverted Gaussian Adaptive Threshold -> Morph Open/Close -> findContours
        """
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self._adap_block,
            self._adap_c,
        )

        # Morphological opening (noise dot suppression) and closing (hole sealing)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._morph_kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._morph_kernel, iterations=2)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours)

    # -----------------------------------------------------------------------
    # Spatial Deduplication
    # -----------------------------------------------------------------------

    def _merge_and_dedup(
        self,
        contours_a: List[Contour],
        contours_b: List[Contour],
    ) -> List[Contour]:
        """
        Merge candidate populations and suppress spatial duplicates via Bounding-Box IoU.

        Prioritizes Stream A (Canny) contours. Stream B contours overlapping any
        Stream A contour with IoU >= dedup_iou_threshold are discarded.
        """
        if not contours_a:
            return contours_b
        if not contours_b:
            return contours_a

        merged: List[Contour] = list(contours_a)
        bboxes_merged: List[Tuple[int, int, int, int]] = [cv2.boundingRect(c) for c in merged]

        for c_new in contours_b:
            bbox_new = cv2.boundingRect(c_new)
            is_dup = any(
                _bbox_iou(bbox_new, bm) >= self._dedup_iou
                for bm in bboxes_merged
            )
            if not is_dup:
                merged.append(c_new)
                bboxes_merged.append(bbox_new)

        return merged


# ===========================================================================
# Geometric Helper: Axis-Aligned Bounding Box IoU
# ===========================================================================

def _bbox_iou(
    bbox_a: Tuple[int, int, int, int],
    bbox_b: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection over Union (IoU) of two (x, y, w, h) bounding boxes.

    Returns
    -------
    float
        Overlap ratio in range [0.0, 1.0].
    """
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    # Intersection coordinates
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
