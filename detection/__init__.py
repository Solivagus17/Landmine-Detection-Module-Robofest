"""
detection/__init__.py
──────────────────────
Detector factory — builds and returns the active detector from config.

Usage (in main.py):
    from detection import build_detector
    detector = build_detector(cfg)
    detections = detector.detect(frame, frame_id, timestamp_ms)

Switching backends requires only a config.yaml change:
    detector_backend: "classical"   # default
    detector_backend: "yolo"        # when model is ready

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
import time
from typing import List

import numpy as np

from output.detection_result import Detection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified Detector wrapper
# ---------------------------------------------------------------------------

class Detector:
    """
    Unified detector interface. Wraps either the classical CV pipeline
    or YOLO inference — same .detect() call from main.py regardless.

    Parameters
    ----------
    cfg : dict
        Full config dict (all sections of config.yaml).
    """

    def __init__(self, cfg: dict):
        backend = cfg.get("detector_backend", "classical").lower()
        self._backend_name = backend

        if backend == "yolo":
            self._detector = self._build_yolo(cfg)
        else:
            # Default: classical CV pipeline
            self._detector = self._build_classical(cfg)

        logger.info("Detector initialized — backend: %s", self._backend_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
    ) -> List[Detection]:
        """
        Run detection on a preprocessed BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Preprocessed frame from FramePreprocessor.
        frame_id : int
            Frame counter (monotonically increasing).
        timestamp_ms : float
            Wall-clock time in ms when the frame was captured.

        Returns
        -------
        List[Detection]
            Confirmed detections, sorted by confidence descending.
        """
        return self._detector.detect(frame, frame_id, timestamp_ms)

    @property
    def backend_name(self) -> str:
        return self._backend_name

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_classical(self, cfg: dict) -> "_ClassicalPipeline":
        from detection.classical.candidate_proposal import CandidateProposer
        from detection.classical.shape_filter import ShapeFilter
        from detection.classical.patch_classifier import PatchClassifier
        return _ClassicalPipeline(
            proposer   = CandidateProposer(cfg.get("candidate_proposal", {})),
            shape_filt = ShapeFilter(
                mine_cfg   = cfg.get("surface_mine", {}),
                marker_cfg = cfg.get("buried_marker", {}),
            ),
            classifier = PatchClassifier(cfg.get("patch_classifier", {})),
            out_cfg    = cfg.get("output", {}),
        )

    def _build_yolo(self, cfg: dict) -> "_YoloPipeline":
        from detection.yolo.yolo_detector import YOLODetector
        return _YoloPipeline(
            yolo_det = YOLODetector(cfg.get("yolo", {})),
            out_cfg  = cfg.get("output", {}),
        )


# ---------------------------------------------------------------------------
# Classical pipeline wrapper
# ---------------------------------------------------------------------------

class _ClassicalPipeline:
    """Chains CandidateProposer → ShapeFilter → PatchClassifier."""

    def __init__(self, proposer, shape_filt, classifier, out_cfg):
        self._proposer   = proposer
        self._shape_filt = shape_filt
        self._classifier = classifier
        self._conf_gate  = float(out_cfg.get("confidence_threshold", 0.35))

    def detect(self, frame, frame_id, timestamp_ms) -> List[Detection]:
        # Stage 1: candidate generation
        contours = self._proposer.propose(frame)

        # Stage 2: shape filtering + classification
        detections = self._shape_filt.filter(
            contours    = contours,
            frame       = frame,
            frame_id    = frame_id,
            timestamp_ms = timestamp_ms,
            detector_backend = "classical",
        )

        # Stage 3: optional CNN patch re-scoring
        detections = self._classifier.classify(detections, frame)

        # Stage 4: final confidence gate
        detections = [d for d in detections if d.confidence >= self._conf_gate]

        return detections


# ---------------------------------------------------------------------------
# YOLO pipeline wrapper
# ---------------------------------------------------------------------------

class _YoloPipeline:
    """Wraps YOLODetector with the same interface."""

    def __init__(self, yolo_det, out_cfg):
        self._yolo      = yolo_det
        self._conf_gate = float(out_cfg.get("confidence_threshold", 0.35))

    def detect(self, frame, frame_id, timestamp_ms) -> List[Detection]:
        detections = self._yolo.detect(frame, frame_id, timestamp_ms)
        return [d for d in detections if d.confidence >= self._conf_gate]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_detector(cfg: dict) -> Detector:
    """
    Build and return the active detector from config.

    Parameters
    ----------
    cfg : dict
        Full parsed config.yaml as a dict.

    Returns
    -------
    Detector
        Ready-to-use detector. Call .detect(frame, frame_id, ts) per frame.
    """
    return Detector(cfg)
