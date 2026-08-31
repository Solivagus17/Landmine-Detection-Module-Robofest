"""
detection/__init__.py
──────────────────────
Polymorphic Detector Factory and Subsystem Pipeline Strategy Abstraction.

Robofest Gujarat 6.0 | Aerial Robotics | Senior Division

Provides a unified facade (`Detector`) wrapping interchangeable detection backends:
  1. Classical CV Pipeline (Default):
       CandidateProposer (Canny + Adaptive Gaussian + IoU Dedup)
         └──▶ ShapeFilter (9-Stage Geometric & Statistical Rejection)
               └──▶ PatchClassifier (Optional CNN Verification Stub)
  2. YOLO Neural Pipeline (Upgrade Slot):
       YOLODetector (Ultralytics / ONNX / OpenVINO / Myriad X VPU Engine)

Switching between execution backends is achieved via a single configuration key
in `config.yaml` (`detector_backend: "classical"` or `"yolo"`).

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
from typing import List, Protocol

import numpy as np

from output.detection_result import Detection

logger = logging.getLogger(__name__)


# ===========================================================================
# Detector Pipeline Protocol (Interface Specification)
# ===========================================================================

class _DetectorBackend(Protocol):
    """Protocol defining the core `.detect()` interface for all backend implementations."""

    def detect(self, frame: np.ndarray, frame_id: int, timestamp_ms: float) -> List[Detection]:
        ...


# ===========================================================================
# Unified Detector Facade
# ===========================================================================

class Detector:
    """
    High-level polymorphic detector facade exposing a uniform API to `main.py`.

    Parameters
    ----------
    cfg : dict
        Complete parsed configuration parameter dictionary from config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        backend: str = cfg.get("detector_backend", "classical").lower()
        self._backend_name: str = backend

        if backend == "yolo":
            self._detector: _DetectorBackend = self._build_yolo(cfg)
        else:
            # Default production path: deterministic classical CV pipeline
            self._detector = self._build_classical(cfg)

        logger.info("Detector subsystem initialized (active_backend=%s)", self._backend_name)

    # -----------------------------------------------------------------------
    # Public Inference Method
    # -----------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
    ) -> List[Detection]:
        """
        Execute target perception on a preprocessed BGR working frame.

        Parameters
        ----------
        frame : np.ndarray
            Conditioned BGR image of shape (target_h, target_w, 3) from FramePreprocessor.
        frame_id : int
            Monotonically increasing integer frame counter.
        timestamp_ms : float
            Epoch timestamp in milliseconds recorded at frame ingestion.

        Returns
        -------
        List[Detection]
            Confirmed detection records, sorted by confidence in descending order.
        """
        return self._detector.detect(frame, frame_id, timestamp_ms)

    @property
    def backend_name(self) -> str:
        """Return active detector backend identifier string ('classical' | 'yolo')."""
        return self._backend_name

    # -----------------------------------------------------------------------
    # Internal Pipeline Builders
    # -----------------------------------------------------------------------

    def _build_classical(self, cfg: dict) -> _ClassicalPipeline:
        """Instantiate the 4-stage classical computer vision pipeline."""
        from detection.classical.candidate_proposal import CandidateProposer
        from detection.classical.patch_classifier import PatchClassifier
        from detection.classical.shape_filter import ShapeFilter

        return _ClassicalPipeline(
            proposer   = CandidateProposer(cfg.get("candidate_proposal", {})),
            shape_filt = ShapeFilter(
                mine_cfg   = cfg.get("surface_mine", {}),
                marker_cfg = cfg.get("buried_marker", {}),
            ),
            classifier = PatchClassifier(cfg.get("patch_classifier", {})),
            out_cfg    = cfg.get("output", {}),
        )

    def _build_yolo(self, cfg: dict) -> _YoloPipeline:
        """Instantiate the neural YOLO inference pipeline wrapper."""
        from detection.yolo.yolo_detector import YOLODetector

        return _YoloPipeline(
            yolo_det = YOLODetector(cfg.get("yolo", {})),
            out_cfg  = cfg.get("output", {}),
        )


# ===========================================================================
# Pipeline Implementations
# ===========================================================================

class _ClassicalPipeline:
    """
    Sequential execution wrapper for the classical vision pipeline.

    Chains: CandidateProposer -> ShapeFilter -> PatchClassifier -> ConfidenceGate.
    """

    def __init__(self, proposer, shape_filt, classifier, out_cfg: dict) -> None:
        self._proposer = proposer
        self._shape_filt = shape_filt
        self._classifier = classifier
        self._conf_gate: float = float(out_cfg.get("confidence_threshold", 0.35))

    def detect(self, frame: np.ndarray, frame_id: int, timestamp_ms: float) -> List[Detection]:
        # Stage 1: Dual-stream candidate contour proposal & IoU deduplication
        contours = self._proposer.propose(frame)

        # Stage 2: 9-stage geometric, topological, and photometric classification
        detections: List[Detection] = self._shape_filt.filter(
            contours         = contours,
            frame            = frame,
            frame_id         = frame_id,
            timestamp_ms     = timestamp_ms,
            detector_backend = "classical",
        )

        # Stage 3: Optional patch-level deep learning verification
        detections = self._classifier.classify(detections, frame)

        # Stage 4: Strict confidence threshold gating
        detections = [d for d in detections if d.confidence >= self._conf_gate]

        return detections


class _YoloPipeline:
    """
    Execution wrapper for the YOLO deep learning detector.

    Dispatches preprocessed frames to YOLODetector and applies confidence gating.
    """

    def __init__(self, yolo_det, out_cfg: dict) -> None:
        self._yolo = yolo_det
        self._conf_gate: float = float(out_cfg.get("confidence_threshold", 0.35))

    def detect(self, frame: np.ndarray, frame_id: int, timestamp_ms: float) -> List[Detection]:
        detections: List[Detection] = self._yolo.detect(frame, frame_id, timestamp_ms)
        return [d for d in detections if d.confidence >= self._conf_gate]


# ===========================================================================
# Factory Function
# ===========================================================================

def build_detector(cfg: dict) -> Detector:
    """
    Construct and return the configured Detector instance.

    Parameters
    ----------
    cfg : dict
        Parsed config.yaml dictionary.

    Returns
    -------
    Detector
        Ready-to-use detector facade instance.
    """
    return Detector(cfg)
