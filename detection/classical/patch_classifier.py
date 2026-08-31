"""
detection/classical/patch_classifier.py
────────────────────────────────────────
Optional patch-level CNN classifier stub.

This module is a STUB — it currently passes detections through unchanged
(returns the heuristic confidence from ShapeFilter as-is).

When you have ~150+ labeled mine/non-mine patch images, train a small CNN
(MobileNetV2-tiny or a 4-conv-layer net) and export it to ONNX format.
Then:
  1. Set `patch_classifier.enabled: true` in config.yaml.
  2. Set `patch_classifier.model_path` to your .onnx file.
  3. The real classifier will be loaded here and will re-score each detection
     by cropping the candidate patch from the frame and running inference.

No other module changes are needed — the interface is already wired.

Integration note for future CNN implementation
-----------------------------------------------
Input:  64×64 BGR patch cropped from the detection bounding box
Output: float in [0, 1] — probability that the patch contains a mine/marker
        (trained as a binary classifier: mine=1, background=0)

Recommended architecture:
  Conv(32, 3×3) → BN → ReLU → MaxPool(2)
  Conv(64, 3×3) → BN → ReLU → MaxPool(2)
  Conv(128, 3×3) → BN → ReLU → GlobalAvgPool
  Dense(64) → ReLU → Dense(1) → Sigmoid
  ~250K params — runs in <5 ms/patch on Pi 5 CPU

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from output.detection_result import Detection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PatchClassifier
# ---------------------------------------------------------------------------

class PatchClassifier:
    """
    Optional CNN patch classifier that re-scores shape-filtered detections.

    When disabled (default), acts as a transparent pass-through.
    When enabled, crops each candidate from the frame and runs ONNX inference.

    Parameters
    ----------
    cfg : dict
        The `patch_classifier` section of config.yaml.
    """

    def __init__(self, cfg: dict):
        self._enabled       = bool(cfg.get("enabled", False))
        self._model_path    = str(cfg.get("model_path", ""))
        self._input_size    = int(cfg.get("input_size", 64))
        self._score_thresh  = float(cfg.get("score_threshold", 0.55))
        self._session       = None   # ONNX Runtime InferenceSession (loaded lazily)

        if self._enabled:
            self._load_model()
        else:
            logger.info(
                "PatchClassifier: DISABLED (stub pass-through). "
                "Enable in config.yaml once you have a trained model."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> List[Detection]:
        """
        Re-score and optionally filter detections using the patch classifier.

        Parameters
        ----------
        detections : List[Detection]
            Detections from ShapeFilter — passed through as-is when disabled.
        frame : np.ndarray
            Preprocessed BGR frame to crop patches from.

        Returns
        -------
        List[Detection]
            Same list with updated confidence scores, or unchanged if disabled.
        """
        if not self._enabled or self._session is None:
            # Stub: pass through unchanged
            return detections

        confirmed = []
        for det in detections:
            patch_score = self._infer_patch(det.bbox, frame)
            if patch_score is None or patch_score >= self._score_thresh:
                # Blend heuristic + CNN score: give CNN score more weight
                blended_conf = (
                    0.35 * det.confidence + 0.65 * (patch_score or det.confidence)
                )
                # Create new Detection with updated confidence (dataclasses are immutable-ish)
                confirmed.append(Detection(
                    class_name       = det.class_name,
                    bbox             = det.bbox,
                    contour          = det.contour,
                    confidence       = round(min(1.0, blended_conf), 4),
                    frame_id         = det.frame_id,
                    timestamp_ms     = det.timestamp_ms,
                    detector_backend = det.detector_backend,
                ))
        return confirmed

    # ------------------------------------------------------------------
    # Model loading (ONNX Runtime)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load ONNX model via onnxruntime (lazy — only when enabled)."""
        if not Path(self._model_path).exists():
            logger.warning(
                "PatchClassifier: model file not found at '%s'. "
                "Falling back to stub pass-through.",
                self._model_path,
            )
            self._enabled = False
            return

        try:
            import onnxruntime as ort   # type: ignore
            self._session = ort.InferenceSession(
                self._model_path,
                providers=["CPUExecutionProvider"],
            )
            self._input_name  = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            logger.info(
                "PatchClassifier: loaded model from '%s' (input: %s)",
                self._model_path, self._input_name,
            )
        except ImportError:
            logger.error(
                "PatchClassifier: onnxruntime not installed. "
                "Run: pip install onnxruntime"
            )
            self._enabled = False
        except Exception as exc:
            logger.error("PatchClassifier: failed to load model: %s", exc)
            self._enabled = False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_patch(
        self,
        bbox: tuple,
        frame: np.ndarray,
    ) -> Optional[float]:
        """
        Crop the bounding box from the frame, resize to model input size,
        and run ONNX inference.

        Returns
        -------
        float or None
            Patch probability in [0, 1], or None if cropping/inference fails.
        """
        if self._session is None:
            return None

        x, y, w, h = bbox
        fh, fw = frame.shape[:2]

        # Clamp crop to frame bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)

        if x2 <= x1 or y2 <= y1:
            return None

        patch = frame[y1:y2, x1:x2]
        patch = cv2.resize(patch, (self._input_size, self._input_size))

        # Normalize to [0, 1] and convert to NCHW float32
        patch_f = patch.astype(np.float32) / 255.0
        patch_f = np.transpose(patch_f, (2, 0, 1))   # HWC → CHW
        patch_f = np.expand_dims(patch_f, axis=0)     # → (1, 3, H, W)

        try:
            outputs = self._session.run(
                [self._output_name], {self._input_name: patch_f}
            )
            score = float(outputs[0].flatten()[0])
            # Apply sigmoid if model outputs logits (detect by range)
            if score > 1.0 or score < 0.0:
                score = 1.0 / (1.0 + np.exp(-score))
            return float(np.clip(score, 0.0, 1.0))
        except Exception as exc:
            logger.warning("PatchClassifier inference error: %s", exc)
            return None
