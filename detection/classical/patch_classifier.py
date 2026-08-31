"""
detection/classical/patch_classifier.py
────────────────────────────────────────
Deep Learning Patch-Level Verification Classifier (Plug-in Stub Interface).

Robofest Gujarat 6.0 | Aerial Robotics | Senior Division

Interface Architecture:
  Acts as an optional secondary stage following the 9-stage ShapeFilter.
  - When disabled (`enabled: false`), transparently passes candidate detections through unchanged.
  - When enabled (`enabled: true`), crops candidate bounding boxes from the working frame,
    scales to (input_size x input_size x 3), and executes ONNX Runtime inference to compute
    a neural verification score. The neural confidence is then blended with the classical
    heuristic score:
      Confidence_blended = 0.35 * Heuristic_Conf + 0.65 * CNN_Score

Target Deployment:
  Designed for lightweight MobileNetV2-tiny or custom 4-layer CNN architectures (~250K parameters)
  running via ONNX Runtime CPUExecutionProvider on Raspberry Pi Zero 2 W or offloaded to the
  Luxonis OAK-D Lite Intel Movidius Myriad X VPU.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from output.detection_result import Detection

logger = logging.getLogger(__name__)


# ===========================================================================
# Patch Classifier Engine
# ===========================================================================

class PatchClassifier:
    """
    ONNX-based patch verification classifier.

    Parameters
    ----------
    cfg : dict
        The `patch_classifier` section from config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._enabled: bool = bool(cfg.get("enabled", False))
        self._model_path: str = str(cfg.get("model_path", ""))
        self._input_size: int = int(cfg.get("input_size", 64))
        self._score_thresh: float = float(cfg.get("score_threshold", 0.55))
        self._session = None
        self._input_name: str = ""
        self._output_name: str = ""

        if self._enabled:
            self._load_model()
        else:
            logger.info(
                "PatchClassifier: DISABLED (pass-through mode). "
                "Set patch_classifier.enabled=true in config.yaml once trained ONNX model is available."
            )

    # -----------------------------------------------------------------------
    # Public Classification Method
    # -----------------------------------------------------------------------

    def classify(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> List[Detection]:
        """
        Re-score and verify candidate detections against the patch classifier.

        Parameters
        ----------
        detections : List[Detection]
            Candidate detections from ShapeFilter.
        frame : np.ndarray
            Preprocessed BGR working frame.

        Returns
        -------
        List[Detection]
            Verified detections with updated blended confidence scores.
        """
        if not self._enabled or self._session is None:
            # Pass-through mode when disabled
            return detections

        confirmed: List[Detection] = []
        for det in detections:
            patch_score: Optional[float] = self._infer_patch(det.bbox, frame)
            if patch_score is None or patch_score >= self._score_thresh:
                # Weighted confidence blend (35% classical heuristic, 65% neural classifier)
                blended_conf: float = (
                    0.35 * det.confidence + 0.65 * (patch_score if patch_score is not None else det.confidence)
                )
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

    # -----------------------------------------------------------------------
    # Model Loading (ONNX Runtime)
    # -----------------------------------------------------------------------

    def _load_model(self) -> None:
        """Instantiate ONNX Runtime InferenceSession on CPUExecutionProvider."""
        if not Path(self._model_path).exists():
            logger.warning(
                "PatchClassifier: ONNX model file not found at '%s'. Reverting to pass-through mode.",
                self._model_path,
            )
            self._enabled = False
            return

        try:
            import onnxruntime as ort  # type: ignore
            self._session = ort.InferenceSession(
                self._model_path,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            logger.info(
                "PatchClassifier: Successfully loaded model '%s' (input=%s, output=%s)",
                self._model_path, self._input_name, self._output_name,
            )
        except ImportError:
            logger.error(
                "PatchClassifier: onnxruntime package not installed. Install via: pip install onnxruntime"
            )
            self._enabled = False
        except Exception as exc:
            logger.error("PatchClassifier: Failed to load ONNX model: %s", exc)
            self._enabled = False

    # -----------------------------------------------------------------------
    # Tensor Inference
    # -----------------------------------------------------------------------

    def _infer_patch(
        self,
        bbox: Tuple[int, int, int, int],
        frame: np.ndarray,
    ) -> Optional[float]:
        """
        Crop candidate bounding box, format as NCHW tensor, and execute ONNX inference.

        Returns
        -------
        Optional[float]
            Predicted target probability in range [0.0, 1.0], or None if inference fails.
        """
        if self._session is None:
            return None

        x, y, w, h = bbox
        fh, fw = frame.shape[:2]

        x1: int = max(0, x)
        y1: int = max(0, y)
        x2: int = min(fw, x + w)
        y2: int = min(fh, y + h)

        if x2 <= x1 or y2 <= y1:
            return None

        patch: np.ndarray = frame[y1:y2, x1:x2]
        patch = cv2.resize(patch, (self._input_size, self._input_size))

        # Normalize to [0, 1] float32 and convert HWC -> NCHW
        patch_f: np.ndarray = patch.astype(np.float32) / 255.0
        patch_f = np.transpose(patch_f, (2, 0, 1))
        patch_f = np.expand_dims(patch_f, axis=0)

        try:
            outputs = self._session.run(
                [self._output_name], {self._input_name: patch_f}
            )
            score: float = float(outputs[0].flatten()[0])

            # Apply sigmoid activation if model outputs unnormalized logits
            if score > 1.0 or score < 0.0:
                score = 1.0 / (1.0 + float(np.exp(-score)))

            return float(np.clip(score, 0.0, 1.0))
        except Exception as exc:
            logger.warning("PatchClassifier inference failure: %s", exc)
            return None
