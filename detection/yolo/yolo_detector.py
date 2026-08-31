"""
detection/yolo/yolo_detector.py
────────────────────────────────
YOLO inference stub — future upgrade slot for the classical CV pipeline.

This module is intentionally a STUB. It defines the same interface as the
classical detector so it can be swapped in via config.yaml:
  detector_backend: "yolo"

When you are ready to use YOLO:
  1. Train YOLOv8n or YOLO11n on your labeled dataset.
  2. Export to ONNX (laptop/Pi CPU): yolo export model=best.pt format=onnx
     OR export to NCNN (Pi 5 optimized): yolo export model=best.pt format=ncnn
  3. Set model_path in config.yaml.
  4. Set yolo.enabled: true in config.yaml.
  5. This stub will load the model and run real inference automatically.

The output format is identical to the classical pipeline — List[Detection] —
so no other modules need to change when you switch backends.

Dataset guidance (brief — expand separately when requested)
-----------------------------------------------------------
- Minimum: 150 surface_mine + 100 buried_marker images (real photos)
- Recommended: 300 + 200, multiple altitudes and lighting conditions
- Labeling tool: Roboflow (free tier) — bounding box labels, YOLOv8 export
- Augmentation: Roboflow built-in (flip, HSV jitter, mosaic, rotate ±15°)
- A 300-image set → ~900–1200 augmented → sufficient for 2-class YOLOv8n
  to reach ~70–80% mAP on simple disc-shaped objects

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from output.detection_result import Detection, MineClass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YOLODetector
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    YOLO-based detector stub.

    When enabled and a model path is configured, loads the model and runs
    inference. Otherwise logs a clear warning and returns no detections.

    Parameters
    ----------
    cfg : dict
        The `yolo` section of config.yaml.
    """

    def __init__(self, cfg: dict):
        self._enabled       = bool(cfg.get("enabled", False))
        self._model_path    = str(cfg.get("model_path", ""))
        self._input_size    = int(cfg.get("input_size", 320))
        self._conf_thresh   = float(cfg.get("confidence_threshold", 0.40))
        self._iou_thresh    = float(cfg.get("iou_threshold", 0.45))
        self._class_names   = list(cfg.get("class_names", [
            MineClass.SURFACE_MINE,
            MineClass.BURIED_MARKER,
        ]))
        self._model         = None

        if self._enabled:
            self._load_model()
        else:
            logger.info(
                "YOLODetector: DISABLED (stub). "
                "To enable: set yolo.enabled=true and yolo.model_path in config.yaml."
            )

    # ------------------------------------------------------------------
    # Public API — identical interface to classical pipeline
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
    ) -> List[Detection]:
        """
        Run YOLO inference on a preprocessed BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Preprocessed BGR frame at working resolution.
        frame_id : int
            Current frame counter.
        timestamp_ms : float
            Wall-clock timestamp in ms.

        Returns
        -------
        List[Detection]
            YOLO detections in the same format as the classical pipeline.
            Returns empty list when stub is not yet loaded.
        """
        if not self._enabled or self._model is None:
            return []

        return self._run_inference(frame, frame_id, timestamp_ms)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Attempt to load the YOLO model via ultralytics."""
        if not Path(self._model_path).exists():
            logger.warning(
                "YOLODetector: model not found at '%s'. "
                "Returning empty detections until model is available.",
                self._model_path,
            )
            return

        try:
            from ultralytics import YOLO   # type: ignore
            self._model = YOLO(self._model_path)
            # Warm up
            dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False)
            logger.info(
                "YOLODetector: model loaded from '%s' (input: %dx%d, conf: %.2f)",
                self._model_path, self._input_size, self._input_size, self._conf_thresh,
            )
        except ImportError:
            logger.error(
                "YOLODetector: ultralytics not installed. "
                "Run: pip install ultralytics"
            )
        except Exception as exc:
            logger.error("YOLODetector: failed to load model: %s", exc)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
    ) -> List[Detection]:
        """
        Run ultralytics YOLO inference and convert results to Detection list.
        """
        try:
            results = self._model.predict(
                frame,
                imgsz       = self._input_size,
                conf        = self._conf_thresh,
                iou         = self._iou_thresh,
                verbose     = False,
            )
        except Exception as exc:
            logger.error("YOLODetector inference error: %s", exc)
            return []

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # xyxy format → convert to xywh
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x, y, w, h     = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                cls_idx    = int(boxes.cls[i].item())
                conf       = float(boxes.conf[i].item())
                class_name = (
                    self._class_names[cls_idx]
                    if cls_idx < len(self._class_names)
                    else MineClass.UNKNOWN
                )

                detections.append(Detection(
                    class_name       = class_name,
                    bbox             = (x, y, w, h),
                    contour          = None,   # YOLO doesn't produce contours
                    confidence       = round(conf, 4),
                    frame_id         = frame_id,
                    timestamp_ms     = timestamp_ms,
                    detector_backend = "yolo",
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
