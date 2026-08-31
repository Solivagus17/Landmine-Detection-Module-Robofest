"""
detection/yolo/yolo_detector.py
────────────────────────────────
Ultralytics YOLO Deep Learning Object Detection Interface (Plug-in Upgrade Slot).

Robofest Gujarat 6.0 | Aerial Robotics | Senior Division

Provides a unified interface conforming to the standard `Detector` contract, allowing
seamless replacement of the classical computer vision pipeline with a trained YOLO model
(e.g., YOLOv8n or YOLO11n) via a single `config.yaml` directive (`detector_backend: "yolo"`).

Deployment Options:
  1. Host CPU (Raspberry Pi Zero 2 W):
     Export trained model to ONNX:
       yolo export model=best.pt format=onnx imgsz=320
  2. Edge VPU Offload (Luxonis OAK-D Lite Intel Myriad X):
     Export trained model to OpenVINO IR / Myriad Blob format for zero-host-CPU inference.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from output.detection_result import Detection, MineClass

logger = logging.getLogger(__name__)


# ===========================================================================
# YOLO Detector Engine
# ===========================================================================

class YOLODetector:
    """
    Ultralytics YOLO inference wrapper.

    Parameters
    ----------
    cfg : dict
        The `yolo` section from config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._enabled: bool = bool(cfg.get("enabled", False))
        self._model_path: str = str(cfg.get("model_path", ""))
        self._input_size: int = int(cfg.get("input_size", 320))
        self._conf_thresh: float = float(cfg.get("confidence_threshold", 0.40))
        self._iou_thresh: float = float(cfg.get("iou_threshold", 0.45))
        self._class_names: List[str] = list(cfg.get("class_names", [
            MineClass.SURFACE_MINE,
            MineClass.BURIED_MARKER,
        ]))
        self._model = None

        if self._enabled:
            self._load_model()
        else:
            logger.info(
                "YOLODetector: DISABLED (stub mode). "
                "To enable: set yolo.enabled=true and specify yolo.model_path in config.yaml."
            )

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
        Execute YOLO object detection inference on a conditioned BGR working frame.

        Parameters
        ----------
        frame : np.ndarray
            Preprocessed BGR image of shape (target_h, target_w, 3) from FramePreprocessor.
        frame_id : int
            Monotonically increasing integer frame counter.
        timestamp_ms : float
            Frame acquisition epoch timestamp in milliseconds.

        Returns
        -------
        List[Detection]
            List of detected objects conforming to the unified Detection dataclass contract.
        """
        if not self._enabled or self._model is None:
            return []

        return self._run_inference(frame, frame_id, timestamp_ms)

    # -----------------------------------------------------------------------
    # Model Loading
    # -----------------------------------------------------------------------

    def _load_model(self) -> None:
        """Instantiate YOLO neural model via ultralytics."""
        if not Path(self._model_path).exists():
            logger.warning(
                "YOLODetector: Model file not found at '%s'. Returning empty detections until weights are deployed.",
                self._model_path,
            )
            return

        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(self._model_path)

            # Warm-up inference pass
            dummy: np.ndarray = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False)
            logger.info(
                "YOLODetector: Successfully loaded model '%s' (input=%dx%d, conf_thresh=%.2f)",
                self._model_path, self._input_size, self._input_size, self._conf_thresh,
            )
        except ImportError:
            logger.error(
                "YOLODetector: ultralytics package not installed. Install via: pip install ultralytics"
            )
        except Exception as exc:
            logger.error("YOLODetector: Failed to load model weights: %s", exc)

    # -----------------------------------------------------------------------
    # Tensor Execution & Conversion
    # -----------------------------------------------------------------------

    def _run_inference(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
    ) -> List[Detection]:
        """
        Execute prediction pass and convert output bounding tensors to standard Detection objects.
        """
        try:
            results = self._model.predict(
                frame,
                imgsz   = self._input_size,
                conf    = self._conf_thresh,
                iou     = self._iou_thresh,
                verbose = False,
            )
        except Exception as exc:
            logger.error("YOLODetector prediction exception: %s", exc)
            return []

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Convert xyxy tensor to (x, y, w, h)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                cls_idx: int = int(boxes.cls[i].item())
                conf: float = float(boxes.conf[i].item())
                class_name: str = (
                    self._class_names[cls_idx]
                    if cls_idx < len(self._class_names)
                    else MineClass.UNKNOWN
                )

                detections.append(Detection(
                    class_name       = class_name,
                    bbox             = (x, y, w, h),
                    contour          = None,  # Neural bounding boxes lack discrete polygon contours
                    confidence       = round(conf, 4),
                    frame_id         = frame_id,
                    timestamp_ms     = timestamp_ms,
                    detector_backend = "yolo",
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
