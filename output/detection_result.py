"""
output/detection_result.py
──────────────────────────
Detection result dataclasses for the Landmine Detection Module.

These structures are the OUTPUT CONTRACT consumed by downstream modules
(e.g., mapping, localization). Change field names or types here with care.

JSON serialization is supported out of the box via `to_dict()` / `from_dict()`
for easy IPC between the detection module and a future mapping module.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------------------------------

class MineClass:
    """String constants for class labels — use these, not raw strings."""
    SURFACE_MINE   = "surface_mine"    # Visible mine disc on the ground
    BURIED_MARKER  = "buried_marker"   # Surface marker above a buried mine
    UNKNOWN        = "unknown"         # Fallback / low-confidence candidate


# ---------------------------------------------------------------------------
# SINGLE DETECTION
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    One detected object in a single frame.

    Fields
    ------
    class_name : str
        One of MineClass.SURFACE_MINE, MineClass.BURIED_MARKER, MineClass.UNKNOWN.

    bbox : tuple[int, int, int, int]
        Bounding box in working-resolution pixel coordinates: (x, y, w, h)
        where (x, y) is the top-left corner.
        NOTE: These are in the *working* resolution (e.g. 480×480), NOT the
        native camera resolution. Use FrameResult.working_resolution +
        FrameResult.scale_x / scale_y to map back to original coordinates.

    contour : np.ndarray or None
        Raw contour point array, shape (N, 1, 2), dtype int32. May be None
        if the detector backend doesn't produce per-pixel contours (e.g. YOLO).
        Useful for precise shape overlay and future segmentation mask generation.

    confidence : float
        Detection confidence in range [0.0, 1.0].
        For the classical pipeline this is a heuristic score (not a calibrated
        probability). For YOLO it is the model's objectness × class probability.

    frame_id : int
        Monotonically increasing frame counter from the start of the session.

    timestamp_ms : float
        Wall-clock timestamp in milliseconds since epoch when the frame was
        captured. Use this (not frame_id) for time-series reasoning in the
        mapping module.

    detector_backend : str
        Which backend produced this detection: "classical" | "yolo" | "hybrid".
    """

    class_name:       str
    bbox:             tuple                  # (x, y, w, h) in working-res pixels
    confidence:       float
    frame_id:         int
    timestamp_ms:     float
    detector_backend: str
    contour:          Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def center(self) -> tuple[int, int]:
        """Center of the bounding box (cx, cy)."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    @property
    def area(self) -> int:
        """Bounding box area in pixels²."""
        _, _, w, h = self.bbox
        return w * h

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialize to a JSON-compatible dict.
        Contour is excluded (ndarray not JSON-serializable) — include
        contour data separately if needed for the mapping module.
        """
        return {
            "class_name":       self.class_name,
            "bbox":             list(self.bbox),        # [x, y, w, h]
            "center":           list(self.center),      # [cx, cy]
            "confidence":       round(self.confidence, 4),
            "frame_id":         self.frame_id,
            "timestamp_ms":     self.timestamp_ms,
            "detector_backend": self.detector_backend,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        """Deserialize from a dict produced by to_dict()."""
        return cls(
            class_name       = d["class_name"],
            bbox             = tuple(d["bbox"]),
            confidence       = d["confidence"],
            frame_id         = d["frame_id"],
            timestamp_ms     = d["timestamp_ms"],
            detector_backend = d["detector_backend"],
            contour          = None,
        )


# ---------------------------------------------------------------------------
# FRAME-LEVEL RESULT
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """
    All detections from a single video frame.

    This is what downstream modules (mapping, logging) receive per frame.

    Fields
    ------
    frame_id : int
        Monotonically increasing frame counter.

    timestamp_ms : float
        Wall-clock timestamp in ms when the frame was captured.

    detections : List[Detection]
        All confirmed detections in this frame, sorted by confidence descending.

    working_resolution : tuple[int, int]
        (width, height) of the frame at which detection ran.
        Needed to interpret Detection.bbox correctly.

    scale_x, scale_y : float
        Multiply a working-res coordinate by these to get native-res coordinates.
        Example: native_x = working_x * scale_x

    processing_time_ms : float
        Total time to process this frame (capture → output), in milliseconds.
    """

    frame_id:            int
    timestamp_ms:        float
    detections:          List[Detection]
    working_resolution:  tuple               # (w, h)
    scale_x:             float = 1.0
    scale_y:             float = 1.0
    processing_time_ms:  float = 0.0

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0

    @property
    def surface_mines(self) -> List[Detection]:
        return [d for d in self.detections if d.class_name == MineClass.SURFACE_MINE]

    @property
    def buried_markers(self) -> List[Detection]:
        return [d for d in self.detections if d.class_name == MineClass.BURIED_MARKER]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict (one line per frame in JSONL output)."""
        return {
            "frame_id":           self.frame_id,
            "timestamp_ms":       self.timestamp_ms,
            "working_resolution": list(self.working_resolution),
            "scale_x":            self.scale_x,
            "scale_y":            self.scale_y,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "detections":         [d.to_dict() for d in self.detections],
        }
