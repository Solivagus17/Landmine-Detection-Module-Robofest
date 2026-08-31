"""
output/detection_result.py
──────────────────────────
Perception Output Contracts and Serial Data Structures.

Robofest Gujarat 6.0 | Aerial Robotics | Senior Division

Defines the formal downstream data contract consumed by mapping, SLAM, ray-casting,
and swarm coordination layers:
  - `MineClass`: String constants for semantic class labels ('surface_mine', 'buried_marker', 'unknown').
  - `Detection`: Record of an individual detected target within a single frame.
  - `FrameResult`: Aggregated frame-level perception record with serialization methods (`.to_dict()`)
    for IPC streaming over standard output (JSONL), TCP/UDP, or UNIX domain sockets.

Coordinate Convention:
  Bounding boxes `bbox` are structured as `(x, y, w, h)` in working canvas coordinates (e.g. 480x480).
  Downstream modules can re-project to native camera sensor coordinates using `scale_x`, `scale_y`
  and the letterbox padding offsets.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# Target Semantic Class Enumeration
# ===========================================================================

class MineClass:
    """String constants defining semantic target classes."""
    SURFACE_MINE: str  = "surface_mine"   # Circular/oval planar landmine disc
    BURIED_MARKER: str = "buried_marker"  # Tactical surface indicator for subsurface hazard
    UNKNOWN: str       = "unknown"        # Unclassified anomaly / fallback candidate


# ===========================================================================
# Individual Target Detection Dataclass
# ===========================================================================

@dataclass
class Detection:
    """
    Data contract encapsulating a single detected target instance.

    Fields
    ------
    class_name : str
        Target semantic classification label from `MineClass`.
    bbox : Tuple[int, int, int, int]
        Axis-aligned bounding box (x, y, w, h) in working canvas coordinates.
    confidence : float
        Detection confidence metric in range [0.0, 1.0].
    frame_id : int
        Monotonically increasing integer frame sequence index.
    timestamp_ms : float
        Epoch capture timestamp in milliseconds for time-series flight telemetry fusion.
    detector_backend : str
        Provenance tag indicating generating backend ('classical' | 'yolo').
    contour : Optional[np.ndarray], default=None
        Raw 2D contour point array of shape (N, 1, 2) dtype int32 (excluded from JSON serialization).
    """

    class_name:       str
    bbox:             Tuple[int, int, int, int]
    confidence:       float
    frame_id:         int
    timestamp_ms:     float
    detector_backend: str
    contour:          Optional[np.ndarray] = field(default=None, repr=False)

    # -----------------------------------------------------------------------
    # Spatial Properties
    # -----------------------------------------------------------------------

    @property
    def center(self) -> Tuple[int, int]:
        """Compute bounding box geometric centroid (cx, cy)."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    @property
    def area(self) -> int:
        """Compute 2D bounding rectangle pixel area (width * height)."""
        _, _, w, h = self.bbox
        return w * h

    # -----------------------------------------------------------------------
    # Serialization Methods
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize detection record to a standard Python dictionary for JSON streaming.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable dictionary representation.
        """
        return {
            "class_name":       self.class_name,
            "bbox":             list(self.bbox),
            "center":           list(self.center),
            "confidence":       round(self.confidence, 4),
            "frame_id":         self.frame_id,
            "timestamp_ms":     self.timestamp_ms,
            "detector_backend": self.detector_backend,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Detection:
        """
        Reconstruct a Detection dataclass instance from a dictionary record.

        Parameters
        ----------
        d : Dict[str, Any]
            Dictionary created via .to_dict().

        Returns
        -------
        Detection
            Deserialized instance.
        """
        return cls(
            class_name       = d["class_name"],
            bbox             = tuple(d["bbox"]),  # type: ignore
            confidence       = float(d["confidence"]),
            frame_id         = int(d["frame_id"]),
            timestamp_ms     = float(d["timestamp_ms"]),
            detector_backend = d["detector_backend"],
            contour          = None,
        )


# ===========================================================================
# Frame-Level Aggregation Dataclass
# ===========================================================================

@dataclass
class FrameResult:
    """
    Encapsulates all perception outputs and metadata for a single video frame.

    Fields
    ------
    frame_id : int
        Monotonically increasing integer frame sequence index.
    timestamp_ms : float
        Epoch capture timestamp in milliseconds.
    detections : List[Detection]
        List of confirmed detections in this frame.
    working_resolution : Tuple[int, int]
        (width, height) of the processing canvas in pixels.
    scale_x : float, default=1.0
        Horizontal scaling factor relating working coordinates to native frame.
    scale_y : float, default=1.0
        Vertical scaling factor relating working coordinates to native frame.
    processing_time_ms : float, default=0.0
        Total compute latency for this frame in milliseconds.
    """

    frame_id:           int
    timestamp_ms:       float
    detections:         List[Detection]
    working_resolution: Tuple[int, int]
    scale_x:            float = 1.0
    scale_y:            float = 1.0
    processing_time_ms: float = 0.0

    # -----------------------------------------------------------------------
    # Filtering Properties
    # -----------------------------------------------------------------------

    @property
    def has_detections(self) -> bool:
        """True if one or more confirmed detections exist in this frame."""
        return len(self.detections) > 0

    @property
    def surface_mines(self) -> List[Detection]:
        """Filter detections matching MineClass.SURFACE_MINE."""
        return [d for d in self.detections if d.class_name == MineClass.SURFACE_MINE]

    @property
    def buried_markers(self) -> List[Detection]:
        """Filter detections matching MineClass.BURIED_MARKER."""
        return [d for d in self.detections if d.class_name == MineClass.BURIED_MARKER]

    # -----------------------------------------------------------------------
    # Serialization Methods
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize frame result to a JSON-serializable dictionary (JSONL format).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing frame metadata and serialized detection records.
        """
        return {
            "frame_id":           self.frame_id,
            "timestamp_ms":       self.timestamp_ms,
            "working_resolution": list(self.working_resolution),
            "scale_x":            self.scale_x,
            "scale_y":            self.scale_y,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "detections":         [d.to_dict() for d in self.detections],
        }
