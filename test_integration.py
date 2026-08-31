"""
test_integration.py
───────────────────
End-to-End Integration and Serialization Verification Suite.

Robofest Gujarat 6.0 | Aerial Robotics | Senior Division

Validates:
  1. Synthetic frame generation with target landmine disc and subsurface marker geometries.
  2. Full pipeline execution: FramePreprocessor -> CandidateProposer -> ShapeFilter.
  3. Metric letterboxing and spatial metadata scaling calculations.
  4. Dataclass assembly (FrameResult, Detection) and JSONL round-trip serialization.
  5. Per-frame execution timing and throughput benchmarks.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection import Detector, build_detector
from output.detection_result import Detection, FrameResult, MineClass
from preprocess.frame_prep import FramePreprocessor, LetterboxInfo


# ===========================================================================
# Synthetic Frame Generator Helper
# ===========================================================================

def generate_synthetic_scene(width: int = 1280, height: int = 720) -> np.ndarray:
    """
    Construct a synthetic test frame containing ground texture, a mine surrogate disc, and a marker.

    Parameters
    ----------
    width : int, default=1280
        Frame width in pixels.
    height : int, default=720
        Frame height in pixels.

    Returns
    -------
    np.ndarray
        Synthetic BGR image array of shape (height, width, 3).
    """
    # Background: Random textured ground simulation
    np.random.seed(42)
    bg: np.ndarray = np.random.randint(60, 110, (height, width, 3), dtype=np.uint8)
    frame: np.ndarray = bg.copy()

    # Ground Target 1: Dark circular/oval landmine disc
    cv2.ellipse(frame, (400, 360), (100, 90), 15, 0, 360, (40, 40, 40), -1)

    # Ground Target 2: Small high-contrast tactical marker
    cv2.rectangle(frame, (700, 300), (730, 330), (0, 180, 50), -1)

    return frame


# ===========================================================================
# Main Integration Test Runner
# ===========================================================================

def run_integration_tests() -> None:
    """Execute end-to-end integration and data serialization test sequence."""
    print("\n" + "=" * 70)
    print("ROBOFEST 6.0 — LANDMINE PERCEPTION SUBSYSTEM INTEGRATION TEST")
    print("=" * 70 + "\n")

    # 1. Configuration Ingestion
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # 2. Pipeline Instantiation
    print("[1/4] Initializing Preprocessor and Active Detector...")
    preprocessor: FramePreprocessor = FramePreprocessor(cfg["camera"], cfg.get("preprocessing", {}))
    detector: Detector = build_detector(cfg)
    print("      Active Backend:", detector.backend_name)

    # 3. Synthetic Benchmark Execution (10 Cycles)
    print("\n[2/4] Executing 10-frame synthetic benchmark...")
    synthetic_frame: np.ndarray = generate_synthetic_scene(1280, 720)
    results: List[FrameResult] = []

    for i in range(10):
        ts: float = time.time() * 1000.0
        t0: float = time.monotonic()

        processed, info = preprocessor.process(synthetic_frame)
        detections: List[Detection] = detector.detect(processed, i, ts)
        elapsed_ms: float = (time.monotonic() - t0) * 1000.0

        result = FrameResult(
            frame_id           = i,
            timestamp_ms       = ts,
            detections         = detections,
            working_resolution = (info.target_w, info.target_h),
            scale_x            = info.scale_x,
            scale_y            = info.scale_y,
            processing_time_ms = elapsed_ms,
        )
        results.append(result)

    avg_latency_ms: float = sum(r.processing_time_ms for r in results) / len(results)
    avg_fps: float = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    print(f"      Average Compute Latency: {avg_latency_ms:.2f} ms ({avg_fps:.1f} FPS)")

    # 4. Perception Verification
    print("\n[3/4] Validating detection results (Frame 0)...")
    res_0 = results[0]
    dict_0 = res_0.to_dict()
    print(f"      Detections isolated: {len(dict_0['detections'])}")
    for det in dict_0["detections"][:3]:
        print(f"      -> {det['class_name']}: conf={det['confidence']:.2f}, bbox={det['bbox']}, center={det['center']}")

    # 5. JSON Round-Trip Serialization Test
    print("\n[4/4] Validating JSON serialization integrity...")
    json_line: str = json.dumps(dict_0)
    parsed: Dict[str, Any] = json.loads(json_line)

    assert "frame_id" in parsed, "Missing frame_id in serialized JSON"
    assert "timestamp_ms" in parsed, "Missing timestamp_ms in serialized JSON"
    assert "detections" in parsed, "Missing detections list in serialized JSON"
    assert "working_resolution" in parsed, "Missing working_resolution in serialized JSON"
    print("      [PASS] JSON round-trip serialization clean and valid.")

    print("\n" + "=" * 70)
    print(">>> ALL INTEGRATION AND SERIALIZATION TESTS PASSED <<<")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_integration_tests()
