"""
main.py
───────
Application Entry Point and Execution Orchestrator for the Landmine Detection Subsystem.

Robofest Gujarat 6.0 | Aerial Robotics (Senior Category) | Minefield Navigation

Pipeline Topology:
  CameraSource (Luxonis OAK-D Lite / UVC / Video / RTSP)
    └──▶ FramePreprocessor (Metric Letterbox + CIE LAB-CLAHE + Gaussian Blur)
          └──▶ Detector (Classical 9-Stage CV or YOLO Neural Backend)
                └──▶ FrameResult (Normalized Dataclass & Serializer)
                      ├──▶ Output Stream (JSON Lines to stdout / File IPC)
                      └──▶ DebugOverlay (Diagnostic Visualizer & Interactive HUD)

Target Hardware:
  - Host SBC: Raspberry Pi Zero 2 W (Quad-core ARM Cortex-A53 @ 1.0 GHz, 512 MB RAM)
  - Vision Sensor: Luxonis OAK-D Lite (Sony IMX214 RGB + Dual OV9782 Stereo Depth)

CLI Usage Examples:
  # Default: Ingest from default video capture node (/dev/video0 or webcam)
  python main.py

  # Ingest from flight test video recording or network RTSP stream
  python main.py --source flight_test.mp4
  python main.py --source rtsp://192.168.1.10:554/live

  # Specify custom YAML configuration file
  python main.py --config config.yaml

  # Production Headless Mode (Disables GUI windows for RPi Zero 2 W flight deployment)
  python main.py --no-debug

  # Stream serial JSON detection records to standard output for downstream SLAM/mapping
  python main.py --json

  # Record serial JSON detection records to a newline-delimited JSONL file
  python main.py --json --json-out detections.jsonl

Interactive GUI Controls (Active in Debug Mode):
  q / Escape : Terminate detection loop and release hardware resources
  p          : Toggle live stream pause/resume state
  s          : Capture and save current annotated frame as PNG in saved_frames/

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, TextIO

import cv2
import numpy as np
import yaml

# ===========================================================================
# System Path Initialization
# ===========================================================================
# Ensure the repository root directory is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ===========================================================================
# Subsystem Module Imports
# ===========================================================================
from capture.camera_source import CameraSource, build_source
from detection import Detector, build_detector
from output.detection_result import FrameResult
from preprocess.frame_prep import FramePreprocessor, LetterboxInfo
from visualization.debug_overlay import DebugOverlay


# ===========================================================================
# Logging Infrastructure
# ===========================================================================

def _setup_logging(verbose: bool = False) -> None:
    """
    Initialize system-wide console logging format and severity filter.

    Parameters
    ----------
    verbose : bool, default=False
        If True, sets root logging level to DEBUG. Otherwise defaults to INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
    )


# ===========================================================================
# Configuration Loader
# ===========================================================================

def load_config(path: str | Path) -> dict:
    """
    Parse and validate YAML configuration file from the filesystem.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the target YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration parameter registry dictionary.

    Raises
    ------
    FileNotFoundError
        If the target configuration path does not exist on disk.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            "Run from the project root directory, or pass --config <path>."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logging.getLogger(__name__).info("Loaded configuration: %s", config_path.resolve())
    return cfg


# ===========================================================================
# Command-Line Argument Parser
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """
    Construct command-line interface arguments and help documentation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Landmine & Subsurface Marker Detection Engine — Robofest Gujarat 6.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", default=None,
        help=(
            "Camera source override: Integer device index (e.g., 0 for /dev/video0), "
            "video file path, or rtsp:// URL. Overrides config.yaml camera.source."
        ),
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Disable OpenCV GUI visualization window (mandatory for headless drone deployment).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Stream JSON FrameResult records to standard output (one JSON line per frame).",
    )
    parser.add_argument(
        "--json-out", default=None, metavar="FILE",
        help="Append JSON detection records to a designated JSONL output file.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Activate verbose DEBUG-level logging.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Terminate execution after processing N frames (0 = run indefinitely).",
    )
    return parser.parse_args()


# ===========================================================================
# Main Execution Loop
# ===========================================================================

def run(args: argparse.Namespace) -> None:
    """
    Initialize all vision modules and execute the real-time perception event loop.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments from parse_args().
    """
    logger = logging.getLogger("main")

    # -----------------------------------------------------------------------
    # 1. Configuration Ingestion & CLI Overrides
    # -----------------------------------------------------------------------
    cfg = load_config(args.config)

    if args.source is not None:
        try:
            cfg["camera"]["source"] = int(args.source)
        except ValueError:
            cfg["camera"]["source"] = args.source

    if args.no_debug:
        cfg.setdefault("debug", {})["enabled"] = False

    if args.json:
        cfg.setdefault("output", {})["emit_json"] = True

    if args.json_out:
        cfg.setdefault("output", {})["json_output_file"] = args.json_out

    # -----------------------------------------------------------------------
    # 2. Pipeline Subsystem Construction
    # -----------------------------------------------------------------------
    logger.info("Initializing perception subsystems...")
    source: CameraSource = build_source(cfg["camera"])
    preprocessor: FramePreprocessor = FramePreprocessor(cfg["camera"], cfg.get("preprocessing", {}))
    detector: Detector = build_detector(cfg)
    overlay: DebugOverlay = DebugOverlay(cfg.get("debug", {}))

    # -----------------------------------------------------------------------
    # 3. File Descriptors & Output Stream Setup
    # -----------------------------------------------------------------------
    json_file: Optional[TextIO] = None
    json_path: str = cfg.get("output", {}).get("json_output_file", "")
    emit_json: bool = bool(cfg.get("output", {}).get("emit_json", False))

    if json_path:
        json_file = open(json_path, "w", encoding="utf-8")
        logger.info("Streaming JSON output to file: %s", json_path)

    # -----------------------------------------------------------------------
    # 4. Perception Event Loop
    # -----------------------------------------------------------------------
    logger.info("Starting detection loop. Press 'q' or Ctrl+C to terminate.")
    frame_id: int = 0
    paused: bool = False
    saved: int = 0
    save_dir: Path = PROJECT_ROOT / "saved_frames"

    try:
        while True:
            # Handle interactive pause state
            if paused:
                time.sleep(0.05)
                key = cv2.waitKey(50) & 0xFF
                if key == ord("p"):
                    paused = False
                    logger.info("Perception loop resumed.")
                elif key in (ord("q"), 27):
                    break
                continue

            # Stage A: Frame Acquisition & Timestamping
            t_capture: float = time.monotonic()
            timestamp_ms: float = time.time() * 1000.0
            frame = source.read()
            if frame is None:
                logger.info("Input stream exhausted or disconnected. Terminating loop.")
                break

            # Stage B: Photometric Conditioning & Letterboxing
            processed, lb_info = preprocessor.process(frame)

            # Stage C: Detection & Spatial Extraction
            t_detect: float = time.monotonic()
            detections = detector.detect(processed, frame_id, timestamp_ms)
            t_done: float = time.monotonic()

            processing_ms: float = (t_done - t_capture) * 1000.0
            detect_ms: float = (t_done - t_detect) * 1000.0

            # Stage D: Data Contract Assembly
            result = FrameResult(
                frame_id           = frame_id,
                timestamp_ms       = timestamp_ms,
                detections         = detections,
                working_resolution = (lb_info.target_w, lb_info.target_h),
                scale_x            = lb_info.scale_x,
                scale_y            = lb_info.scale_y,
                processing_time_ms = processing_ms,
            )

            # Stage E: Serial Output & IPC Streaming
            if emit_json:
                line = json.dumps(result.to_dict())
                print(line, flush=True)
            if json_file:
                json_file.write(json.dumps(result.to_dict()) + "\n")
                json_file.flush()

            # Stage F: Telemetry Logging
            if detections:
                logger.debug(
                    "Frame %d: %d target(s) detected in %.1f ms (detect: %.1f ms) — %s",
                    frame_id,
                    len(detections),
                    processing_ms,
                    detect_ms,
                    [(d.class_name, round(d.confidence, 2)) for d in detections],
                )

            # Stage G: Diagnostic HUD Visualization
            annotated = overlay.render(processed, result, detector.backend_name)
            should_continue = overlay.show(annotated)
            if not should_continue:
                logger.info("User requested termination (q/Escape).")
                break

            # Stage H: Interactive Key Event Dispatch
            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                paused = True
                logger.info("Perception loop paused. Press 'p' in window to resume.")
            elif key == ord("s"):
                save_dir.mkdir(exist_ok=True)
                snapshot_path = save_dir / f"frame_{frame_id:06d}.png"
                cv2.imwrite(str(snapshot_path), annotated)
                saved += 1
                logger.info("Diagnostic snapshot exported: %s", snapshot_path)

            frame_id += 1

            # Frame budget gate
            if args.max_frames > 0 and frame_id >= args.max_frames:
                logger.info("Processed maximum frame budget (max_frames=%d). Exiting.", args.max_frames)
                break

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user (Ctrl+C).")

    finally:
        # -------------------------------------------------------------------
        # 5. Hardware Teardown & Resource Cleanup
        # -------------------------------------------------------------------
        source.release()
        overlay.close()
        if json_file:
            json_file.close()
        logger.info(
            "Session terminated. Processed %d frames. Exported %d diagnostic snapshots.",
            frame_id, saved,
        )


# ===========================================================================
# Execution Entry Point
# ===========================================================================

if __name__ == "__main__":
    cli_args = parse_args()
    _setup_logging(cli_args.verbose)
    run(cli_args)
