"""
main.py
────────
Entry point for the Landmine Detection Module.

Wires all components together and runs the real-time detection loop:
  CameraSource → FramePreprocessor → Detector → FrameResult → [Output] → DebugOverlay

Usage
-----
  # Default: webcam, debug overlay, config.yaml in current directory
  python main.py

  # Custom source (video file or RTSP URL)
  python main.py --source test_footage.mp4
  python main.py --source rtsp://192.168.1.10/stream

  # Custom config
  python main.py --config my_config.yaml

  # Headless mode (no display window — for deployment on Pi 5)
  python main.py --no-debug

  # Emit JSON detections to stdout (for future mapping module)
  python main.py --json

  # Write JSON to file
  python main.py --json --json-out detections.jsonl

Keyboard shortcuts (debug window):
  q / Escape  — quit
  p           — pause / resume
  s           — save current frame as PNG

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

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Module imports (after sys.path patch)
# ---------------------------------------------------------------------------
from capture.camera_source import build_source
from detection import build_detector
from output.detection_result import FrameResult
from preprocess.frame_prep import FramePreprocessor
from visualization.debug_overlay import DebugOverlay


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level  = level,
        format = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Load and return the YAML config as a plain dict."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            "Run from the project root directory, or pass --config <path>."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logging.getLogger(__name__).info("Loaded config: %s", config_path.resolve())
    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Landmine Detection Module — Robofest Gujarat 6.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source", default=None,
        help=(
            "Camera source override. "
            "Integer = webcam index, string = file path or rtsp:// URL. "
            "Overrides config.yaml camera.source if provided."
        ),
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML file (default: config.yaml in current directory).",
    )
    p.add_argument(
        "--no-debug", action="store_true",
        help="Disable debug visualization window (headless mode).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit JSON detection results to stdout (one line per frame).",
    )
    p.add_argument(
        "--json-out", default=None, metavar="FILE",
        help="Write JSON detection results to a JSONL file.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose DEBUG logging.",
    )
    p.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after N frames (0 = run indefinitely). Useful for batch testing.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    logger = logging.getLogger("main")

    # --- Load config ---
    cfg = load_config(args.config)

    # --- CLI overrides ---
    if args.source is not None:
        # Try to convert to int (webcam index) if possible
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

    # --- Build components ---
    logger.info("Initializing modules...")
    source      = build_source(cfg["camera"])
    preprocessor = FramePreprocessor(cfg["camera"], cfg.get("preprocessing", {}))
    detector    = build_detector(cfg)
    overlay     = DebugOverlay(cfg.get("debug", {}))

    # --- JSON output file handle ---
    json_file = None
    json_path = cfg.get("output", {}).get("json_output_file", "")
    emit_json = cfg.get("output", {}).get("emit_json", False)
    if json_path:
        json_file = open(json_path, "w", encoding="utf-8")
        logger.info("JSON output → %s", json_path)

    # --- Frame loop ---
    logger.info("Starting detection loop. Press 'q' in window or Ctrl+C to stop.")
    frame_id  = 0
    paused    = False
    saved     = 0
    save_dir  = PROJECT_ROOT / "saved_frames"

    try:
        while True:
            # Pause / resume via 'p' key (handled in overlay.show return value)
            if paused:
                time.sleep(0.05)
                # Still need to poll for keypress to un-pause
                key = cv2.waitKey(50) & 0xFF
                if key == ord("p"):
                    paused = False
                    logger.info("Resumed.")
                elif key in (ord("q"), 27):
                    break
                continue

            # --- Capture ---
            t_capture = time.monotonic()
            timestamp_ms = time.time() * 1000.0
            frame = source.read()
            if frame is None:
                logger.info("Source exhausted — stopping.")
                break

            # --- Preprocess ---
            processed, lb_info = preprocessor.process(frame)

            # --- Detect ---
            t_detect = time.monotonic()
            detections = detector.detect(processed, frame_id, timestamp_ms)
            t_done = time.monotonic()

            processing_ms = (t_done - t_capture) * 1000.0
            detect_ms     = (t_done - t_detect) * 1000.0

            # --- Assemble FrameResult ---
            result = FrameResult(
                frame_id           = frame_id,
                timestamp_ms       = timestamp_ms,
                detections         = detections,
                working_resolution = (lb_info.target_w, lb_info.target_h),
                scale_x            = lb_info.scale_x,
                scale_y            = lb_info.scale_y,
                processing_time_ms = processing_ms,
            )

            # --- JSON output ---
            if emit_json:
                line = json.dumps(result.to_dict())
                print(line, flush=True)
            if json_file:
                json_file.write(json.dumps(result.to_dict()) + "\n")
                json_file.flush()

            # --- Debug log (only when detections present) ---
            if detections:
                logger.debug(
                    "Frame %d: %d detection(s) in %.1f ms — %s",
                    frame_id,
                    len(detections),
                    detect_ms,
                    [(d.class_name, round(d.confidence, 2)) for d in detections],
                )

            # --- Visualization ---
            annotated = overlay.render(processed, result, detector.backend_name)

            # Handle key events from the overlay window
            should_continue = overlay.show(annotated)
            if not should_continue:
                logger.info("User quit (q/Escape).")
                break

            # Additional key handling via a secondary waitKey (already consumed above,
            # but we check a global key state via the overlay)
            # 'p' = pause, 's' = save frame
            # (These are detected via cv2.waitKey inside overlay.show)
            # We poll separately here for save / pause since overlay only checks q/Esc:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                paused = True
                logger.info("Paused. Press 'p' in window to resume.")
            elif key == ord("s"):
                save_dir.mkdir(exist_ok=True)
                fname = save_dir / f"frame_{frame_id:06d}.png"
                cv2.imwrite(str(fname), annotated)
                saved += 1
                logger.info("Saved frame: %s", fname)

            frame_id += 1

            # Max frames gate
            if args.max_frames > 0 and frame_id >= args.max_frames:
                logger.info("Reached max_frames=%d — stopping.", args.max_frames)
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")

    finally:
        # --- Cleanup ---
        source.release()
        overlay.close()
        if json_file:
            json_file.close()
        logger.info(
            "Session complete. Processed %d frames. Saved %d snapshots.",
            frame_id, saved,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    _setup_logging(args.verbose)
    run(args)
