"""
utils/color_picker.py
──────────────────────
Race-Day Buried Mine Marker HSV Color Manifold Calibration Utility.

Robofest Gujarat 6.0 | Aerial Robotics | Senior Division

Provides an interactive utility for extracting the HSV color distribution of real
buried mine markers at the competition arena prior to flight runs:

Operational Modes:
  1. Interactive Trackbar Mode (Default):
     Provides 6 independent OpenCV trackbars (H_low, S_low, V_low, H_high, S_high, V_high)
     with a real-time side-by-side binary mask and green tint overlay.
  2. Click-Sampling Mode (`--click`):
     Allows operators to click directly on 5+ marker pixels within the frame. Automatically
     computes the empirical bounding manifold across sampled pixels with a user-configurable
     tolerance window (`--tolerance 25`).

Workflow:
  1. Capture a still photo or acquire a live frame from the drone camera:
       python utils/color_picker.py --image marker_sample.jpg
       python utils/color_picker.py --source 0
  2. Adjust sliders or click marker pixels until the mask isolates the marker target.
  3. Press 's' to export the validated HSV manifold boundaries to `marker_hsv.json`
     and print the exact snippet ready for insertion into `config.yaml`.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ===========================================================================
# Interactive Trackbar HSV Picker
# ===========================================================================

class SliderPicker:
    """
    Interactive HSV color manifold calibration window utilizing OpenCV trackbars.

    Parameters
    ----------
    frame : np.ndarray
        Raw input image containing the marker target.
    """

    WINDOW: str = "HSV Color Picker — Landmine Marker Tuner"

    def __init__(self, frame: np.ndarray) -> None:
        self._orig: np.ndarray = frame
        self._hsv: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self._result: Optional[Tuple[List[int], List[int]]] = None

        h, w = frame.shape[:2]
        max_w: int = 600
        if w > max_w:
            scale: float = max_w / w
            dw, dh = max_w, int(h * scale)
            self._display: np.ndarray = cv2.resize(frame, (dw, dh))
            self._hsv_d: np.ndarray = cv2.resize(self._hsv, (dw, dh))
        else:
            self._display = frame.copy()
            self._hsv_d = self._hsv.copy()

    def run(self) -> Optional[Tuple[List[int], List[int]]]:
        """
        Execute trackbar calibration event loop.

        Returns
        -------
        Optional[Tuple[List[int], List[int]]]
            ([H_low, S_low, V_low], [H_high, S_high, V_high]) or None if cancelled.
        """
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)

        # Instantiate 6 calibration trackbars
        cv2.createTrackbar("H low",  self.WINDOW, 0,   179, lambda _: None)
        cv2.createTrackbar("S low",  self.WINDOW, 50,  255, lambda _: None)
        cv2.createTrackbar("V low",  self.WINDOW, 50,  255, lambda _: None)
        cv2.createTrackbar("H high", self.WINDOW, 179, 179, lambda _: None)
        cv2.createTrackbar("S high", self.WINDOW, 255, 255, lambda _: None)
        cv2.createTrackbar("V high", self.WINDOW, 255, 255, lambda _: None)

        print("\n[ColorPicker] Interactive Controls:")
        print("  Adjust trackbars -> Bracket HSV manifold until only marker pixels appear in mask")
        print("  s               -> Save HSV manifold boundaries and generate config snippet")
        print("  q / Esc         -> Terminate utility without saving\n")

        while True:
            h_lo: int = cv2.getTrackbarPos("H low",  self.WINDOW)
            s_lo: int = cv2.getTrackbarPos("S low",  self.WINDOW)
            v_lo: int = cv2.getTrackbarPos("V low",  self.WINDOW)
            h_hi: int = cv2.getTrackbarPos("H high", self.WINDOW)
            s_hi: int = cv2.getTrackbarPos("S high", self.WINDOW)
            v_hi: int = cv2.getTrackbarPos("V high", self.WINDOW)

            low: np.ndarray = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
            high: np.ndarray = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)

            mask: np.ndarray = cv2.inRange(self._hsv_d, low, high)
            mask_bgr: np.ndarray = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Apply green highlight overlay on original image
            tinted: np.ndarray = self._display.copy()
            tinted[mask > 0] = (50, 255, 50)

            combined: np.ndarray = np.hstack([tinted, mask_bgr])
            cv2.putText(
                combined,
                f"H:[{h_lo}-{h_hi}]  S:[{s_lo}-{s_hi}]  V:[{v_lo}-{v_hi}]",
                (10, combined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 80), 1, cv2.LINE_AA,
            )
            cv2.putText(
                combined, "Press 's' to save, 'q' to quit",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (200, 200, 200), 1, cv2.LINE_AA,
            )
            cv2.imshow(self.WINDOW, combined)

            key: int = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                cv2.destroyAllWindows()
                return list(low.tolist()), list(high.tolist())
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return None


# ===========================================================================
# Interactive Click-Sampling HSV Picker
# ===========================================================================

class ClickSampler:
    """
    Statistical HSV manifold calibration via direct pixel mouse click sampling.

    Parameters
    ----------
    frame : np.ndarray
        Input image containing target marker.
    tolerance : int, default=25
        Expansion margin applied around empirical sample minimum/maximum values.
    """

    WINDOW: str = "Click Sampler — Click on marker pixels, then press Enter"

    def __init__(self, frame: np.ndarray, tolerance: int = 25) -> None:
        self._orig: np.ndarray = frame
        self._hsv: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self._tolerance: int = tolerance
        self._samples: List[Tuple[int, int, int]] = []

        h, w = frame.shape[:2]
        max_w: int = 800
        if w > max_w:
            scale: float = max_w / w
            self._dw, self._dh = max_w, int(h * scale)
            self._sx: float = w / self._dw
            self._sy: float = h / self._dh
        else:
            self._dw, self._dh = w, h
            self._sx = self._sy = 1.0

        self._display_orig: np.ndarray = cv2.resize(frame, (self._dw, self._dh))
        self._display: np.ndarray = self._display_orig.copy()

    def _on_click(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Mouse callback capturing pixel values at cursor coordinates."""
        if event == cv2.EVENT_LBUTTONDOWN:
            ox: int = int(x * self._sx)
            oy: int = int(y * self._sy)
            ox = min(max(ox, 0), self._orig.shape[1] - 1)
            oy = min(max(oy, 0), self._orig.shape[0] - 1)
            h, s, v = self._hsv[oy, ox].tolist()
            self._samples.append((h, s, v))
            cv2.circle(self._display, (x, y), 5, (0, 255, 255), -1)
            cv2.imshow(self.WINDOW, self._display)
            print(f"  Sampled pixel ({ox}, {oy}) -> HSV ({h}, {s}, {v})")

    def run(self) -> Optional[Tuple[List[int], List[int]]]:
        """
        Execute interactive click sampling loop.

        Returns
        -------
        Optional[Tuple[List[int], List[int]]]
            ([H_low, S_low, V_low], [H_high, S_high, V_high]) or None if cancelled.
        """
        print("\n[ClickSampler] Click on 5+ representative marker pixels in the window.")
        print("  Press Enter to calculate manifold boundaries. Press 'q' to cancel.\n")

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW, self._on_click)
        cv2.putText(
            self._display,
            "Click marker pixels. Enter to confirm, q to quit.",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (200, 200, 200), 1, cv2.LINE_AA,
        )
        cv2.imshow(self.WINDOW, self._display)

        while True:
            key: int = cv2.waitKey(50) & 0xFF
            if key in (13, ord("\r"), ord("\n")):  # Enter
                break
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()

        if not self._samples:
            print("[ClickSampler] No pixel samples recorded.")
            return None

        samples_arr: np.ndarray = np.array(self._samples)
        h_vals, s_vals, v_vals = samples_arr[:, 0], samples_arr[:, 1], samples_arr[:, 2]

        tol: int = self._tolerance
        low: List[int] = [
            max(0, int(h_vals.min()) - tol),
            max(0, int(s_vals.min()) - tol),
            max(0, int(v_vals.min()) - tol),
        ]
        high: List[int] = [
            min(179, int(h_vals.max()) + tol),
            min(255, int(s_vals.max()) + tol),
            min(255, int(v_vals.max()) + tol),
        ]

        return low, high


# ===========================================================================
# Output & Configuration Helpers
# ===========================================================================

def print_config_snippet(low: List[int], high: List[int]) -> None:
    """Print formatted YAML configuration snippet to stdout."""
    print("\n" + "=" * 70)
    print("COPY AND PASTE INTO config.yaml -> buried_marker section:")
    print("=" * 70)
    print(f"""
  use_color_gate: true
  marker_color_hsv_low:  {low}
  marker_color_hsv_high: {high}
  # Note: If marker pigment is red (wraps around 0 deg hue), also configure:
  #   use_hue_wrap: true
  #   marker_color_hsv_low2:  [170, {low[1]}, {low[2]}]
  #   marker_color_hsv_high2: [179, {high[1]}, {high[2]}]
""")


def save_json(low: List[int], high: List[int], out_path: str) -> None:
    """Save extracted HSV manifold bounds to a JSON file."""
    data: Dict[str, Any] = {
        "marker_color_hsv_low":  low,
        "marker_color_hsv_high": high,
        "extracted_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[ColorPicker] Manifold parameters saved to: {out_path}")


# ===========================================================================
# CLI Driver Entry Point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Race-Day HSV Color Manifold Calibration Utility for Buried Mine Markers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", metavar="PATH", help="Filesystem path to sample marker image")
    group.add_argument("--source", metavar="INT", type=int, help="V4L2 camera device index (e.g. 0)")

    parser.add_argument(
        "--click", action="store_true",
        help="Activate statistical click-sampling mode instead of trackbars",
    )
    parser.add_argument(
        "--tolerance", type=int, default=25,
        help="HSV tolerance margin for click-sampling mode (default: 25)",
    )
    parser.add_argument(
        "--out", default="marker_hsv.json",
        help="Output JSON file destination (default: marker_hsv.json)",
    )

    args = parser.parse_args()

    # Ingest target frame
    if args.image:
        frame: Optional[np.ndarray] = cv2.imread(args.image)
        if frame is None:
            print(f"[ERROR] Failed to read image from path: {args.image}", file=sys.stderr)
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(args.source)
        print(f"[ColorPicker] Acquiring test frame from camera node {args.source}...")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print("[ERROR] Failed to acquire video frame from capture device.", file=sys.stderr)
            sys.exit(1)

    # Launch designated picker mode
    if args.click:
        picker = ClickSampler(frame, tolerance=args.tolerance)
    else:
        picker = SliderPicker(frame)

    result = picker.run()
    if result is None:
        print("[ColorPicker] Calibration cancelled.")
        sys.exit(0)

    low, high = result
    print_config_snippet(low, high)
    save_json(low, high, args.out)


if __name__ == "__main__":
    main()
