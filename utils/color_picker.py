"""
utils/color_picker.py
──────────────────────
Venue-day color range extraction utility.

Use this on race day BEFORE the run to extract the HSV color range of the
actual buried mine markers. This takes ~2–5 minutes and outputs the exact
values to paste into config.yaml → buried_marker → marker_color_hsv_*.

WORKFLOW
--------
1. Take a photo of the real marker (or screenshot from your camera feed).
2. Run:
     python utils/color_picker.py --image path/to/marker_photo.jpg
   OR capture directly from webcam:
     python utils/color_picker.py --source 0

3. A window opens with the image. Use the HSV sliders to bracket the marker
   color until ONLY the marker pixels are visible in the mask preview.

4. Press 's' to save the extracted HSV range to a JSON file.
   Press 'c' to copy the config.yaml snippet to stdout.
   Press 'q' to quit.

5. Paste the output values into config.yaml and set use_color_gate: true.

ALTERNATIVE: Click mode
   Run with --click to use interactive click-sampling instead of sliders.
   Click 5+ points ON the marker in the image. The tool auto-computes an
   HSV bounding box around the sampled pixels with a configurable tolerance.

Author: Robofest 6.0 — Landmine Detection Team
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Slider-based HSV picker
# ---------------------------------------------------------------------------

class SliderPicker:
    """
    Interactive HSV range picker using OpenCV trackbars.
    Left panel: original image.
    Right panel: binary mask showing pixels matching current HSV range.
    """

    WINDOW = "HSV Color Picker — Landmine Marker Tuner"

    def __init__(self, frame: np.ndarray):
        self._orig   = frame
        self._hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self._result = None  # (low, high) tuples when user saves

        h, w = frame.shape[:2]
        # Cap display size so it fits on a laptop screen
        max_w = 600
        if w > max_w:
            scale = max_w / w
            dw, dh = max_w, int(h * scale)
            self._display = cv2.resize(frame, (dw, dh))
            self._hsv_d   = cv2.resize(self._hsv, (dw, dh))
        else:
            self._display = frame.copy()
            self._hsv_d   = self._hsv.copy()

    def run(self) -> Optional[Tuple[List[int], List[int]]]:
        """Run the picker loop. Returns (hsv_low, hsv_high) or None if cancelled."""
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)

        # Create 6 trackbars: H_low, S_low, V_low, H_high, S_high, V_high
        cv2.createTrackbar("H low",  self.WINDOW, 0,   179, lambda _: None)
        cv2.createTrackbar("S low",  self.WINDOW, 50,  255, lambda _: None)
        cv2.createTrackbar("V low",  self.WINDOW, 50,  255, lambda _: None)
        cv2.createTrackbar("H high", self.WINDOW, 179, 179, lambda _: None)
        cv2.createTrackbar("S high", self.WINDOW, 255, 255, lambda _: None)
        cv2.createTrackbar("V high", self.WINDOW, 255, 255, lambda _: None)

        print("\n[ColorPicker] Controls:")
        print("  Drag sliders → tune HSV range until ONLY the marker is visible in the mask")
        print("  s  → save / print config snippet")
        print("  q  → quit without saving\n")

        while True:
            h_lo = cv2.getTrackbarPos("H low",  self.WINDOW)
            s_lo = cv2.getTrackbarPos("S low",  self.WINDOW)
            v_lo = cv2.getTrackbarPos("V low",  self.WINDOW)
            h_hi = cv2.getTrackbarPos("H high", self.WINDOW)
            s_hi = cv2.getTrackbarPos("S high", self.WINDOW)
            v_hi = cv2.getTrackbarPos("V high", self.WINDOW)

            low  = np.array([h_lo, s_lo, v_lo],  dtype=np.uint8)
            high = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)

            mask = cv2.inRange(self._hsv_d, low, high)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Tint original image where mask is active
            tinted = self._display.copy()
            tinted[mask > 0] = (50, 255, 50)  # green highlight

            combined = np.hstack([tinted, mask_bgr])
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

            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                cv2.destroyAllWindows()
                return list(low.tolist()), list(high.tolist())
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return None


# ---------------------------------------------------------------------------
# Click-sampling mode
# ---------------------------------------------------------------------------

class ClickSampler:
    """
    Click on the marker pixels to auto-compute HSV range.
    Samples HSV values at each click point, then expands by a configurable
    tolerance to produce the final range.
    """

    WINDOW = "Click Sampler — Click on marker pixels, then press Enter"

    def __init__(self, frame: np.ndarray, tolerance: int = 25):
        self._orig      = frame
        self._hsv       = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self._tolerance = tolerance
        self._samples: List[Tuple[int, int, int]] = []

        h, w = frame.shape[:2]
        max_w = 800
        if w > max_w:
            scale = max_w / w
            self._dw, self._dh = max_w, int(h * scale)
            self._sx = w / self._dw
            self._sy = h / self._dh
        else:
            self._dw, self._dh = w, h
            self._sx = self._sy = 1.0

        self._display_orig = cv2.resize(frame, (self._dw, self._dh))
        self._display      = self._display_orig.copy()

    def _on_click(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map display coords back to original image coords
            ox = int(x * self._sx)
            oy = int(y * self._sy)
            ox = min(max(ox, 0), self._orig.shape[1] - 1)
            oy = min(max(oy, 0), self._orig.shape[0] - 1)
            h, s, v = self._hsv[oy, ox].tolist()
            self._samples.append((h, s, v))
            # Draw circle on display
            cv2.circle(self._display, (x, y), 5, (0, 255, 255), -1)
            cv2.imshow(self.WINDOW, self._display)
            print(f"  Sampled pixel ({ox},{oy}) → HSV ({h}, {s}, {v})")

    def run(self) -> Optional[Tuple[List[int], List[int]]]:
        print("\n[ClickSampler] Click on 5+ marker pixels in the image.")
        print("  Press Enter when done. Press 'q' to cancel.\n")

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
            key = cv2.waitKey(50) & 0xFF
            if key in (13, ord("\r"), ord("\n")):  # Enter
                break
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()

        if not self._samples:
            print("[ClickSampler] No samples collected.")
            return None

        samples = np.array(self._samples)
        h_vals, s_vals, v_vals = samples[:, 0], samples[:, 1], samples[:, 2]

        tol = self._tolerance
        low  = [max(0,   int(h_vals.min()) - tol),
                max(0,   int(s_vals.min()) - tol),
                max(0,   int(v_vals.min()) - tol)]
        high = [min(179, int(h_vals.max()) + tol),
                min(255, int(s_vals.max()) + tol),
                min(255, int(v_vals.max()) + tol)]

        return low, high


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_config_snippet(low: List[int], high: List[int]) -> None:
    """Print the config.yaml snippet to paste into buried_marker section."""
    print("\n" + "=" * 60)
    print("PASTE THIS INTO config.yaml → buried_marker section:")
    print("=" * 60)
    print(f"""
  use_color_gate: true
  marker_color_hsv_low:  {low}
  marker_color_hsv_high: {high}
  # Note: if marker color is RED (wraps around hue 0), also set:
  #   use_hue_wrap: true
  #   marker_color_hsv_low2:  [170, {low[1]}, {low[2]}]
  #   marker_color_hsv_high2: [179, {high[1]}, {high[2]}]
""")


def save_json(low: List[int], high: List[int], out_path: str) -> None:
    data = {
        "marker_color_hsv_low":  low,
        "marker_color_hsv_high": high,
        "extracted_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(out_path).write_text(json.dumps(data, indent=2))
    print(f"\n[ColorPicker] Saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Venue-day HSV color range picker for buried mine markers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--image",  metavar="PATH", help="Path to marker image")
    grp.add_argument("--source", metavar="INT",  type=int, help="Webcam device index")

    parser.add_argument(
        "--click", action="store_true",
        help="Use click-sampling mode instead of slider mode",
    )
    parser.add_argument(
        "--tolerance", type=int, default=25,
        help="HSV tolerance for click-sampling mode (default: 25)",
    )
    parser.add_argument(
        "--out", default="marker_hsv.json",
        help="Output JSON file for extracted range (default: marker_hsv.json)",
    )

    args = parser.parse_args()

    # Load frame
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[ERROR] Cannot read image: {args.image}", file=sys.stderr)
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(args.source)
        print(f"[ColorPicker] Capturing single frame from webcam {args.source}...")
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("[ERROR] Failed to capture from webcam.", file=sys.stderr)
            sys.exit(1)

    # Run picker
    if args.click:
        picker = ClickSampler(frame, tolerance=args.tolerance)
    else:
        picker = SliderPicker(frame)

    result = picker.run()
    if result is None:
        print("[ColorPicker] Cancelled.")
        sys.exit(0)

    low, high = result
    print_config_snippet(low, high)
    save_json(low, high, args.out)


if __name__ == "__main__":
    main()
