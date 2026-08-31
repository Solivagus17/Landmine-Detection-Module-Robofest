<div align="center">

# 💣 Landmine Detection Module
### Robofest Gujarat 6.0 · Aerial Robotics · Minefield Navigation (Senior Division)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red?style=for-the-badge&logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Competition%20Ready-brightgreen?style=for-the-badge)]()

> **Real-time landmine & buried-mine-marker detection** for an autonomous aerial drone using RGB camera only.  
> Optimized for CPU-only inference on Raspberry Pi 5, achieving **20–50 ms/frame** with the classical pipeline.

</div>

---

## 🎯 Overview

This module is the **perception layer** of our autonomous drone system for Robofest Gujarat 6.0. It processes live aerial footage to detect:

- 🔴 **Surface Mines** — Oval/circular plate-shaped objects on the ground (see detection demo below)
- 🟡 **Buried Mine Markers** — Color-coded markers indicating subsurface mines

The system uses a **dual-backend architecture**: a classical computer vision pipeline (production-ready, zero model weight) and a YOLO inference stub (plug-in upgrade path when a trained model is available).

---

## 🖼️ Mine Detection Demo — Physical Plates as Mine Surrogates

> Tested on actual hardware using oval plates as mine stand-ins before the competition.

![Mine Detection using plates](assets/mine_detection_demo.png)

*Real-time `surface_mine` detections using the classical pipeline. Bounding boxes drawn around oval plate-shaped mine surrogates with confidence scores ranging **0.59 – 0.65**. Detection powered by Canny edge detection + shape geometry filtering.*

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Solivagus17/Landmine-Detection-Module-Robofest.git
cd Landmine-Detection-Module-Robofest

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run with default webcam
python main.py

# 4. Run with a video file
python main.py --source test_footage.mp4

# 5. Headless mode (no display — for Pi 5 deployment)
python main.py --no-debug

# 6. Emit JSON detections to stdout (for mapping module)
python main.py --json

# 7. Write JSON output to file
python main.py --json --json-out detections.jsonl
```

### Debug Window Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `p` | Pause / Resume |
| `s` | Save current annotated frame as PNG |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DETECTION PIPELINE                           │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Camera  │───▶│    Frame     │───▶│ Detector │───▶│  Frame   │  │
│  │  Source  │    │ Preprocessor │    │ Backend  │    │  Result  │  │
│  └──────────┘    └──────────────┘    └──────────┘    └──────────┘  │
│   capture/        preprocess/            │             output/      │
│ camera_source.py  frame_prep.py          │         detection_       │
│                                    ┌─────┴─────┐   result.py       │
│                                    │           │                    │
│                               Classical      YOLO                  │
│                               Pipeline      Detector               │
│                            (production)    (upgradeable)           │
│                                    │                               │
│                             ┌──────▼──────┐                        │
│                             │DebugOverlay │                         │
│                             │visualization│                         │
│                             └─────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | File | Responsibility |
|-----------|------|----------------|
| **Camera Source** | `capture/camera_source.py` | Webcam / video file / RTSP stream abstraction |
| **Frame Preprocessor** | `preprocess/frame_prep.py` | Letterbox resize + CLAHE enhancement + Gaussian blur |
| **Detector Factory** | `detection/__init__.py` | Backend selector (classical vs. YOLO) |
| **Candidate Proposal** | `detection/classical/candidate_proposal.py` | Canny edge detection + adaptive threshold → contours |
| **Shape Filter** | `detection/classical/shape_filter.py` | Area / solidity / aspect-ratio filtering + classifier |
| **Patch Classifier** | `detection/classical/patch_classifier.py` | CNN patch classifier (stub, upgradeable) |
| **YOLO Detector** | `detection/yolo/yolo_detector.py` | YOLOv8/NCNN inference (stub, upgradeable) |
| **Frame Result** | `output/detection_result.py` | `Detection` + `FrameResult` dataclasses |
| **Debug Overlay** | `visualization/debug_overlay.py` | Live annotated video feed renderer |
| **Color Picker** | `utils/color_picker.py` | Venue-day HSV color range extractor tool |

---

## 📁 Project Structure

```
Landmine-Detection-Module-Robofest/
│
├── 📄 config.yaml                      ← ALL tunable parameters (edit at venue!)
├── 📄 main.py                          ← Entry point / main detection loop
├── 📄 requirements.txt                 ← Python dependencies
├── 📄 test_integration.py              ← Integration test suite
│
├── 📁 assets/
│   └── mine_detection_demo.png         ← Real detection output with plate mines
│
├── 📁 capture/
│   └── camera_source.py               ← Source abstraction (webcam/file/RTSP)
│
├── 📁 preprocess/
│   └── frame_prep.py                  ← Letterbox resize + CLAHE + Gaussian blur
│
├── 📁 detection/
│   ├── __init__.py                    ← Detector factory
│   ├── classical/
│   │   ├── candidate_proposal.py      ← Canny + adaptive threshold → contours
│   │   ├── shape_filter.py            ← Geometry filter + mine classifier
│   │   └── patch_classifier.py        ← CNN patch classifier (stub)
│   └── yolo/
│       └── yolo_detector.py           ← YOLO inference (stub, plug-in ready)
│
├── 📁 output/
│   └── detection_result.py            ← Detection + FrameResult dataclasses
│
├── 📁 visualization/
│   └── debug_overlay.py               ← Live annotated feed overlay
│
└── 📁 utils/
    └── color_picker.py                ← Venue-day HSV color range extractor
```

---

## 🔬 Technical Deep-Dive — Classical Detection Pipeline

The production pipeline runs entirely on CPU, achieving **~20–50 ms/frame** on Raspberry Pi 5:

```
Raw Frame (e.g. 1280×720)
       │
       ▼
Letterbox Resize ──────────────────── 480×480 working resolution
       │                               (preserves aspect ratio with black padding)
       ▼
CLAHE Enhancement ─────────────────── clipLimit=2.0, tileGridSize=8×8
       │                               (boosts local contrast for outdoor conditions)
       ▼
Gaussian Blur ──────────────────────── noise suppression before edge detection
       │
       ├─────────────────────────────────────────────────┐
       ▼                                                 ▼
 Canny Edge Detection                        Adaptive Thresholding
 (primary candidate proposal)                (secondary / fallback)
       │                                                 │
       └───────────────────────┬─────────────────────────┘
                               ▼
                    Contour Extraction (findContours)
                               │
                               ▼
                    ┌──────────────────────┐
                    │   3-Stage Shape Filter│
                    │  1. Area gate         │ ← min/max pixel area
                    │  2. Solidity gate     │ ← convex hull ratio
                    │  3. Aspect ratio gate │ ← ellipse fit (oval mines)
                    └──────────────────────┘
                               │
                               ▼
                    Detection Objects → FrameResult → JSON / Overlay
```

### Detection Output Format (JSON)

Each frame produces a `FrameResult` — consumed directly by the mapping module:

```json
{
  "frame_id": 42,
  "timestamp_ms": 1787066047409.87,
  "working_resolution": [480, 480],
  "scale_x": 0.375,
  "scale_y": 0.375,
  "processing_time_ms": 28.4,
  "detections": [
    {
      "class_name": "surface_mine",
      "bbox": [120, 85, 68, 61],
      "center": [154, 115],
      "confidence": 0.72,
      "frame_id": 42,
      "timestamp_ms": 1787066047409.87,
      "detector_backend": "classical"
    }
  ]
}
```

> **Coordinate convention:** `bbox` is `[x, y, w, h]` in **working-resolution pixels** (480×480).  
> Map back to native pixels: `native_x = (bbox_x - pad_x) / scale_x`  
> `scale`, `pad_x`, `pad_y` are available in `LetterboxInfo` from `FramePreprocessor.process()`.

---

## ⚙️ Configuration Reference (`config.yaml`)

### Critical Tuning Knobs at Altitude

```yaml
surface_mine:
  min_area_px: 150     # ← Raise if tiny noise causes false positives
  max_area_px: 20000   # ← Lower if large ground features cause false positives
```

### Altitude → Pixel Area Reference Table

*(480×480 working resolution, ~90° FOV camera)*

| Altitude | Mine Diameter | Approx Area |
|----------|--------------|-------------|
| 0.5 m    | ~120 px      | ~11,000 px² |
| 1.0 m    | ~60 px       | ~2,800 px²  |
| 2.0 m    | ~30 px       | ~700 px²    |
| 3.0 m    | ~20 px       | ~310 px²    |
| 4.0 m    | ~15 px       | ~175 px²    |

### Marker Color Gate — Race-Day Tuning

Once real markers are visible at the venue, run the interactive color picker:

```bash
# Slider mode (recommended — interactive HSV sliders)
python utils/color_picker.py --image marker_photo.jpg

# Click mode — click on marker pixels, auto-computes HSV range
python utils/color_picker.py --image marker_photo.jpg --click

# Capture directly from drone camera
python utils/color_picker.py --source 0
```

Paste the output directly into `config.yaml`:

```yaml
buried_marker:
  use_color_gate: true
  marker_color_hsv_low:  [25, 100, 80]    # ← from color_picker output
  marker_color_hsv_high: [35, 255, 255]   # ← from color_picker output
```

---

## 🚀 YOLO Upgrade Path — Zero Code Changes

When a trained YOLO model becomes available, switch backends with **no code edits**:

**Step 1 — Export the model:**
```bash
yolo export model=best.pt format=onnx    # for Pi 5 with onnxruntime
# OR
yolo export model=best.pt format=ncnn    # for maximum speed on Pi 5
```

**Step 2 — Place model file:**
```
models/mine_detector.pt
```

**Step 3 — Update `config.yaml`:**
```yaml
detector_backend: "yolo"
yolo:
  enabled: true
  model_path: "models/mine_detector.pt"
```

**Step 4 — Run as normal.** No other changes needed.

### Dataset Collection Guidelines

| Class | Minimum | Recommended |
|-------|---------|-------------|
| Surface Mine | 150 images | 300 images |
| Buried Marker | 100 images | 200 images |

**Best practices:**
- Shoot at multiple altitudes: 0.5, 1, 2, 3 m
- Include varied lighting: morning, noon, overcast, shadow
- 20% zero-mine frames (hard negatives)
- Include partially occluded mines (leaves, soil edge)
- Label with **Roboflow** (free tier) → export as **YOLOv8 format**
- Enable augmentations: flip, HSV jitter ±20%, rotate ±15°, mosaic

---

## 🛡️ Risk Register & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Marker design unknown until race day | 🔴 HIGH | Color gate is OFF by default; `color_picker.py` extracts HSV range in 5 min |
| Altitude varies → apparent mine size changes | 🔴 HIGH | Tune `min/max_area_px` in config per flight; altitude table in this README |
| Outdoor lighting changes (shadows, glare) | 🟡 MEDIUM | CLAHE enabled; adaptive thresholding fallback |
| False positives from rocks/leaves | 🟡 MEDIUM | Solidity gate + area limits; adjust at venue |
| Partial occlusion by soil/shadow | 🟡 MEDIUM | Contour still fires on visible arc; `min_solidity` tolerates incomplete shapes |
| Pi 5 CPU latency | 🟢 LOW | 480×480 working res; classical pipeline ≈ 20–50 ms/frame |

---

## 🔗 Future System Integration

```
Landmine Detection Module (this repo)
         │
         │── JSON stream via --json flag
         ▼
    Mapping Module
    (geo-locate mine positions on arena map)
         │
         ▼
    Localization Module
    (pair timestamp_ms with drone telemetry)
         │
         ▼
    Swarm Coordination
    (multi-drone mine avoidance comms)
```

- **Mapping module:** Consume `FrameResult.to_dict()` JSON lines via `--json` mode.
- **Localization module:** Pair `timestamp_ms` with drone telemetry timestamps to geo-locate each detection.
- **Swarm coordination:** `FrameResult` is network-serializable (pure JSON-compatible types, no numpy in `.to_dict()`).

---

## 🧪 Running Tests

```bash
python test_integration.py
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥ 4.8.0 | Frame capture, image processing, display |
| `numpy` | ≥ 1.24.0 | Array operations |
| `PyYAML` | ≥ 6.0 | Configuration file loading |
| `ultralytics` *(optional)* | ≥ 8.0.0 | YOLO backend upgrade path |
| `onnxruntime` *(optional)* | ≥ 1.16.0 | CNN patch classifier upgrade path |

---

## 👥 Team

**Robofest Gujarat 6.0 — Landmine Detection Team**  
Senior Division · Aerial Robotics · Minefield Navigation

---

<div align="center">

*Built with ❤️ for Robofest Gujarat 6.0*

</div>
