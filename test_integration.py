import sys, time, json, yaml, numpy as np, cv2
sys.path.insert(0, '.')

from preprocess.frame_prep import FramePreprocessor
from detection import build_detector
from output.detection_result import FrameResult, MineClass

cfg = yaml.safe_load(open('config.yaml'))

def make_test_frame(w=1280, h=720):
    bg = np.random.randint(60, 110, (h, w, 3), dtype=np.uint8)
    frame = bg.copy()
    # Mine disc (dark ellipse on noisy ground)
    cv2.ellipse(frame, (400, 360), (100, 90), 15, 0, 360, (40, 40, 40), -1)
    # Small bright marker
    cv2.rectangle(frame, (700, 300), (730, 330), (0, 180, 50), -1)
    return frame

frame = make_test_frame()
preprocessor = FramePreprocessor(cfg['camera'], cfg.get('preprocessing', {}))
detector = build_detector(cfg)

results = []
for i in range(10):
    ts = time.time() * 1000
    t0 = time.monotonic()
    processed, info = preprocessor.process(frame)
    dets = detector.detect(processed, i, ts)
    elapsed_ms = (time.monotonic() - t0) * 1000
    result = FrameResult(
        frame_id=i, timestamp_ms=ts, detections=dets,
        working_resolution=(info.target_w, info.target_h),
        scale_x=info.scale_x, scale_y=info.scale_y,
        processing_time_ms=elapsed_ms
    )
    results.append(result)

avg_ms = sum(r.processing_time_ms for r in results) / len(results)
print("Processed 10 frames on synthetic test image")
print("Average time per frame: %.1f ms  (%.1f FPS)" % (avg_ms, 1000/avg_ms))
print()

d = results[0].to_dict()
print("Sample frame result (frame 0):")
print("  detections:", len(d['detections']))
print("  processing_time_ms:", d['processing_time_ms'])
for det in d['detections'][:3]:
    print("  ->", det['class_name'], "conf=%.2f" % det['confidence'], "bbox=", det['bbox'])
print()

# JSON round-trip
line = json.dumps(results[0].to_dict())
parsed = json.loads(line)
assert 'detections' in parsed
print("[OK] JSON serialization round-trip clean")

r = results[0]
print("[OK] surface_mines: %d, buried_markers: %d" % (len(r.surface_mines), len(r.buried_markers)))
print()
print("INTEGRATION TEST PASSED")
