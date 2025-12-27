"""
traffic_controller_backend.py

Backend-only traffic controller using YOLOv8 for vehicle detection.
- Accepts up to 4 sources (image paths or video paths). If none provided, uses webcam 0 for lane 1.
- Counts vehicles per lane (for images: single snapshot; for videos: continuous counting every second).
- Chooses green-light order based on highest-to-lowest traffic and assigns dynamic green durations.
- If a lane remains "full" while green, it will retain green until it clears (with a cap on extension).
- Displays a 2x2 (or single/side-by-side) OpenCV window showing detections, counts, and current signal state.

Usage examples:
    # 4 video files
    python traffic_controller_backend.py lane1.mp4 lane2.mp4 lane3.mp4 lane4.mp4

    # 4 images
    python traffic_controller_backend.py lane1.jpg lane2.jpg lane3.jpg lane4.jpg

    # mix of webcam (0) and files
    python traffic_controller_backend.py 0 lane2.mp4 lane3.jpg lane4.mp4

Notes:
    - Requires 'yolov8m.pt' to be present in cwd or change YOLO_WEIGHTS variable.
    - pip install ultralytics opencv-python-headless numpy
    - Press 'q' to quit, 'p' to pause/resume, 'n' to force next lane immediately.

Author: adapted for Rohit — preserves 4-lane green-light logic from Streamlit version but as backend.
"""

import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
import math
import os

YOLO_WEIGHTS = "yolov8m.pt"

# COCO vehicle class ids used by YOLO: car, motorcycle, bus, truck
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Timing parameters (seconds)
BASE_GREEN = 5          # minimum green time
PER_VEHICLE = 1.0       # extra seconds per vehicle counted initially
MAX_GREEN = 30          # maximum green time cap
EXTEND_CHECK_INTERVAL = 1.0  # seconds between checks while green
EXTEND_MAX_ADDITIONAL = 30   # max extra seconds to extend beyond initial green (cap)

# Detection interval for videos (seconds) — how often to recalc counts
DETECTION_INTERVAL = 1.0

# Small utility functions
def is_image_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def open_capture(source):
    if isinstance(source, str) and source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)

def read_frame(cap, is_image=False):
    if is_image:
        # cap is actually the file path for images (we'll return the loaded image)
        img = cv2.imread(cap)
        return img
    if not cap or not cap.isOpened():
        return None
    ret, frame = cap.read()
    if not ret:
        # try rewind for file-based video
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                return None
        except Exception:
            return None
    return frame

def preprocess(frame, target=(640,360)):
    if frame is None:
        return None
    return cv2.resize(frame, target)

def draw_annotations(frame, results, vehicle_ids=VEHICLE_CLASS_IDS):
    if frame is None:
        return None, 0
    annotated = frame.copy()
    count = 0
    for r in results:
        boxes = getattr(r, 'boxes', [])
        for box in boxes:
            try:
                cls_id = int(box.cls[0])
            except Exception:
                continue
            if cls_id in vehicle_ids:
                count += 1
                # draw box
                xyxy = box.xyxy[0].tolist()
                x1,y1,x2,y2 = map(int, xyxy)
                cv2.rectangle(annotated, (x1,y1),(x2,y2),(0,255,0),2)
                label = f"{vehicle_ids[cls_id]} {float(box.conf[0]):.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0,255,0), -1)
                cv2.putText(annotated, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return annotated, count

def combine_grid(frames, cols=2, rows=2):
    # ensure frames length == 4 by padding with black images
    valid = [f for f in frames if f is not None]
    if not valid:
        return None
    # target size: max width and height among valid frames
    widths = [f.shape[1] for f in valid]
    heights = [f.shape[0] for f in valid]
    tw, th = max(widths), max(heights)
    norm = []
    for f in frames:
        if f is None:
            norm.append(np.zeros((th, tw, 3), dtype=np.uint8))
        else:
            if f.shape[1] != tw or f.shape[0] != th:
                norm.append(cv2.resize(f, (tw, th)))
            else:
                norm.append(f)
    top = np.hstack((norm[0], norm[1]))
    bottom = np.hstack((norm[2], norm[3]))
    grid = np.vstack((top, bottom))
    return grid

def compute_initial_green_time(count):
    t = BASE_GREEN + PER_VEHICLE * count
    t = max(t, BASE_GREEN)
    return min(t, MAX_GREEN)

def overlay_signal_info(frame, counts, current_idx, remaining_time):
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    # overlay small panels for each quadrant
    # draw at top-left of each tile
    tile_h = h // 2
    tile_w = w // 2
    for i in range(4):
        gx = (i % 2) * tile_w
        gy = (i // 2) * tile_h
        # panel background
        cv2.rectangle(frame, (gx+5, gy+5), (gx+220, gy+55), (20,20,20), -1)
        status = "GREEN" if i == current_idx else "RED"
        color = (0,180,0) if i == current_idx else (0,0,180)
        cv2.putText(frame, f"Lane {i+1}: {status}", (gx+10, gy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Count: {counts[i]}", (gx+10, gy+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
    # central timer
    cv2.rectangle(frame, (w-240, 10), (w-10, 70), (30,30,30), -1)
    cv2.putText(frame, f"Current Lane: {current_idx+1 if current_idx is not None else '-'}", (w-230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Time left: {int(remaining_time)}s", (w-230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
    return frame

def load_model(weights=YOLO_WEIGHTS):
    print("Loading YOLO model (this may download weights if missing)...")
    model = YOLO(weights)
    print("Model loaded.")
    return model

def detect_count_for_frame(model, frame):
    if frame is None:
        return 0, None
    small = preprocess(frame, target=(640,360))
    results = model(small)
    annotated, count = draw_annotations(small, results)
    return count, annotated

def is_full_lane(count, other_counts, threshold_ratio=0.6, absolute_min=3):
    # returns True if this lane is significantly fuller than others or above absolute threshold
    if count >= absolute_min and all(count >= (oc * (threshold_ratio)) for oc in other_counts):
        return True
    return False

def main(sources):
    # normalize to 4 sources by padding with None
    sources = sources[:4] + [None] * max(0, 4 - len(sources))
    is_image = [bool(is_image_file(s)) if s is not None else False for s in sources]
    caps = [None]*4
    for i, s in enumerate(sources):
        if s is None:
            caps[i] = None
        elif is_image[i]:
            caps[i] = s  # we'll read image directly
        else:
            caps[i] = open_capture(s)

    model = load_model()

    # initial counts (for images do one-shot; for videos we'll update continuously)
    counts = [0]*4
    annotated_frames = [None]*4

    last_detection_time = 0.0
    paused = False
    current_lane = None
    remaining_time = 0.0
    # order will be a rotating queue updated after each green cycle
    order = list(range(4))

    # initial detection to seed counts
    for i in range(4):
        if sources[i] is None:
            counts[i] = 0
            annotated_frames[i] = np.zeros((360,640,3), dtype=np.uint8)
            cv2.putText(annotated_frames[i], f"Lane {i+1}: No source", (20,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        elif is_image[i]:
            frame = read_frame(sources[i], is_image=True)
            c, ann = detect_count_for_frame(model, frame)
            counts[i] = c
            annotated_frames[i] = ann if ann is not None else np.zeros((360,640,3), dtype=np.uint8)
        else:
            # read one frame to seed
            frame = read_frame(caps[i], is_image=False)
            c, ann = detect_count_for_frame(model, frame)
            counts[i] = c
            annotated_frames[i] = ann if ann is not None else np.zeros((360,640,3), dtype=np.uint8)

    print("Initial counts per lane:", counts)
    # determine initial order by descending counts (highest first)
    order = sorted(range(4), key=lambda x: counts[x], reverse=True)

    # main control loop
    last_update = time.time()
    while True:
        now = time.time()

        # update detections periodically for video feeds
        if not paused and (now - last_detection_time >= DETECTION_INTERVAL):
            for i in range(4):
                if sources[i] is None:
                    counts[i] = 0
                    continue
                if is_image[i]:
                    # images don't change
                    continue
                frame = read_frame(caps[i], is_image=False)
                c, ann = detect_count_for_frame(model, frame)
                counts[i] = c
                annotated_frames[i] = ann if ann is not None else annotated_frames[i]
            last_detection_time = now

        # If no current lane (begin cycle), pick top in order that's not empty if possible
        if current_lane is None and not paused:
            # recompute order by latest counts descending
            order = sorted(range(4), key=lambda x: counts[x], reverse=True)
            # pick first non-empty; if all zero, pick first in order
            picked = None
            for idx in order:
                if counts[idx] > 0:
                    picked = idx
                    break
            if picked is None:
                picked = order[0]
            current_lane = picked
            # compute initial green time based on current count snapshot
            initial_count = counts[current_lane]
            green_time = compute_initial_green_time(initial_count)
            remaining_time = green_time
            extension_used = 0.0
            last_extend_check = now
            print(f"Starting green for Lane {current_lane+1} | count={initial_count} | green_time={green_time}s")

        # Build display grid and overlay signals
        # ensure annotated_frames have images (use black tiles for missing)
        frames_for_grid = []
        for i in range(4):
            if annotated_frames[i] is None:
                annotated_frames[i] = np.zeros((360,640,3), dtype=np.uint8)
                cv2.putText(annotated_frames[i], f"Lane {i+1}: No frame", (20,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)

            # draw count on each tile
            tile = annotated_frames[i].copy()
            cv2.rectangle(tile, (5,5), (180,45), (20,20,20), -1)
            cv2.putText(tile, f"Count: {counts[i]}", (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
            frames_for_grid.append(tile)

        out = combine_grid(frames_for_grid, cols=2, rows=2)
        if out is None:
            out = np.zeros((720,1280,3), dtype=np.uint8)

        # Overlay signals and timer
        out = overlay_signal_info(out, counts, current_lane if current_lane is not None else -1, remaining_time)

        # Display
        cv2.imshow("Traffic Controller Backend (q to quit, p pause, n next)", out)

        # handle green time countdown and extension logic
        if current_lane is not None and not paused:
            elapsed = now - last_update
            last_update = now
            remaining_time -= elapsed

            # periodically (EXTEND_CHECK_INTERVAL) check if lane should be extended
            if now - last_extend_check >= EXTEND_CHECK_INTERVAL:
                last_extend_check = now
                # re-evaluate count for current lane (if video)
                if not is_image[current_lane] and sources[current_lane] is not None:
                    frame = read_frame(caps[current_lane], is_image=False)
                    c_cur, ann = detect_count_for_frame(model, frame)
                    counts[current_lane] = c_cur
                    if ann is not None:
                        annotated_frames[current_lane] = ann
                # check if lane is 'full' compared to others or above min threshold
                others = [counts[j] for j in range(4) if j != current_lane]
                if is_full_lane(counts[current_lane], others):
                    # extend remaining time by up to EXTEND_CHECK_INTERVAL each check, capped
                    if extension_used < EXTEND_MAX_ADDITIONAL:
                        remaining_time += EXTEND_CHECK_INTERVAL
                        extension_used += EXTEND_CHECK_INTERVAL
                        # clamp to MAX_GREEN + EXTEND_MAX_ADDITIONAL (hard cap to avoid infinite)
                        max_allowed = MAX_GREEN + EXTEND_MAX_ADDITIONAL
                        if remaining_time > max_allowed:
                            remaining_time = max_allowed

            # if time up, move to next lane in order
            if remaining_time <= 0:
                # determine next lane: pick next in order that's not current; if all zero, pick next cyclically
                # recompute order by latest counts
                order = sorted(range(4), key=lambda x: counts[x], reverse=True)
                # find next different lane
                try:
                    pos = order.index(current_lane)
                    # next position
                    next_pos = (pos + 1) % 4
                    next_lane = order[next_pos]
                except ValueError:
                    # fallback: first in order that's not current
                    next_lane = order[0] if order[0] != current_lane else order[1]
                print(f"Switching: Lane {current_lane+1} -> Lane {next_lane+1} | counts: {counts}")
                current_lane = next_lane
                initial_count = counts[current_lane]
                green_time = compute_initial_green_time(initial_count)
                remaining_time = green_time
                extension_used = 0.0
                last_extend_check = now

        # key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        if key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        if key == ord('n'):
            # force next lane immediately
            if current_lane is not None:
                order = sorted(range(4), key=lambda x: counts[x], reverse=True)
                try:
                    pos = order.index(current_lane)
                    next_lane = order[(pos+1)%4]
                except ValueError:
                    next_lane = order[0]
                print(f"Forced next: {current_lane+1} -> {next_lane+1}")
                current_lane = next_lane
                initial_count = counts[current_lane]
                green_time = compute_initial_green_time(initial_count)
                remaining_time = green_time
                extension_used = 0.0
                last_extend_check = now

    # cleanup
    for i in range(4):
        if caps[i] is not None and not is_image[i]:
            caps[i].release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = sys.argv[1:]
    # if no args, default to webcam 0 for lane 1 and others empty
    if len(args) == 0:
        sources = ["0"]
    else:
        sources = args[:4]
    # pad to 4 for internal handling
    while len(sources) < 4:
        sources.append(None)
    main(sources)
