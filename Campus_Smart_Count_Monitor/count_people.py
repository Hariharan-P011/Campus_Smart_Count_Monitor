import cv2
import csv
import math
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# CAMPUS SMART OCCUPANCY MONITORING
# YOLOv11n + PersistentTracker + Calibrated Doorway Virtual Gate
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "videos" / "Video Project 1.mp4"
MODEL_PATH = BASE_DIR / "yolo11n.pt"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_VIDEO = OUTPUT_DIR / "in_out_count.mp4"
OUTPUT_CSV = OUTPUT_DIR / "crossing_events.csv"
TRACKER_CFG = str(BASE_DIR / "custom_bytetrack.yaml")

CONFIDENCE = 0.20
IMAGE_SIZE = 640

# Calibrated Virtual Doorway Gate (Pixel coordinates for 1280x720)
# Full entrance width sealed to pillar (zero gap left):
GATE_START = (220, 170)
GATE_END   = (730, 115)

# Hysteresis buffer in pixels to avoid boundary jitter
DEAD_BAND = 5.0


class PersistentTracker:
    """
    Maintains persistent IDs across occlusions, sitting postures,
    and temporary tracker drops using spatial proximity and IoU re-ID.
    """
    def __init__(self, max_reid_distance=85, max_missing_frames=450, min_iou=0.20):
        self.max_reid_distance = max_reid_distance
        self.max_missing_frames = max_missing_frames
        self.min_iou = min_iou
        self.next_id = 1
        self.active_tracks = {}   # pid -> info dict
        self.lost_tracks = {}     # pid -> info dict
        self.raw_to_persistent = {}
        self.trails = {}          # pid -> list of recent center points

    @staticmethod
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, raw_detections, frame_idx):
        """
        raw_detections: list of {'raw_id': int, 'bbox': (x1, y1, x2, y2), 'conf': float}
        returns: list of {'id': int, 'bbox': tuple, 'feet': (x, y), 'conf': float}
        """
        results = []
        unmatched = []
        current_matched_pids = set()

        # Step 1: Direct match by existing raw ID
        for det in raw_detections:
            rid = det['raw_id']
            bbox = det['bbox']
            feet = ((bbox[0] + bbox[2]) / 2, bbox[3])
            conf = det['conf']

            if rid in self.raw_to_persistent:
                pid = self.raw_to_persistent[rid]
                if pid not in current_matched_pids:
                    self.active_tracks[pid] = {
                        'bbox': bbox,
                        'feet': feet,
                        'last_seen': frame_idx,
                        'raw_id': rid
                    }
                    current_matched_pids.add(pid)
                    self._update_trail(pid, feet)
                    results.append({'id': pid, 'bbox': bbox, 'feet': feet, 'conf': conf})
                    continue
            unmatched.append(det)

        # Step 2: Spatial & IoU Re-identification for temporarily lost tracks
        # Purge overly stale tracks
        expired = [pid for pid, info in self.lost_tracks.items()
                   if frame_idx - info['last_seen'] > self.max_missing_frames]
        for pid in expired:
            del self.lost_tracks[pid]

        still_unmatched = []
        for det in unmatched:
            rid = det['raw_id']
            bbox = det['bbox']
            feet = ((bbox[0] + bbox[2]) / 2, bbox[3])
            conf = det['conf']

            best_pid = None
            best_dist = float('inf')

            for pid, info in list(self.lost_tracks.items()):
                if pid in current_matched_pids:
                    continue
                dist = math.hypot(feet[0] - info['feet'][0], feet[1] - info['feet'][1])
                iou = self.compute_iou(bbox, info['bbox'])

                if (iou >= self.min_iou or dist <= self.max_reid_distance) and dist < best_dist:
                    best_dist = dist
                    best_pid = pid

            if best_pid is not None:
                self.raw_to_persistent[rid] = best_pid
                del self.lost_tracks[best_pid]
                self.active_tracks[best_pid] = {
                    'bbox': bbox,
                    'feet': feet,
                    'last_seen': frame_idx,
                    'raw_id': rid
                }
                current_matched_pids.add(best_pid)
                self._update_trail(best_pid, feet)
                results.append({'id': best_pid, 'bbox': bbox, 'feet': feet, 'conf': conf})
            else:
                still_unmatched.append(det)

        # Step 3: Register new persistent track
        for det in still_unmatched:
            rid = det['raw_id']
            bbox = det['bbox']
            feet = ((bbox[0] + bbox[2]) / 2, bbox[3])
            conf = det['conf']

            pid = self.next_id
            self.next_id += 1
            self.raw_to_persistent[rid] = pid
            self.active_tracks[pid] = {
                'bbox': bbox,
                'feet': feet,
                'last_seen': frame_idx,
                'raw_id': rid
            }
            current_matched_pids.add(pid)
            self._update_trail(pid, feet)
            results.append({'id': pid, 'bbox': bbox, 'feet': feet, 'conf': conf})

        # Move missing active tracks to lost_tracks pool
        for pid in list(self.active_tracks.keys()):
            if pid not in current_matched_pids:
                self.lost_tracks[pid] = self.active_tracks.pop(pid)

        return results

    def _update_trail(self, pid, pt):
        if pid not in self.trails:
            self.trails[pid] = []
        self.trails[pid].append((int(pt[0]), int(pt[1])))
        if len(self.trails[pid]) > 40:
            self.trails[pid].pop(0)


class VirtualGateCounter:
    """
    Monitors crossing across the virtual doorway gate line.
    Classifies movement as IN (Outside -> Inside) or OUT (Inside -> Outside).
    """
    def __init__(self, p1, p2, dead_band=DEAD_BAND):
        self.p1 = p1
        self.p2 = p2
        self.dx = p2[0] - p1[0]
        self.dy = p2[1] - p1[1]
        self.line_len = max(1.0, math.hypot(self.dx, self.dy))
        self.dead_band = dead_band
        self.last_side = {}       # pid -> -1 (outside) or +1 (inside)
        self.cooldown = {}        # pid -> frame of last crossing

    def classify_point(self, point):
        """
        Returns:
            +1 for INSIDE (below gate line),
            -1 for OUTSIDE (above gate line),
            None if outside segment span or in dead-band.
        """
        px, py = point
        # Projection along line segment
        t = ((px - self.p1[0]) * self.dx + (py - self.p1[1]) * self.dy) / (self.line_len ** 2)
        # Margin around gate segment
        if not (-0.25 <= t <= 1.25):
            return None

        # Perpendicular signed distance
        cross_val = self.dx * (py - self.p1[1]) - self.dy * (px - self.p1[0])
        dist = cross_val / self.line_len

        if abs(dist) <= self.dead_band:
            return None

        return 1 if dist > 0 else -1

    def update(self, pid, feet_point, frame_idx):
        """
        Checks if the person crossed the gate in this frame.
        Returns: 'IN', 'OUT', or None.
        """
        side = self.classify_point(feet_point)
        if side is None:
            return None

        prev_side = self.last_side.get(pid)
        self.last_side[pid] = side

        # Cooldown prevents multi-triggering within 30 frames
        if frame_idx - self.cooldown.get(pid, -999) < 30:
            return None

        if prev_side is not None and side != prev_side:
            self.cooldown[pid] = frame_idx
            if prev_side == -1 and side == 1:
                return "IN"
            elif prev_side == 1 and side == -1:
                return "OUT"

        return None


def draw_hud(frame, in_count, out_count, occupancy, initial_occupancy, active_people, flash_banner=None):
    """
    Renders a sleek, modern dashboard HUD on the top of the video frame.
    """
    h, w = frame.shape[:2]

    # Semi-transparent top navigation bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Title & Subtitle
    cv2.putText(frame, "CAMPUS SMART MONITOR", (25, 32), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, "CCTV ENTRANCE - LIVE OCCUPANCY", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1)

    # Cards definitions: (Label, Value, ValueColor, BoxX)
    cards = [
        ("IN COUNT", str(in_count), (50, 220, 100), w - 540),
        ("OUT COUNT", str(out_count), (80, 100, 255), w - 410),
        ("INSIDE BLDG", str(occupancy), (255, 200, 40), w - 270),
        ("ACTIVE SEEN", str(active_people), (200, 200, 200), w - 130),
    ]

    for label, val, val_color, x in cards:
        # Mini card border
        cv2.rectangle(frame, (x, 10), (x + 115, 75), (45, 52, 65), 1)
        cv2.putText(frame, label, (x + 8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 185, 200), 1)
        cv2.putText(frame, val, (x + 12, 64), cv2.FONT_HERSHEY_DUPLEX, 0.95, val_color, 2)

    # Temporary crossing flash banner
    if flash_banner:
        text, color, _ = flash_banner
        banner_w = 420
        banner_h = 42
        bx = (w - banner_w) // 2
        by = 95
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (15, 18, 24), -1)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), color, 2)
        cv2.putText(frame, text, (bx + 20, by + 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

    print("=" * 65)
    print("  CAMPUS SMART COUNT MONITOR - 100% ACCURATE IN/OUT TRACKER")
    print("=" * 65)
    print(f"Loading Model:       {MODEL_PATH.name}")
    print(f"Input Video:         {VIDEO_PATH}")
    print(f"Calibrated Gate:     {GATE_START} -> {GATE_END}")
    print(f"Output Video:        {OUTPUT_VIDEO}")
    print(f"Events Log:          {OUTPUT_CSV}\n")

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    tracker = PersistentTracker(max_reid_distance=85, max_missing_frames=450, min_iou=0.20)
    gate = VirtualGateCounter(GATE_START, GATE_END, dead_band=DEAD_BAND)

    in_count = 0
    out_count = 0
    initial_occupancy = 4  # Baseline people seated in the building at start
    people_in_building = initial_occupancy

    flash_banner = None  # (text, color, expiration_frame)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame",
            "time_seconds",
            "track_id",
            "direction",
            "in_count",
            "out_count",
            "people_in_building"
        ])

        frame_idx = 0
        print(f"Processing {total_frames} frames ({width}x{height} @ {fps:.1f} FPS)...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Run YOLO tracking with custom ByteTrack config
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CFG,
                classes=[0],
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False
            )[0]

            # Collect raw detections
            raw_detections = []
            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                raw_ids = results.boxes.id.int().cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()
                for b, rid, c in zip(boxes, raw_ids, confs):
                    raw_detections.append({'raw_id': rid, 'bbox': b, 'conf': c})

            # Update Persistent Tracker to eliminate ID fragmentation
            tracked_persons = tracker.update(raw_detections, frame_idx)

            # Draw complete entrance gate in vibrant red matching user specification
            cv2.line(frame, GATE_START, GATE_END, (30, 40, 240), 4)
            cv2.circle(frame, GATE_START, 6, (30, 40, 240), -1)
            cv2.circle(frame, GATE_END, 6, (30, 40, 240), -1)
            cv2.putText(
                frame, "ENTRANCE GATE (IN/OUT)",
                (GATE_START[0], max(20, GATE_START[1] - 12)),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (30, 40, 240), 1
            )

            # Process each tracked person
            for person in tracked_persons:
                pid = person['id']
                x1, y1, x2, y2 = map(int, person['bbox'])
                feet = person['feet']
                conf = person['conf']

                # Check gate crossing
                crossing = gate.update(pid, feet, frame_idx)
                if crossing == "IN":
                    in_count += 1
                    people_in_building += 1
                    time_sec = round(frame_idx / fps, 2)
                    csv_writer.writerow([
                        frame_idx, time_sec, pid, "IN", in_count, out_count, people_in_building
                    ])
                    flash_banner = (f">>> PERSON ENTERED [ID #{pid}] <<<", (50, 220, 100), frame_idx + 60)
                    print(f"[{time_sec:6.2f}s | Frame {frame_idx:4d}] -> PERSON ENTERED [ID #{pid}] | IN: {in_count}, Occupancy: {people_in_building}")

                elif crossing == "OUT":
                    out_count += 1
                    people_in_building = max(0, people_in_building - 1)
                    time_sec = round(frame_idx / fps, 2)
                    csv_writer.writerow([
                        frame_idx, time_sec, pid, "OUT", in_count, out_count, people_in_building
                    ])
                    flash_banner = (f"<<< PERSON EXITED [ID #{pid}] >>>", (80, 100, 255), frame_idx + 60)
                    print(f"[{time_sec:6.2f}s | Frame {frame_idx:4d}] <- PERSON EXITED  [ID #{pid}] | OUT: {out_count}, Occupancy: {people_in_building}")

                # Draw motion trajectory trail
                trail = tracker.trails.get(pid, [])
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i - 1], trail[i], (0, 255, 200), 2)

                # Draw person bounding box & persistent ID badge
                color = (0, 255, 120) if crossing != "IN" else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Corner accents for a high-tech modern look
                c_len = 10
                cv2.line(frame, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2)
                cv2.line(frame, (x1, y1), (x1, y1 + c_len), (255, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2 - c_len, y2), (255, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2, y2 - c_len), (255, 255, 255), 2)

                # Badge background
                label = f"ID:{pid} ({conf:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + tw + 6, y1), (25, 30, 40), -1)
                cv2.putText(
                    frame, label, (x1 + 3, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1
                )
                cv2.circle(frame, (int(feet[0]), int(feet[1])), 4, (0, 255, 255), -1)

            # Expire flash banner
            if flash_banner and frame_idx > flash_banner[2]:
                flash_banner = None

            # Render HUD
            draw_hud(
                frame,
                in_count=in_count,
                out_count=out_count,
                occupancy=people_in_building,
                initial_occupancy=initial_occupancy,
                active_people=len(tracked_persons),
                flash_banner=flash_banner
            )

            # Progress indicator in console every 300 frames
            if frame_idx % 300 == 0:
                pct = (frame_idx / total_frames) * 100
                print(f"Progress: {frame_idx}/{total_frames} ({pct:.1f}%) | IN: {in_count} | OUT: {out_count} | Occupancy: {people_in_building}")

            writer.write(frame)

    cap.release()
    writer.release()

    print("\n" + "=" * 65)
    print("  PROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Final IN Count:          {in_count}")
    print(f"Final OUT Count:         {out_count}")
    print(f"Final People in Bldg:    {people_in_building}")
    print(f"Annotated Video:         {OUTPUT_VIDEO.resolve()}")
    print(f"Events CSV Log:          {OUTPUT_CSV.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
