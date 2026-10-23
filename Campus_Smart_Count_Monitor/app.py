import os
import time
import json
import threading
from pathlib import Path
import cv2
from flask import Flask, Response, render_template, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)

# Base directory
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = str(BASE_DIR / "videos" / "Video Project 1.mp4")
MODEL_PATH = str(BASE_DIR / "yolo11n.pt")
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

from count_people import PersistentTracker, VirtualGateCounter, GATE_START, GATE_END, TRACKER_CFG

# Calibrated doorway counting line parameters
LINE_START = (round(GATE_START[0] / 1280.0, 3), round(GATE_START[1] / 720.0, 3))
LINE_END = (round(GATE_END[0] / 1280.0, 3), round(GATE_END[1] / 720.0, 3))
CONFIDENCE = 0.20
IMAGE_SIZE = 640
INITIAL_OCCUPANCY = 4

class StateManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.in_count = 0
        self.out_count = 0
        self.net_count = 0
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0.0
        self.is_running = False
        self.is_paused = False
        self.is_completed = False
        self.active_tracks = 0
        self.events = []
        self.latest_frame_jpeg = None
        self.last_update_ts = time.time()
        self.subscribers = []

    def register_subscriber(self):
        q = []
        with self.lock:
            self.subscribers.append(q)
        return q

    def remove_subscriber(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def notify_subscribers(self, data):
        with self.lock:
            for q in list(self.subscribers):
                q.append(data)

    def record_event(self, track_id, direction):
        timestamp = time.strftime("%H:%M:%S")
        event = {
            "id": len(self.events) + 1,
            "track_id": track_id,
            "direction": direction,
            "time": timestamp,
            "in_count": self.in_count,
            "out_count": self.out_count,
            "net_count": self.net_count,
            "frame": self.current_frame
        }
        self.events.insert(0, event)
        if len(self.events) > 100:
            self.events.pop()
        return event

    def get_status(self):
        with self.lock:
            return {
                "in_count": self.in_count,
                "out_count": self.out_count,
                "net_count": self.net_count,
                "current_frame": self.current_frame,
                "total_frames": self.total_frames,
                "fps": round(self.fps, 1),
                "progress": round((self.current_frame / max(1, self.total_frames)) * 100, 1) if self.total_frames > 0 else 0,
                "is_running": self.is_running,
                "is_paused": self.is_paused,
                "is_completed": self.is_completed,
                "active_tracks": self.active_tracks,
                "events": self.events[:20]
            }

state = StateManager()

class VideoWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.stop_requested = False

    def run(self):
        global state
        print(f"Loading YOLO model from: {MODEL_PATH}")
        try:
            model = YOLO(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            with state.lock:
                state.is_running = False
            return

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Failed to open video: {VIDEO_PATH}")
            with state.lock:
                state.is_running = False
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        line_x1 = int(width * LINE_START[0])
        line_y1 = int(height * LINE_START[1])
        line_x2 = int(width * LINE_END[0])
        line_y2 = int(height * LINE_END[1])

        gate = VirtualGateCounter(GATE_START, GATE_END)
        tracker = PersistentTracker(max_reid_distance=85, max_missing_frames=450, min_iou=0.20)

        with state.lock:
            state.total_frames = total_frames
            state.is_running = True
            state.is_paused = False
            state.is_completed = False
            state.in_count = 0
            state.out_count = 0
            state.net_count = INITIAL_OCCUPANCY

        frame_idx = 0
        fps_start_time = time.time()
        fps_frames = 0

        while not self.stop_requested:
            # Handle pause state
            while state.is_paused and not self.stop_requested:
                time.sleep(0.1)

            success, frame = cap.read()
            if not success:
                print("Video stream finished.")
                with state.lock:
                    state.is_completed = True
                    state.is_running = False
                state.notify_subscribers(state.get_status())
                break

            frame_idx += 1
            fps_frames += 1
            if time.time() - fps_start_time >= 1.0:
                calc_fps = fps_frames / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frames = 0
            else:
                calc_fps = state.fps

            # YOLO tracking
            result = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CFG,
                classes=[0],
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False
            )[0]

            active_in_frame = 0
            new_crossing_event = None

            # Draw complete entrance gate in vibrant blue matching user specification
            cv2.line(frame, GATE_START, GATE_END, (255, 80, 0), 4)
            cv2.circle(frame, GATE_START, 6, (255, 80, 0), -1)
            cv2.circle(frame, GATE_END, 6, (255, 80, 0), -1)
            cv2.putText(
                frame, "VIRTUAL GATE (IN/OUT)",
                (GATE_START[0], max(20, GATE_START[1] - 12)),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 80, 0), 1
            )

            raw_detections = []
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                confs = result.boxes.conf.cpu().tolist()
                for b, rid, c in zip(boxes, track_ids, confs):
                    raw_detections.append({'raw_id': rid, 'bbox': b, 'conf': c})

            tracked_persons = tracker.update(raw_detections, frame_idx)
            active_in_frame = len(tracked_persons)

            for person in tracked_persons:
                pid = person['id']
                bx1, by1, bx2, by2 = map(int, person['bbox'])
                feet = person['feet']
                conf = person['conf']

                crossing = gate.update(pid, feet, frame_idx)
                if crossing == "IN":
                    with state.lock:
                        state.in_count += 1
                        state.net_count = INITIAL_OCCUPANCY + state.in_count - state.out_count
                        new_crossing_event = state.record_event(pid, "IN")
                elif crossing == "OUT":
                    with state.lock:
                        state.out_count += 1
                        state.net_count = max(0, INITIAL_OCCUPANCY + state.in_count - state.out_count)
                        new_crossing_event = state.record_event(pid, "OUT")

                # Trajectory trail
                trail = tracker.trails.get(pid, [])
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i - 1], trail[i], (0, 255, 200), 2)

                # Stylish bounding box & ID
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40, 167, 69), 2)
                cv2.circle(frame, (int(feet[0]), int(feet[1])), 4, (235, 99, 37), -1)
                
                label = f"ID #{pid} ({conf:.2f})"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (bx1, max(0, by1 - 20)), (bx1 + lw + 6, max(20, by1)), (40, 167, 69), -1)
                cv2.putText(
                    frame, label,
                    (bx1 + 3, max(15, by1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1
                )

            # Update state counts and frame
            with state.lock:
                state.current_frame = frame_idx
                state.fps = calc_fps
                state.active_tracks = active_in_frame
                state.last_update_ts = time.time()

            # Encode frame for live streaming
            # Resize slightly if too large to ensure fast, buttery-smooth browser streaming
            preview_frame = frame
            if width > 960:
                scale = 960.0 / width
                preview_frame = cv2.resize(frame, (960, int(height * scale)))

            _, jpeg_buf = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with state.lock:
                state.latest_frame_jpeg = jpeg_buf.tobytes()

            # Push SSE updates whenever crossing happens or every few frames
            status_data = state.get_status()
            if new_crossing_event:
                status_data["latest_event"] = new_crossing_event
            state.notify_subscribers(status_data)

            # Small delay to keep CPU usage optimal and simulate smooth real-time video
            time.sleep(0.01)

        cap.release()
        with state.lock:
            state.is_running = False

worker_thread = None

def start_worker():
    global worker_thread
    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = VideoWorker()
        worker_thread.start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def get_status():
    return jsonify(state.get_status())

@app.route("/api/stream_events")
def stream_events():
    def event_stream():
        q = state.register_subscriber()
        try:
            # Yield initial status immediately on connect
            init_data = state.get_status()
            yield f"data: {json.dumps(init_data)}\n\n"
            while True:
                # Wait for next event or heartbeat
                if len(q) > 0:
                    data = q.pop(0)
                    yield f"data: {json.dumps(data)}\n\n"
                else:
                    time.sleep(0.05)
        except GeneratorExit:
            state.remove_subscriber(q)

    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/video_feed")
def video_feed():
    def generate_feed():
        while True:
            frame_bytes = None
            with state.lock:
                frame_bytes = state.latest_frame_jpeg
            
            if frame_bytes is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.033)  # ~30 fps

    return Response(generate_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/control", methods=["POST"])
def control():
    action = request.json.get("action")
    global worker_thread

    if action == "start":
        if not state.is_running or state.is_completed:
            with state.lock:
                state.in_count = 0
                state.out_count = 0
                state.net_count = 0
                state.events.clear()
            start_worker()
        elif state.is_paused:
            with state.lock:
                state.is_paused = False

    elif action == "pause":
        with state.lock:
            state.is_paused = True

    elif action == "resume":
        with state.lock:
            state.is_paused = False

    elif action == "reset":
        if worker_thread and worker_thread.is_alive():
            worker_thread.stop_requested = True
            worker_thread.join(timeout=2.0)
        with state.lock:
            state.in_count = 0
            state.out_count = 0
            state.net_count = 0
            state.current_frame = 0
            state.active_tracks = 0
            state.events.clear()
            state.is_running = False
            state.is_paused = False
            state.is_completed = False
            state.latest_frame_jpeg = None
        start_worker()

    return jsonify({"success": True, "state": state.get_status()})

if __name__ == "__main__":
    # Auto-start worker on launch
    start_worker()
    print("Dashboard starting at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
