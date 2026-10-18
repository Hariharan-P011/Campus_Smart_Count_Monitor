
import cv2
from pathlib import Path
from ultralytics import YOLO

video_path = Path("videos/Video Project 1.mp4")
output_path = Path("output/detection_vs_tracking.mp4")

output_path.parent.mkdir(parents=True, exist_ok=True)

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width * 2, height)
)

frame_no = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Independent person detection
    detection = model.predict(
        frame,
        classes=[0],
        conf=0.10,
        imgsz=640,
        device="cpu",
        verbose=False
    )[0]

    # Persistent BoT-SORT tracking
    tracking = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[0],
        conf=0.10,
        imgsz=640,
        device="cpu",
        verbose=False
    )[0]

    left = frame.copy()
    right = frame.copy()

    if detection.boxes is not None:
        for box in detection.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(
                left, (x1, y1), (x2, y2), (255, 0, 0), 2
            )

    if tracking.boxes is not None:
        boxes = tracking.boxes.xyxy.cpu().numpy()
        ids = tracking.boxes.id

        if ids is not None:
            ids = ids.int().cpu().tolist()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(
                    right, (x1, y1), (x2, y2), (0, 255, 0), 2
                )
                cv2.putText(
                    right, f"ID: {track_id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

    cv2.putText(
        left, "DETECTION", (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    cv2.putText(
        right, "BOT-SORT TRACKING", (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    combined = cv2.hconcat([left, right])
    writer.write(combined)
    frame_no += 1

cap.release()
writer.release()

print(f"Frames processed: {frame_no}")
print(f"Diagnostic video saved: {output_path}")
