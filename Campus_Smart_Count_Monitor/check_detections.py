
import cv2
from ultralytics import YOLO
from pathlib import Path

video_path = Path("videos/Video Project 1.mp4")
output_path = Path("output/detection_diagnostic.mp4")

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
    (width, height)
)

frame_number = 0
empty_frames = 0
max_frames = 900

while frame_number < max_frames:
    success, frame = cap.read()

    if not success:
        break

    results = model.predict(
        frame,
        classes=[0],
        conf=0.10,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    result = results[0]
    annotated = frame.copy()

    count = 0

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().tolist()
        count = len(boxes)

        for box, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                annotated, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cv2.putText(
                annotated,
                f"Person {confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2
            )

    if count == 0:
        empty_frames += 1

    cv2.putText(
        annotated,
        f"Frame: {frame_number} | People detected: {count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 255), 2
    )

    writer.write(annotated)
    frame_number += 1

cap.release()
writer.release()

print(f"Frames processed: {frame_number}")
print(f"Frames with zero person detections: {empty_frames}")
print(f"Diagnostic video saved to: {output_path}")
