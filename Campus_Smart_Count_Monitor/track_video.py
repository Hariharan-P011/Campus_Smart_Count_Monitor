
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO

video_path = Path("videos/Video Project 1.mp4")
output_path = Path("output/botsort_debug.mp4")
csv_path = Path("output/botsort_tracks.csv")

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

with open(csv_path, "w", newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([
        "frame", "track_id", "x1", "y1", "x2", "y2", "confidence"
    ])

    while True:
        success, frame = cap.read()

        if not success:
            break

        result = model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],
            conf=0.20,
            iou=0.50,
            imgsz=640,
            device="cpu",
            verbose=False
        )[0]

        annotated = frame.copy()

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()

            for box, track_id, confidence in zip(
                boxes, ids, confidences
            ):
                x1, y1, x2, y2 = map(int, box)

                csv_writer.writerow([
                    frame_number, track_id,
                    x1, y1, x2, y2, round(confidence, 3)
                ])

                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

                cv2.putText(
                    annotated,
                    f"ID:{track_id} C:{confidence:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2
                )

        cv2.putText(
            annotated,
            f"Frame: {frame_number}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 0), 2
        )

        writer.write(annotated)
        frame_number += 1

cap.release()
writer.release()

print(f"Frames processed: {frame_number}")
print(f"Tracking video: {output_path}")
print(f"Track log: {csv_path}")
