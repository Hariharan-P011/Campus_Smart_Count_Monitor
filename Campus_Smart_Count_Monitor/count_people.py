
import cv2
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

VIDEO_PATH = "videos/Video Project 1.mp4"
MODEL_PATH = "yolo11n.pt"
OUTPUT_PATH = "output/in_out_count.mp4"

# Diagonal line estimated from your screenshot
LINE_START = (0.21, 0.26)
LINE_END = (0.50, 0.085)

CONFIDENCE = 0.20
IMAGE_SIZE = 640
LINE_OFFSET = 10

# ------------------------------------------------

Path("output").mkdir(exist_ok=True)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Convert normalized coordinates to actual video coordinates
line_x1 = int(width * LINE_START[0])
line_y1 = int(height * LINE_START[1])
line_x2 = int(width * LINE_END[0])
line_y2 = int(height * LINE_END[1])

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps if fps > 0 else 25,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Cannot create output video")

in_count = 0
out_count = 0

# Last stable side recorded for each track ID
last_side = {}

# Avoid repeated counts for the same ID and crossing
last_counted_side = {}

frame_number = 0

print("Processing video...")
print(f"Counting line: ({line_x1}, {line_y1}) to "
      f"({line_x2}, {line_y2})")

while True:
    success, frame = cap.read()

    if not success:
        break

    result = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False
    )[0]

    # Draw the diagonal counting line
    cv2.line(
        frame,
        (line_x1, line_y1),
        (line_x2, line_y2),
        (0, 255, 255),
        3
    )

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            bx1, by1, bx2, by2 = map(int, box)

            # Bottom-center is a better crossing reference
            # than the center of the person's bounding box.
            center_x = (bx1 + bx2) // 2
            center_y = by2

            # Signed position relative to the diagonal line
            cross_value = (
                (line_x2 - line_x1) * (center_y - line_y1)
                - (line_y2 - line_y1) * (center_x - line_x1)
            )

            # Ignore points too close to the line
            if abs(cross_value) > LINE_OFFSET * max(
                1, ((line_x2 - line_x1) ** 2
                    + (line_y2 - line_y1) ** 2) ** 0.5
            ):
                side = 1 if cross_value > 0 else -1

                previous_side = last_side.get(track_id)

                if (
                    previous_side is not None
                    and side != previous_side
                    and last_counted_side.get(track_id) != side
                ):
                    # Direction labels are provisional.
                    # Reverse IN and OUT if your entrance orientation differs.
                    if previous_side == -1 and side == 1:
                        in_count += 1
                        last_counted_side[track_id] = side

                    elif previous_side == 1 and side == -1:
                        out_count += 1
                        last_counted_side[track_id] = side

                last_side[track_id] = side

            # Draw person detection and tracking ID
            cv2.rectangle(
                frame,
                (bx1, by1),
                (bx2, by2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (255, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (bx1, max(20, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Display counts
    cv2.putText(
        frame, f"IN: {in_count}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (0, 255, 0), 2
    )

    cv2.putText(
        frame, f"OUT: {out_count}", (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (0, 0, 255), 2
    )

    cv2.putText(
        frame, f"NET: {in_count - out_count}", (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (255, 255, 0), 2
    )

    writer.write(frame)
    frame_number += 1

cap.release()
writer.release()

print("\nProcessing completed.")
print(f"Frames processed: {frame_number}")
print(f"IN: {in_count}")
print(f"OUT: {out_count}")
print(f"Net count: {in_count - out_count}")
print(f"Output video: {OUTPUT_PATH}")
