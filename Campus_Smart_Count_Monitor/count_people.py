
import cv2
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

VIDEO_PATH = "videos/Video Project 1.mp4"
MODEL_PATH = "yolo11n.pt"
OUTPUT_PATH = "output/in_out_count.mp4"

# Approximate positions of the two red lines in your screenshot.
# Coordinates are normalized: (x, y), each from 0.0 to 1.0.
LINE_A_START = (0.21, 0.23)
LINE_A_END = (0.48, 0.075)

LINE_B_START = (0.19, 0.25)
LINE_B_END = (0.475, 0.105)

CONFIDENCE = 0.15
IMAGE_SIZE = 640
LINE_OFFSET = 8

# ------------------------------------------------

Path("output").mkdir(exist_ok=True)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def convert_line(start, end):
    return (
        (int(start[0] * width), int(start[1] * height)),
        (int(end[0] * width), int(end[1] * height))
    )

line_a = convert_line(LINE_A_START, LINE_A_END)
line_b = convert_line(LINE_B_START, LINE_B_END)

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

# Previous stable side of each line for every track ID
previous_sides = {}

# Crossing sequence for each track ID
zone_state = {}

# Count each track ID at most once
counted_ids = set()

def line_side(point, line):
    (x1, y1), (x2, y2) = line
    px, py = point

    value = (
        (x2 - x1) * (py - y1)
        - (y2 - y1) * (px - x1)
    )

    length = max(
        1.0,
        ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    )

    if abs(value) <= LINE_OFFSET * length:
        return 0

    return 1 if value > 0 else -1

frame_number = 0

print("Processing video...")

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

    # Draw the two independent counting lines
    cv2.line(frame, line_a[0], line_a[1], (0, 255, 255), 3)
    cv2.line(frame, line_b[0], line_b[1], (255, 0, 255), 3)

    cv2.putText(
        frame, "A", line_a[0],
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
    )
    cv2.putText(
        frame, "B", line_b[0],
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2
    )

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            bx1, by1, bx2, by2 = map(int, box)

            # Bottom-center of the person box
            point = ((bx1 + bx2) // 2, by2)

            side_a = line_side(point, line_a)
            side_b = line_side(point, line_b)

            if track_id not in previous_sides:
                previous_sides[track_id] = {
                    "A": side_a,
                    "B": side_b
                }
                zone_state[track_id] = None

            else:
                old_a = previous_sides[track_id]["A"]
                old_b = previous_sides[track_id]["B"]

                crossed_a = (
                    side_a != 0 and old_a != 0 and side_a != old_a
                )
                crossed_b = (
                    side_b != 0 and old_b != 0 and side_b != old_b
                )

                state = zone_state.get(track_id)

                if track_id not in counted_ids:

                    # Both lines crossed between consecutive observations.
                    # Infer the direction using the prior and current sides.
                    if crossed_a and crossed_b:
                        if old_a == -1 and side_a == 1:
                            in_count += 1
                            counted_ids.add(track_id)
                            zone_state[track_id] = "COUNTED_IN"

                        elif old_a == 1 and side_a == -1:
                            out_count += 1
                            counted_ids.add(track_id)
                            zone_state[track_id] = "COUNTED_OUT"

                    elif crossed_a:
                        if state is None:
                            zone_state[track_id] = "A_FIRST"

                        elif state == "B_FIRST":
                            out_count += 1
                            counted_ids.add(track_id)
                            zone_state[track_id] = "COUNTED_OUT"

                    elif crossed_b:
                        if state is None:
                            zone_state[track_id] = "B_FIRST"

                        elif state == "A_FIRST":
                            in_count += 1
                            counted_ids.add(track_id)
                            zone_state[track_id] = "COUNTED_IN"

                # Update stable sides after processing the crossing.
                if side_a != 0:
                    previous_sides[track_id]["A"] = side_a

                if side_b != 0:
                    previous_sides[track_id]["B"] = side_b

            cv2.rectangle(
                frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2
            )

            cv2.circle(
                frame, point, 4, (255, 255, 0), -1
            )

            cv2.putText(
                frame, f"ID: {track_id}",
                (bx1, max(20, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

    cv2.putText(
        frame, f"IN: {in_count}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    cv2.putText(
        frame, f"OUT: {out_count}", (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )

    cv2.putText(
        frame, f"NET: {in_count - out_count}", (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2
    )

    writer.write(frame)
    frame_number += 1

cap.release()
writer.release()

print("\nProcessing completed.")
print(f"Frames processed: {frame_number}")
print(f"IN: {in_count}")
print(f"OUT: {out_count}")
print(f"Net: {in_count - out_count}")
print(f"Output video: {OUTPUT_PATH}")
