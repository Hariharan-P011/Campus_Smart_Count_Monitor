
from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n.pt")

video_path = Path("videos/sample.mp4")
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

results = model.predict(
    source=str(video_path),
    classes=[0],
    imgsz=640,
    device="cpu",
    conf=0.30,
    vid_stride=2,
    save=True,
    project=str(output_dir),
    name="person_detection",
    exist_ok=True
)

print("CCTV person detection completed!")
