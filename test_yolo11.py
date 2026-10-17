from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    classes=[0],
    imgsz=640,
    device="cpu",
    save=True
)

print("YOLO11 test completed successfully!")