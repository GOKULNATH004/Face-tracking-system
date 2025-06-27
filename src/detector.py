from ultralytics import YOLO

# Load YOLOv8 face model
model = YOLO("model/yolov8m-face-lindevs.pt")

def detect_faces(frame):
    results = model.predict(source=frame, conf=0.3, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(((x1, y1, x2, y2), conf))
    return detections