import cv2
import os
from ultralytics import YOLO

# === Configuration ===
input_video_path = "D:/Test Videos/Video Datasets/video_sample1.mp4"
model_path = "model/yolov8m-face-lindevs.pt"
output_folder = "annotated_videos"

# === Load YOLOv8 Face Detection Model ===
model = YOLO(model_path)

# === Create Output Folder If Needed ===
os.makedirs(output_folder, exist_ok=True)

# === Open Input Video ===
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()

# === Get Video Properties ===
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# === Setup Output Video Writer ===
base_name = os.path.basename(input_video_path)
output_video_path = os.path.join(output_folder, f"annotated_{base_name}")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"📥 Input Video: {input_video_path}")
print(f"📤 Annotated Output: {output_video_path}")
print(f"🎥 Resolution: {width}x{height}, FPS: {fps}")

# === Setup Real-Time Display Window (Full Resolution) ===
cv2.namedWindow("YOLOv8 Face Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Face Detection", width, height)

# === Process Frame-by-Frame ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.3, verbose=False)

    face_count = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            face_count += 1

    # Overlay face count on frame
    cv2.putText(frame, f"Faces detected: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # Save frame + show live
    writer.write(frame)
    cv2.imshow("YOLOv8 Face Detection", frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === Cleanup ===
cap.release()
writer.release()
cv2.destroyAllWindows()
print("✅ Annotated video saved and display complete.")
