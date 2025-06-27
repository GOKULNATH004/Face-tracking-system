import cv2
import json
import os
from src.detector import detect_faces
from src.tracker import update_tracks
from src.logger import (
    setup_logging_dirs, setup_event_logger,
    save_face_crop, log_event
)
from src.database import init_db, log_to_db
from src.recognizer import get_embedding, recognize_face, register_embedding

# Load config
with open("config.json") as f:
    CONFIG = json.load(f)

# Setup
setup_logging_dirs()
setup_event_logger()
init_db()

# ID management
def load_face_id_counter(path="face_id_counter.txt"):
    if not os.path.exists(path) or not open(path).read().strip().isdigit():
        with open(path, "w") as f:
            f.write("1")
        return 1
    with open(path, "r") as f:
        val = f.read().strip()
        return int(val) if val.isdigit() else 1

def save_face_id_counter(counter, path="face_id_counter.txt"):
    with open(path, "w") as f:
        f.write(str(counter))

next_new_id = load_face_id_counter()

# Tracking state
active_ids = set()
seen_once_ids = set()
exit_logged_ids = set()
id_last_seen = {}
id_entry_time = {}

# Open video
video_path = CONFIG["video_path"]
cap = cv2.VideoCapture(video_path)
cv2.namedWindow("Face Recognition Tracker", cv2.WINDOW_NORMAL)

frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS) or 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Video ended.")
        break

    if frame_count == 0:
        print("📐 Frame shape:", frame.shape)

    if frame_count % CONFIG["frame_skip"] == 0:
        detections = detect_faces(frame)
        tracked_faces = update_tracks(detections, frame)
        current_ids = set()

        for track_id, (x1, y1, x2, y2) in tracked_faces:
            # Expand box for visibility
            box_w, box_h = x2 - x1, y2 - y1
            pad_w, pad_h = int(box_w * 0.1), int(box_h * 0.2)
            x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
            x2, y2 = min(frame.shape[1], x2 + pad_w), min(frame.shape[0], y2 + pad_h)

            face_crop = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            current_ids.add(track_id)
            id_last_seen[track_id] = frame_count

            if face_crop is None or face_crop.size == 0:
                continue

            embedding = get_embedding(face_crop)
            if embedding is None:
                continue

            identity = recognize_face(embedding)
            if identity is None:
                identity = f"face_{next_new_id}"
                register_embedding(embedding, identity)
                next_new_id += 1
                save_face_id_counter(next_new_id)
                log_event(f"🧠 New face registered: {identity}")

            current_ids.add(identity)
            id_last_seen[identity] = frame_count

            if identity not in seen_once_ids:
                log_event(f"🟢 Face ID {identity} entered")
                path = save_face_crop(frame, (x1, y1, x2, y2), identity, "entries")
                log_to_db(identity, "entry", path, duration=None)
                seen_once_ids.add(identity)
                active_ids.add(identity)
                id_entry_time[identity] = frame_count

            cv2.putText(frame, f"ID: {identity}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detect exits
        for identity in list(active_ids):
            if identity not in current_ids:
                if frame_count - id_last_seen.get(identity, 0) > 20:
                    if identity not in exit_logged_ids:
                        duration = (frame_count - id_entry_time.get(identity, frame_count)) / fps
                        log_event(f"🔴 Face ID {identity} exited")
                        path = save_face_crop(frame, (x1, y1, x2, y2), identity, "exits")
                        log_to_db(identity, "exit", path, duration)
                        exit_logged_ids.add(identity)
                    active_ids.remove(identity)

    frame_count += 1
    cv2.imshow("Face Recognition Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()