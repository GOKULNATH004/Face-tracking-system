import os
import json
import logging
from datetime import datetime
from PIL import Image

with open("config.json") as f:
    CONFIG = json.load(f)

LOG_DIR = CONFIG["log_dir"]

# Creating files
def setup_logging_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "entries"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "exits"), exist_ok=True)
    os.makedirs(CONFIG["embedding_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["db_path"]), exist_ok=True)

def setup_event_logger():
    log_file = os.path.join(LOG_DIR, "events.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

def log_event(message):
    print(message)
    logging.info(message)

def save_face_crop(frame, box, face_id, event_type):
    today = datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join(LOG_DIR, event_type, today)
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%H-%M-%S")
    filename = f"{face_id}_{timestamp}.jpg"
    path = os.path.join(folder, filename)

    x1, y1, x2, y2 = box
    face_img = frame[y1:y2, x1:x2]
    if face_img.size == 0:
        print("⚠️ Face crop was empty.")
        return ""
    im = Image.fromarray(face_img)
    im.save(path)
    print(f"✅ Saved face crop: {path}")
    return path
