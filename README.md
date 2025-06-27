# 🎯 Face Recognition & Tracking System

This project is a **real-time face recognition and tracking system** built using **YOLOv8**, **Deep SORT**, and **InsightFace (ArcFace)** for video surveillance and identity logging. It captures faces, assigns persistent IDs, logs entry/exit events, and prevents ID duplication across multiple videos.

---

## 📂 Project Structure

```
FACE_TRACKER/
├── model/
│   └── yolov8m-face-lindevs.pt
├── embedding/
│   └── face_{id}.npy                # Saved facial embeddings
├── logs/
│   ├── entries/YYYY-MM-DD/         # Entry face crops
│   ├── exits/YYYY-MM-DD/           # Exit face crops
│   └── events.log                  # Logs of all events
├── database/
│   └── face_events.db              # SQLite DB storing face logs
├── src/
│   ├── detector.py                 # YOLOv8 detection
│   ├── tracker.py                  # Deep SORT tracking
│   ├── recognizer.py               # Face embedding & recognition
│   ├── logger.py                   # Logging & image saving
│   └── database.py                 # SQLite DB operations
├── model/
│   └── yolov8m-face-lindevs.pt     # Face detection model
├── main.py                         # Main execution file
├── config.json                     # Config file for paths & thresholds
├── face_id_counter.txt            # Persistent ID tracking
├── test.ipynb                      # For testing model inference
└── Video representation/
    └── video_links.txt / video_info.json
```

---

## 🧠 Features

- ✅ Real-time **face detection** using **YOLOv8**
- ✅ **Face tracking** with **Deep SORT**
- ✅ **Face recognition** via **InsightFace (ArcFace) embeddings**
- ✅ **No duplicate IDs** — IDs are persistent across videos
- ✅ **Automatic logging**:
  - Entry & exit snapshots
  - Timestamped logs in `events.log`
  - Metadata in `face_events.db`
- ✅ **Handles multiple videos** one at a time for testing
- ✅ **Face quality check** before embedding (blurriness & size)
- ✅ **Logs face duration** in frame
- ✅ Works with **recorded video feeds**

---

## 🛠️ Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Key libraries:
- `ultralytics`
- `insightface`
- `deep_sort_realtime`
- `opencv-python`
- `numpy`
- `sqlite3`
- `Pillow`

---

## ⚙️ Configuration (`config.json`)

```json
{
  "frame_skip": 9,
  "similarity_threshold": 0.6,
  "log_dir": "logs",
  "embedding_dir": "embedding",
  "db_path": "database/face_events.db",
  "video_path": "D:/Test Videos/Video Datasets/record_20250620_183903.mp4"
}
```

> ✅ You can change `video_path` for testing multiple videos (one at a time).

---

## 🚀 How to Run

```bash
python main.py
```

This will:
- Detect & track faces
- Assign and persist unique IDs
- Store embeddings
- Save entry/exit face crops
- Log everything in DB and log file

---

## 📊 Database Logs

Use tools like **DB Browser for SQLite** to view:
```bash
database/face_events.db
```

Contains:
- `face_id`
- `event` (entry/exit)
- `timestamp`
- `image_path`

---

## 📦 Output Example

- `embedding/face_1.npy`
- `logs/entries/2025-06-27/face_1_14-33-21.jpg`
- `logs/exits/2025-06-27/face_1_14-36-44.jpg`
- `face_id_counter.txt → 14`
- `events.log`:
  ```
  2025-06-27 14:33:21 - 🟢 Face ID face_1 entered
  2025-06-27 14:36:44 - 🔴 Face ID face_1 exited
  ```

---

## 🎥 Demo Videos

Link to sample outputs stored in Google Drive:

📎 `Video representation/video_links.txt`

---

## 🧠 Credits

- **YOLOv8**: Ultralytics
- **Deep SORT**: Realtime multi-object tracking
- **InsightFace / ArcFace**: Face recognition & embedding

---

## Acknowledgement
This project is a part of a hackathon run by Katomaran [https://katomaran.com]
