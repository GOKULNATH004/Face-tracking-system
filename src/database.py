import sqlite3
import os
import json
from datetime import datetime

with open("config.json") as f:
    CONFIG = json.load(f)

db_path = CONFIG["db_path"]

# Create table
def init_db():
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id TEXT,
            event TEXT,
            timestamp TEXT,
            image_path TEXT,
            duration REAL
        )
    ''')
    conn.commit()
    conn.close()

# Insert values
def log_to_db(face_id, event, image_path, duration=None):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        INSERT INTO events (face_id, event, timestamp, image_path, duration)
        VALUES (?, ?, ?, ?, ?)
    ''', (face_id, event, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path, duration))
    conn.commit()
    conn.close()
    print(f"📥 DB log: {face_id} - {event}")
