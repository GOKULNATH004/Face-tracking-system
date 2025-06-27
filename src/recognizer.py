import insightface
import numpy as np
import os
import json
import cv2
from numpy.linalg import norm

with open("config.json") as f:
    CONFIG = json.load(f)

embedding_dir = CONFIG["embedding_dir"]
similarity_threshold = CONFIG["similarity_threshold"]

model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(face_img):
    h, w = face_img.shape[:2]
    if h < 50 or w < 50:
        return None
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 30:
        return None
    
    faces = model.get(face_img)
    if faces:
        return faces[0].embedding
    return None

def recognize_face(embedding):
    best_match = None
    best_score = float("inf")

    for file in os.listdir(embedding_dir):
        if file.endswith(".npy"):
            known_vector = np.load(os.path.join(embedding_dir, file))
            sim = norm(known_vector - embedding)
            if sim < best_score:
                best_score = sim
                best_match = file.replace(".npy", "")
    return best_match if best_score < similarity_threshold else None

def register_embedding(embedding, face_id):
    os.makedirs(embedding_dir, exist_ok=True)
    path = os.path.join(embedding_dir, f"{face_id}.npy")
    np.save(path, embedding)
    print(f"💾 Registered new face: {face_id} → {path}")
