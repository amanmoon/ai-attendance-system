import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import cv2
import pickle
import tempfile
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from deepface import DeepFace

from mark_attendance import enhance_face_crop

DATASET_DIR      = "course_project_dataset"
OUTPUT_FILE      = "embeddings_dl.pkl"
EMBEDDING_MODEL  = "Facenet512"
DETECTOR_BACKEND = "retinaface"

BBOX_PADDING = 0.20

students = [
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
]

student_embeddings: dict[str, list] = defaultdict(list)

for student_name in tqdm(students, desc="Processing students"):
    student_dir = os.path.join(DATASET_DIR, student_name)

    for filename in os.listdir(student_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(student_dir, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Cannot read {image_path}")
            continue

        h0, w0 = img.shape[:2]

        try:
            detected = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )
        except Exception as e:
            continue

        if not detected:
            continue

        best = max(detected, key=lambda d: d.get("confidence", 0.0))
        fa   = best["facial_area"]
        x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

        pad_x  = int(w * BBOX_PADDING)
        pad_y  = int(h * BBOX_PADDING)
        x_pad  = max(0, x - pad_x)
        y_pad  = max(0, y - pad_y)
        x2_pad = min(w0, x + w + pad_x)
        y2_pad = min(h0, y + h + pad_y)

        crop = img[y_pad:y2_pad, x_pad:x2_pad]
        if crop.size == 0:
            print(f"Empty crop for {image_path}")
            continue

        enhanced = enhance_face_crop(crop)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as ftmp:
            tmp_path = ftmp.name
        cv2.imwrite(tmp_path, enhanced)

        try:
            results = DeepFace.represent(
                img_path=tmp_path,
                model_name=EMBEDDING_MODEL,
                detector_backend="skip",
                enforce_detection=False,
            )
            if results:
                embedding = results[0]["embedding"]
                student_embeddings[student_name].append(embedding)
                print(f"{student_name} / {filename}")
            else:
                print(f"Empty embedding for {image_path}")
        except Exception as e:
            print(f"Embedding error for {image_path}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

known_face_encodings = []
known_face_names     = []

for name, embs in student_embeddings.items():
    if not embs:
        print(f"WARNING: No valid embeddings for {name} — skipping.")
        continue
    avg_embedding = np.mean(embs, axis=0).tolist()
    known_face_encodings.append(avg_embedding)
    known_face_names.append(name)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
