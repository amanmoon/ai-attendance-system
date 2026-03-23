import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from deepface import DeepFace

from mark_attendance import enhance_face_crop

DATASET_DIR      = "course_project_dataset"
OUTPUT_FILE      = "embeddings_dl_no_enhancement.pkl"
EMBEDDING_MODEL = "Facenet512"
DETECTOR         = "retinaface"

MIN_FACE_SIZE    = 80
MIN_CONFIDENCE   = 0.90

student_embeddings = defaultdict(list)

for student in tqdm(os.listdir(DATASET_DIR), desc="Processing"):
    student_dir = os.path.join(DATASET_DIR, student)
    if not os.path.isdir(student_dir):
        continue

    for file in os.listdir(student_dir):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(student_dir, file)

        try:
            faces = DeepFace.extract_faces(
                img_path=path,
                detector_backend=DETECTOR,
                align=True,
                enforce_detection=True,
            )
        except:
            continue

        if not faces:
            continue

        best = max(faces, key=lambda x: x.get("confidence", 0))

        conf = best.get("confidence", 0)
        fa   = best.get("facial_area", {})
        w, h = fa.get("w", 0), fa.get("h", 0)

        if conf < MIN_CONFIDENCE or min(w, h) < MIN_FACE_SIZE:
            continue

        face = best["face"]
        face = (face * 255).astype(np.uint8)
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

        # face = enhance_face_crop(face)

        try:
            emb = DeepFace.represent(
                img_path=face,
                model_name=EMBEDDING_MODEL,
                detector_backend="skip",
                enforce_detection=False,
            )[0]["embedding"]

            student_embeddings[student].append(np.array(emb))

        except:
            continue


def clean_embeddings(embs, threshold=0.7):
    if len(embs) < 3:
        return embs

    embs = np.array(embs)
    centroid = np.mean(embs, axis=0)

    sims = np.dot(embs, centroid) / (
        np.linalg.norm(embs, axis=1) * np.linalg.norm(centroid)
    )

    return embs[sims > threshold]


known_encodings = []
known_names     = []

for name, embs in student_embeddings.items():
    if not embs:
        print(f"Skipping {name}")
        continue

    embs = clean_embeddings(embs)

    final_emb = np.median(embs, axis=0)

    known_encodings.append(final_emb.tolist())
    known_names.append(name)


with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({
        "encodings": known_encodings,
        "names": known_names
    }, f)