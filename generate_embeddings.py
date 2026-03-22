import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import cv2
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import insightface
from insightface.app import FaceAnalysis

DATASET_DIR      = "course_project_dataset"
OUTPUT_FILE      = "embeddings.pkl"

MIN_FACE_SIZE    = 80
MIN_CONFIDENCE   = 0.90

face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

student_embeddings = defaultdict(list)

for student in tqdm(os.listdir(DATASET_DIR), desc="Processing"):
    student_dir = os.path.join(DATASET_DIR, student)
    if not os.path.isdir(student_dir):
        continue

    for file in os.listdir(student_dir):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(student_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue

        try:
            faces = face_app.get(img)
        except:
            continue

        if not faces:
            print("no face found")
            continue

        best = max(faces, key=lambda x: x.det_score)

        conf = best.det_score
        bbox = best.bbox.astype(int)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # if conf < MIN_CONFIDENCE or min(w, h) < MIN_FACE_SIZE:
        #     print("face not clear")
        #     continue

        emb = best.embedding
        student_embeddings[student].append(np.array(emb))


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


def cosine_distance(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def euclidean_distance(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    if a.shape != b.shape:
        raise ValueError("Vectors must be of the same size")
    
    return np.linalg.norm(a - b)

min_dist = float('inf')
min_pair = ("", "")

for i in range(len(known_encodings)):
    for j in range(i + 1, len(known_encodings)):
        dist = cosine_distance(known_encodings[i], known_encodings[j])
        if dist < min_dist:
            min_dist = dist
            min_pair = (known_names[i], known_names[j])

if min_dist != float('inf'):
    print(f"\nMinimum distance between two different persons: {min_dist:.4f} ({min_pair[0]} & {min_pair[1]})\n")

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({
        "encodings": known_encodings,
        "names": known_names
    }, f)