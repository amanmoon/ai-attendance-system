import os
import cv2
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics import precision_recall_curve, f1_score, auc

DATASET_DIR = "course_project_dataset"
EMBEDDINGS_FILE = "clustering_calibration_embeddings.pkl"
OUTPUT_DIR = "clustering_calibration_output"

def cosine_distance(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def extract_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)

    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    data = []
    
    for person_name in tqdm(os.listdir(DATASET_DIR), desc="Processing persons"):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            try:
                faces = face_app.get(img)
            except Exception:
                continue

            if not faces:
                continue

            best_face = max(faces, key=lambda x: x.det_score)
            
            if best_face.det_score < 0.1:
                continue

            data.append({
                "person": person_name,
                "image": img_path,
                "embedding": best_face.embedding
            })

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(data, f)
        
    return data

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = extract_embeddings()
    
    if len(data) < 2:
        return

    labels = []
    distances = []
    
    n = len(data)
    for i in tqdm(range(n), desc="Pairwise comparisons"):
        for j in range(i + 1, n):
            emb1, p1 = data[i]["embedding"], data[i]["person"]
            emb2, p2 = data[j]["embedding"], data[j]["person"]
            
            dist = cosine_distance(emb1, emb2)
            distances.append(dist)
            labels.append(1 if p1 == p2 else 0)

    distances = np.array(distances)
    labels = np.array(labels)
    
    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)
    print(f"Same person: {pos_count}")
    print(f"Different person: {neg_count}")

    similarities = 1.0 - distances
    
    precision, recall, thresholds_sim = precision_recall_curve(labels, similarities)
    
    thresholds_dist = 1.0 - thresholds_sim
    
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds_dist[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    
    pr_auc = auc(recall, precision)

    print(f"\n--- Calibration Results ---")
    print(f"Optimal Threshold (Max F1): {best_threshold:.4f}")
    print(f"Max F1 Score: {best_f1:.4f}")
    print(f"Precision at optimal: {best_precision:.4f}")
    print(f"Recall at optimal: {best_recall:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})', color='blue')
    ax.plot(best_recall, best_precision, marker='o', markersize=8, color='red', 
            label=f'Optimal (Thr={best_threshold:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve for Face Clustering')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    pr_plot_path = os.path.join(OUTPUT_DIR, "precision_recall_curve.png")
    fig.savefig(pr_plot_path, dpi=150)
    plt.close(fig)
    
    sorted_indices = np.argsort(thresholds_dist)
    sorted_thresholds = thresholds_dist[sorted_indices]
    sorted_f1 = f1_scores[sorted_indices]
    sorted_prec = precision[:-1][sorted_indices]
    sorted_rec = recall[:-1][sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_thresholds, sorted_f1, label='F1 Score', color='green', linewidth=2)
    ax.plot(sorted_thresholds, sorted_prec, label='Precision', color='blue', linestyle='--')
    ax.plot(sorted_thresholds, sorted_rec, label='Recall', color='red', linestyle='--')
    ax.axvline(x=best_threshold, color='black', linestyle=':', label=f'Best Threshold ({best_threshold:.3f})')
    ax.set_xlabel('Cosine Distance Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Clustering Metrics vs Threshold')
    ax.legend(loc='lower center')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    
    f1_plot_path = os.path.join(OUTPUT_DIR, "f1_vs_threshold.png")
    fig.savefig(f1_plot_path, dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
