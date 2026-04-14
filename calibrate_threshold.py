import os
import sys
import csv
import argparse
import pickle
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from collections import defaultdict

import insightface
from insightface.app import FaceAnalysis

DEFAULT_TEST_DIR       = "test_data"
DEFAULT_EMBEDDINGS     = "embeddings/embeddings_dl.pkl"
DEFAULT_GROUND_TRUTH   = "test_data/final_attendance.csv"
DEFAULT_OUTPUT_DIR     = "calibration_output"
DEFAULT_TARGET_FAR     = 0.01

USE_TILING    = True
TILE_SIZE     = 1500
TILE_OVERLAP  = 0.20


GT_TO_ENROLLED = {
    "Shah Pratham Manish":         "Pratham Manish Shah",
    "Pulkit":                     "Pulkit Pulkit",
    "Abhishek":                   "Abhishek Abhishek",
    "Jyoti":                      "Jyoti Jyoti",
    "Ganesh Dattu Yadawate":      "Ganesh Yadawate",
    "Amresh Kumar Jha":           "Amresh Jha",
    "Anish Prashant Mayanache":   "Anish Mayanache",
    "Maitreya Gautam Shelare":    "Maitreya Shelare",
    "Manas Sudam Patil":          "Manas Patil",
    "Samyak Sanjay Parakh":       "Samyak Parakh",
}


def cosine_distance(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def non_max_suppression(boxes, scores, iou_threshold=0.4):
    if not boxes:
        return []
    boxes  = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0];  y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2];  y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr   = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order  = order[np.where(ovr <= iou_threshold)[0] + 1]
    return keep


def tiled_face_detection(face_app, image, tile_size=1500, overlap=0.20):
    h, w = image.shape[:2]
    stride = int(tile_size * (1.0 - overlap))
    all_dets = []

    y_coords = list(range(0, max(1, h - tile_size + stride), stride))
    x_coords = list(range(0, max(1, w - tile_size + stride), stride))
    if y_coords and y_coords[-1] + tile_size < h: y_coords.append(h - tile_size)
    if x_coords and x_coords[-1] + tile_size < w: x_coords.append(w - tile_size)
    if not y_coords: y_coords = [0]
    if not x_coords: x_coords = [0]

    for ty in y_coords:
        for tx in x_coords:
            tile = image[ty:min(ty + tile_size, h), tx:min(tx + tile_size, w)]
            try:
                faces = face_app.get(tile)
            except Exception:
                faces = []
            for f in faces:
                if f.det_score < 0.1:
                    continue
                bbox = f.bbox.copy()
                bbox[0] += tx;  bbox[1] += ty;  bbox[2] += tx;  bbox[3] += ty
                all_dets.append({
                    "bbox": bbox,
                    "confidence": float(f.det_score),
                    "embedding": f.embedding,
                })

    if not all_dets:
        return []
    boxes  = [[d["bbox"][0], d["bbox"][1],
               d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]]
              for d in all_dets]
    scores = [d["confidence"] for d in all_dets]
    keep   = non_max_suppression(boxes, scores, iou_threshold=0.3)
    return [all_dets[i] for i in keep]


def load_ground_truth(csv_path, enrolled_names):
    """Return dict  image_basename → set-of-enrolled-names marked Present."""
    # The CSV has rows:  name, status  (P/Present/A/Absent)
    # All rows belong to one common image set (same test session).
    present = set()
    enrolled_set = set(enrolled_names)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            raw_name, status = row[0].strip(), row[1].strip().upper()
            if status not in ("P", "PRESENT"):
                continue
            name = GT_TO_ENROLLED.get(raw_name, raw_name)
            if name in enrolled_set:
                present.add(name)
            else:
                for en in enrolled_set:
                    if raw_name in en or en in raw_name:
                        present.add(en)
                        break

    return present


def run_calibration(args):
    with open(args.embeddings, "rb") as f:
        data = pickle.load(f)
    enrolled_encodings = [np.asarray(e, dtype=np.float32) for e in data["encodings"]]
    enrolled_names     = data["names"]

    gt_present = load_ground_truth(args.ground_truth, enrolled_names)
    gt_absent  = set(enrolled_names) - gt_present
    print(f"Present: {len(gt_present)}   Absent: {len(gt_absent)}")

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    image_files = sorted([
        f for f in os.listdir(args.test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(image_files)} images")

    all_test_embeddings = []
    for img_file in image_files:
        img_path = os.path.join(args.test_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_file}, skipping")
            continue

        if USE_TILING:
            dets = tiled_face_detection(face_app, img, TILE_SIZE, TILE_OVERLAP)
        else:
            try:
                faces = face_app.get(img)
            except Exception:
                faces = []
            dets = [{"embedding": f.embedding, "confidence": float(f.det_score)}
                    for f in faces if f.det_score >= 0.1]

        for det in dets:
            all_test_embeddings.append(det["embedding"])

        print(f"{img_file}: {len(dets)} faces detected")

    n_faces = len(all_test_embeddings)
    print(f"Total detected faces: {n_faces}")

    if n_faces == 0:
        print("ERROR: No faces detected.  Check test images / model.")
        sys.exit(1)

    genuine_distances  = []
    impostor_distances = []
    all_distances = []
    face_results = []

    for face_emb in all_test_embeddings:
        dists = []
        for i, enc in enumerate(enrolled_encodings):
            d = cosine_distance(enc, face_emb)
            dists.append((d, enrolled_names[i]))
        dists.sort(key=lambda x: x[0])

        best_dist, best_name = dists[0]
        second_dist = dists[1][0] if len(dists) > 1 else 999.0

        face_results.append((best_name, best_dist, second_dist))

        if best_name in gt_present:
            genuine_distances.append(best_dist)
        else:
            impostor_distances.append(best_dist)

    enrolled_min_dist = {}
    for name in enrolled_names:
        idx = enrolled_names.index(name)
        enc = enrolled_encodings[idx]
        min_d = min(cosine_distance(enc, fe) for fe in all_test_embeddings)
        enrolled_min_dist[name] = min_d

    for name in gt_present:
        if name in enrolled_min_dist:
            genuine_distances.append(enrolled_min_dist[name])
    for name in gt_absent:
        if name in enrolled_min_dist:
            impostor_distances.append(enrolled_min_dist[name])

    genuine_distances  = np.array(genuine_distances)
    impostor_distances = np.array(impostor_distances)

    print(f"Genuine  pairs: {len(genuine_distances)}")
    print(f"Impostor pairs: {len(impostor_distances)}")
    print(f"Genuine  dist range: [{genuine_distances.min():.4f}, {genuine_distances.max():.4f}]  "
          f"mean={genuine_distances.mean():.4f}  median={np.median(genuine_distances):.4f}")
    print(f"Impostor dist range: [{impostor_distances.min():.4f}, {impostor_distances.max():.4f}]  "
          f"mean={impostor_distances.mean():.4f}  median={np.median(impostor_distances):.4f}")
    print(f"\n[5/5] Sweeping thresholds and generating plots ...")
    os.makedirs(args.output_dir, exist_ok=True)

    thresholds = np.linspace(0.0, 1.0, 2001)
    far_arr = np.zeros_like(thresholds)
    frr_arr = np.zeros_like(thresholds)
    tpr_arr = np.zeros_like(thresholds)

    n_gen = len(genuine_distances)
    n_imp = len(impostor_distances)

    for i, t in enumerate(thresholds):
        far_arr[i] = np.sum(impostor_distances <= t) / max(n_imp, 1)
        frr_arr[i] = np.sum(genuine_distances > t) / max(n_gen, 1)
        tpr_arr[i] = 1.0 - frr_arr[i]

    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer_threshold = thresholds[eer_idx]
    eer_value     = (far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0

    valid = np.where(far_arr <= args.target_far)[0]
    if len(valid) > 0:
        target_idx = valid[-1]
    else:
        target_idx = 0
    target_threshold = thresholds[target_idx]
    target_frr       = frr_arr[target_idx]
    target_far       = far_arr[target_idx]

    youden = tpr_arr - far_arr
    youden_idx = np.argmax(youden)
    recommended_threshold = thresholds[youden_idx]
    rec_far = far_arr[youden_idx]
    rec_frr = frr_arr[youden_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1.0, 101)
    ax.hist(genuine_distances,  bins=bins, alpha=0.6, color="#2196F3", label="Genuine (same person)", density=True)
    ax.hist(impostor_distances, bins=bins, alpha=0.6, color="#F44336", label="Impostor (different person)", density=True)
    ax.axvline(x=0.4,                   color="gray",   linestyle="--", linewidth=1.5, label=f"Current threshold (0.400)")
    ax.axvline(x=eer_threshold,         color="#FF9800", linestyle="-",  linewidth=2,   label=f"EER threshold ({eer_threshold:.3f})")
    ax.axvline(x=recommended_threshold, color="#4CAF50", linestyle="-",  linewidth=2,   label=f"Recommended ({recommended_threshold:.3f})")
    ax.set_xlabel("Cosine Distance", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Genuine vs Impostor Distance Distributions (Indian Face Data)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    dist_plot = os.path.join(args.output_dir, "distance_distributions.png")
    fig.savefig(dist_plot, dpi=150)
    plt.close(fig)
    print(f"       Saved: {dist_plot}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(far_arr, tpr_arr, color="#1565C0", linewidth=2, label="ROC Curve")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.scatter([far_arr[eer_idx]], [tpr_arr[eer_idx]],
               color="#FF9800", s=100, zorder=5, label=f"EER ({eer_threshold:.3f})")
    ax.scatter([rec_far], [1 - rec_frr],
               color="#4CAF50", s=100, zorder=5, label=f"Recommended ({recommended_threshold:.3f})")

    # Mark current threshold (0.4) on the curve
    cur_idx = np.argmin(np.abs(thresholds - 0.4))
    ax.scatter([far_arr[cur_idx]], [tpr_arr[cur_idx]],
               color="gray", s=100, zorder=5, marker="x", label=f"Current (0.400)")

    ax.set_xlabel("False Acceptance Rate (FAR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (1 - FRR)", fontsize=12)
    ax.set_title("ROC Curve — Face Verification", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    roc_plot = os.path.join(args.output_dir, "roc_curve.png")
    fig.savefig(roc_plot, dpi=150)
    plt.close(fig)
    print(f"       Saved: {roc_plot}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, far_arr, color="#F44336", linewidth=2, label="FAR (False Acceptance)")
    ax.plot(thresholds, frr_arr, color="#2196F3", linewidth=2, label="FRR (False Rejection)")
    ax.axvline(x=eer_threshold,         color="#FF9800", linestyle="--", linewidth=1.5, label=f"EER ({eer_threshold:.3f})")
    ax.axvline(x=recommended_threshold, color="#4CAF50", linestyle="--", linewidth=1.5, label=f"Recommended ({recommended_threshold:.3f})")
    ax.axvline(x=target_threshold,      color="#9C27B0", linestyle="--", linewidth=1.5, label=f"FAR≤{args.target_far:.2%} ({target_threshold:.3f})")
    ax.axvline(x=0.4,                   color="gray",   linestyle=":",  linewidth=1.5, label="Current (0.400)")
    ax.set_xlabel("Threshold (Cosine Distance)", fontsize=12)
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title("FAR / FRR vs Threshold", fontsize=13)
    ax.legend(fontsize=9, loc="center right")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    err_plot = os.path.join(args.output_dir, "far_frr_curve.png")
    fig.savefig(err_plot, dpi=150)
    plt.close(fig)
    print(f"       Saved: {err_plot}")

    present_dists = {n: d for n, d in enrolled_min_dist.items() if n in gt_present}
    absent_dists  = {n: d for n, d in enrolled_min_dist.items() if n in gt_absent}

    fig, ax = plt.subplots(figsize=(14, 6))
    sorted_present = sorted(present_dists.items(), key=lambda x: x[1])
    sorted_absent  = sorted(absent_dists.items(),  key=lambda x: x[1])

    names_ordered = [n for n, _ in sorted_present] + [n for n, _ in sorted_absent]
    dists_ordered = [d for _, d in sorted_present] + [d for _, d in sorted_absent]
    colors = ["#2196F3"] * len(sorted_present) + ["#F44336"] * len(sorted_absent)

    x_pos = range(len(names_ordered))
    bars = ax.bar(x_pos, dists_ordered, color=colors, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0.4,                   color="gray",   linestyle="--", linewidth=1.5, label="Current (0.400)")
    ax.axhline(y=recommended_threshold, color="#4CAF50", linestyle="-",  linewidth=2,   label=f"Recommended ({recommended_threshold:.3f})")
    ax.axhline(y=eer_threshold,         color="#FF9800", linestyle="-",  linewidth=1.5, label=f"EER ({eer_threshold:.3f})")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.split()[0] for n in names_ordered], rotation=90, fontsize=7)
    ax.set_ylabel("Min Cosine Distance to Any Detected Face", fontsize=11)
    ax.set_title("Per-Student Best Match Distance  (Blue = Present, Red = Absent)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(-1, len(names_ordered))
    fig.tight_layout()
    student_plot = os.path.join(args.output_dir, "per_student_distances.png")
    fig.savefig(student_plot, dpi=150)
    plt.close(fig)
    print(f"       Saved: {student_plot}")

    sim_thresholds = [0.30, 0.35, recommended_threshold, eer_threshold, 0.40, 0.45, 0.50]
    sim_thresholds = sorted(set(round(t, 4) for t in sim_thresholds))

    sim_results = []
    for t in sim_thresholds:
        tp = sum(1 for n in gt_present if enrolled_min_dist.get(n, 999) <= t)
        fp = sum(1 for n in gt_absent  if enrolled_min_dist.get(n, 999) <= t)
        fn = len(gt_present) - tp
        tn = len(gt_absent) - fp
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)
        sim_results.append({
            "threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        })

    report_path = os.path.join(args.output_dir, "calibration_report.txt")
    with open(report_path, "w") as rpt:
        rpt.write("=" * 70 + "\n")
        rpt.write("  THRESHOLD CALIBRATION REPORT\n")
        rpt.write(f"  Model:       InsightFace buffalo_l\n")
        rpt.write(f"  Metric:      Cosine distance\n")
        rpt.write(f"  Test images: {len(image_files)}\n")
        rpt.write(f"  Faces found: {n_faces}\n")
        rpt.write(f"  Enrolled:    {len(enrolled_names)}\n")
        rpt.write(f"  GT Present:  {len(gt_present)}   GT Absent: {len(gt_absent)}\n")
        rpt.write("=" * 70 + "\n\n")

        rpt.write("── Distance Statistics ──\n")
        rpt.write(f"  Genuine  (same person):      min={genuine_distances.min():.4f}  "
                  f"max={genuine_distances.max():.4f}  mean={genuine_distances.mean():.4f}  "
                  f"median={np.median(genuine_distances):.4f}  std={genuine_distances.std():.4f}\n")
        rpt.write(f"  Impostor (different person): min={impostor_distances.min():.4f}  "
                  f"max={impostor_distances.max():.4f}  mean={impostor_distances.mean():.4f}  "
                  f"median={np.median(impostor_distances):.4f}  std={impostor_distances.std():.4f}\n\n")

        rpt.write("── Key Thresholds ──\n")
        rpt.write(f"  Current threshold:       0.400\n")
        rpt.write(f"  EER threshold:           {eer_threshold:.4f}  (EER = {eer_value:.4f})\n")
        rpt.write(f"  Youden-optimal:          {recommended_threshold:.4f}  (FAR={rec_far:.4f}, FRR={rec_frr:.4f})\n")
        rpt.write(f"  Target FAR ≤ {args.target_far:.2%}:    {target_threshold:.4f}  (actual FAR={target_far:.4f}, FRR={target_frr:.4f})\n\n")

        rpt.write("  ★ RECOMMENDED THRESHOLD: {:.4f}\n\n".format(recommended_threshold))

        rpt.write("── Attendance Simulation ──\n")
        rpt.write(f"  {'Threshold':>10}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}  "
                  f"{'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'Accuracy':>8}\n")
        rpt.write("  " + "-" * 68 + "\n")
        for r in sim_results:
            marker = " ◀" if abs(r["threshold"] - recommended_threshold) < 0.001 else ""
            rpt.write(f"  {r['threshold']:>10.4f}  {r['TP']:>4}  {r['FP']:>4}  "
                      f"{r['FN']:>4}  {r['TN']:>4}  "
                      f"{r['precision']:>9.4f}  {r['recall']:>7.4f}  "
                      f"{r['f1']:>7.4f}  {r['accuracy']:>8.4f}{marker}\n")

        rpt.write("\n── Per-Student Distances (sorted) ──\n")
        rpt.write(f"  {'Student':>35}  {'MinDist':>8}  {'GT':>8}  {'@Recommended':>12}\n")
        rpt.write("  " + "-" * 68 + "\n")
        for n, d in sorted(enrolled_min_dist.items(), key=lambda x: x[1]):
            gt_status = "PRESENT" if n in gt_present else "ABSENT"
            pred = "Accept" if d <= recommended_threshold else "Reject"
            correct = (gt_status == "PRESENT" and pred == "Accept") or \
                      (gt_status == "ABSENT"  and pred == "Reject")
            mark = "✓" if correct else "✗"
            rpt.write(f"  {n:>35}  {d:>8.4f}  {gt_status:>8}  {pred:>7}  {mark}\n")

    print(f'THRESHOLD = {recommended_threshold:.4f}')
    print("  Attendance simulation at different thresholds:")
    print(f"  {'Thresh':>7}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Acc':>6}")
    print("  " + "-" * 56)
    for r in sim_results:
        m = " ◀" if abs(r["threshold"] - recommended_threshold) < 0.001 else ""
        print(f"  {r['threshold']:>7.3f}  {r['TP']:>4}  {r['FP']:>4}  "
              f"{r['FN']:>4}  {r['TN']:>4}  {r['precision']:>6.3f}  "
              f"{r['recall']:>6.3f}  {r['f1']:>6.3f}  {r['accuracy']:>6.3f}{m}")

    print(f"\n  Plots saved to: {args.output_dir}/")
    print("=" * 70)

    return recommended_threshold


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate face recognition threshold on Indian face data."
    )
    parser.add_argument(
        "--test-dir", default=DEFAULT_TEST_DIR,
        help=f"Directory containing labelled test images (default: {DEFAULT_TEST_DIR})"
    )
    parser.add_argument(
        "--embeddings", default=DEFAULT_EMBEDDINGS,
        help=f"Path to enrolled embeddings .pkl (default: {DEFAULT_EMBEDDINGS})"
    )
    parser.add_argument(
        "--ground-truth", default=DEFAULT_GROUND_TRUTH,
        help=f"CSV with ground-truth attendance (default: {DEFAULT_GROUND_TRUTH})"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for calibration outputs (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--target-far", type=float, default=DEFAULT_TARGET_FAR,
        help=f"Target maximum FAR (default: {DEFAULT_TARGET_FAR})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_calibration(args)
