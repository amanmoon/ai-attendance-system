import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import insightface
from insightface.app import FaceAnalysis

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

CLUSTERING_THRESHOLD = 0.4
THRESHOLD = 0.605
MIN_FACE_PX = 10
BBOX_PADDING = 0.30
DETECT_MAX_LONG_EDGE = None
USE_TILING    = True
TILE_SIZE      = 1500
TILE_OVERLAP   = 0.20

face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def cosine_distance(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def euclidean_distance(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.linalg.norm(a - b)

def resize_for_detection(image):

    if DETECT_MAX_LONG_EDGE is None:
        return image, 1.0

    h, w = image.shape[:2]
    long_edge = max(h, w)

    if long_edge <= DETECT_MAX_LONG_EDGE:
        return image, 1.0

    scale = DETECT_MAX_LONG_EDGE / long_edge
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def enhance_face_crop(face_bgr):

    denoised = cv2.bilateralFilter(face_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_eq  = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b_ch]), cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
    sharp   = cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)

    h, w = sharp.shape[:2]
    interp = cv2.INTER_AREA if (h > 160 or w > 160) else cv2.INTER_LINEAR
    return cv2.resize(sharp, (160, 160), interpolation=interp)


def non_max_suppression(boxes, scores, iou_threshold=0.4):

    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def tiled_face_detection(image, tile_size=1800, overlap=0.15):

    h, w = image.shape[:2]
    stride = int(tile_size * (1.0 - overlap))
    
    all_detections = []
    
    y_coords = list(range(0, max(1, h - tile_size + stride), stride))
    x_coords = list(range(0, max(1, w - tile_size + stride), stride))
    
    if y_coords and y_coords[-1] + tile_size < h: y_coords.append(h - tile_size)
    if x_coords and x_coords[-1] + tile_size < w: x_coords.append(w - tile_size)
    
    if not y_coords: y_coords = [0]
    if not x_coords: x_coords = [0]

    for ty in y_coords:
        for tx in x_coords:
            y_end = min(ty + tile_size, h)
            x_end = min(tx + tile_size, w)
            tile = image[ty:y_end, tx:x_end]

            try:
                faces = face_app.get(tile)
            except Exception:
                faces = []

            if not faces:
                continue
                
            for face in faces:
                if face.det_score < 0.1:
                    continue

                bbox = face.bbox.copy()
                bbox[0] += tx
                bbox[1] += ty
                bbox[2] += tx
                bbox[3] += ty

                all_detections.append({
                    "bbox": bbox,
                    "confidence": float(face.det_score),
                    "embedding": face.embedding,
                })

    if not all_detections:
        return []

    boxes = []
    scores = []
    for det in all_detections:
        b = det["bbox"]
        boxes.append([b[0], b[1], b[2] - b[0], b[3] - b[1]])
        scores.append(det["confidence"])
        
    keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.3)
    
    return [all_detections[i] for i in keep_indices]


def cluster_faces(all_detections):
    clusters = []
    for det in all_detections:
        matched_cluster = None
        for cluster in clusters:
            rep_embedding = cluster[0]['embedding']
            if cosine_distance(rep_embedding, det['embedding']) <= CLUSTERING_THRESHOLD:
                matched_cluster = cluster
                break
        if matched_cluster:
            matched_cluster.append(det)
        else:
            clusters.append([det])
    return clusters


def mark_attendance(class_image_paths, embeddings_file="embeddings.pkl", output_dir="output"):
    if isinstance(class_image_paths, str):
        class_image_paths = [class_image_paths]

    with open(embeddings_file, "rb") as f:
        data = pickle.load(f)

    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
    unique_students = sorted(set(known_face_names))

    all_detections = []
    original_images = []

    # Detection
    for img_idx, img_path in enumerate(class_image_paths):
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"Error loading image: {img_path}")
            original_images.append(None)
            continue
        
        original_images.append(original_image)
        h0, w0 = original_image.shape[:2]
        detect_image, det_scale = resize_for_detection(original_image)
        
        if USE_TILING:
            detected = tiled_face_detection(detect_image, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
        else:
            try:
                faces = face_app.get(detect_image)
                detected = [{"bbox": face.bbox.copy(), "confidence": float(face.det_score), "embedding": face.embedding} for face in faces if face.det_score >= 0.1]
            except Exception as e:
                detected = []

        if not isinstance(detected, list):
            detected = [detected] if isinstance(detected, dict) else []

        scale_inv = 1.0 / det_scale
        for det in detected:
            bbox = det["bbox"]
            x = int(bbox[0] * scale_inv)
            y = int(bbox[1] * scale_inv)
            w = int((bbox[2] - bbox[0]) * scale_inv)
            h = int((bbox[3] - bbox[1]) * scale_inv)

            if w < MIN_FACE_PX or h < MIN_FACE_PX:
                continue

            pad_x = int(w * BBOX_PADDING)
            pad_y = int(h * BBOX_PADDING)
            x_pad = max(0, x - pad_x)
            y_pad = max(0, y - pad_y)
            x2_pad = min(w0, x + w + pad_x)
            y2_pad = min(h0, y + h + pad_y)

            crop = original_image[y_pad:y2_pad, x_pad:x2_pad]
            if crop.size == 0:
                continue

            all_detections.append({
                "img_idx": img_idx,
                "crop": crop,
                "bbox": (x, y, w, h),
                "confidence": det.get("confidence", 0.0),
                "embedding": det["embedding"],
            })

    # Clustering
    clusters = cluster_faces(all_detections)

    os.makedirs(output_dir, exist_ok=True)
    unknown_dir = os.path.join(output_dir, "unknown_faces")
    identified_dir = os.path.join(output_dir, "identified_faces")
    os.makedirs(unknown_dir, exist_ok=True)
    os.makedirs(identified_dir, exist_ok=True)

    present_students = set()
    unknown_count = 0
    import uuid

    # Matching
    for cluster in clusters:
        rep = cluster[0]
        face_encoding = rep["embedding"]

        best_distance = float("inf")
        best_match_index = -1
        if face_encoding is not None:
            for i, known_enc in enumerate(known_face_encodings):
                dist = cosine_distance(known_enc, face_encoding)
                if dist < best_distance:
                    best_distance = dist
                    best_match_index = i

        closest_name = known_face_names[best_match_index] if best_match_index != -1 else "None"

        if best_distance <= THRESHOLD and best_match_index != -1:
            name = closest_name
            present_students.add(name)
            person_dir = os.path.join(identified_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            for j, c in enumerate(cluster):
                cv2.imwrite(os.path.join(person_dir, f"{j}.png"), c["crop"])
        else:
            name = "Unknown"
            unknown_count += 1
            uid = uuid.uuid4().hex[:8]
            person_dir = os.path.join(unknown_dir, f"unknown_{uid}_{closest_name}")
            os.makedirs(person_dir, exist_ok=True)
            for j, c in enumerate(cluster):
                cv2.imwrite(os.path.join(person_dir, f"{j}.png"), c["crop"])

        for det in cluster:
            det["label"] = name if name != "Unknown" else f"Unknown ({closest_name})"
            det["color"] = (0, 200, 80) if name != "Unknown" else (0, 60, 220)

    # Drawing
    annotated_image_paths = []
    for img_idx, original_image in enumerate(original_images):
        if original_image is None:
            continue
        annotated_image = original_image.copy()
        
        for det in [d for d in all_detections if d["img_idx"] == img_idx]:
            x, y, w, h = det["bbox"]
            left, top = x, y
            right, bottom = x + w, y + h
            color = det["color"]
            label = det["label"]
            
            cv2.rectangle(annotated_image, (left, top), (right, bottom), color, 2)
            cv2.rectangle(annotated_image, (left, bottom), (right, bottom + 35), color, cv2.FILLED)
            cv2.putText(annotated_image, label, (left + 6, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        base_name = os.path.basename(class_image_paths[img_idx])
        out_img_path = os.path.join(output_dir, f"annotated_{base_name}")
        cv2.imwrite(out_img_path, annotated_image)
        annotated_image_paths.append(out_img_path)

    attendance_data = [{"Student Name": s, "Status": "Present" if s in present_students else "Absent"} for s in unique_students]
    df = pd.DataFrame(attendance_data)
    df["_sort"] = df["Status"].map({"Present": 0, "Absent": 1})
    df = df.sort_values(["_sort", "Student Name"]).drop(columns="_sort").reset_index(drop=True)
    csv_path = os.path.join(output_dir, "attendance.csv")
    df.to_csv(csv_path, index=False)

    result = {
        "present": list(present_students),
        "absent": [s for s in unique_students if s not in present_students],
        "unknown_count": unknown_count,
        "annotated_image_paths": annotated_image_paths,
        "csv_path": csv_path,
        "identified_dir": identified_dir,
        "unknown_dir": unknown_dir
    }

    print(f"\nSummary — Present: {len(present_students)}  Absent: {len(result['absent'])}  Unknown: {unknown_count}")
    return result

if __name__ == "__main__":
    mark_attendance(sys.argv[1:])