import os
import cv2
import pickle
import tempfile
import numpy as np
import pandas as pd
from deepface import DeepFace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

EMBEDDING_MODEL  = "Facenet512"
DETECTOR_BACKEND = "retinaface"
THRESHOLD = 0.40
MIN_FACE_PX = 10
BBOX_PADDING = 0.20
DETECT_MAX_LONG_EDGE = None
USE_TILING    = True
TILE_SIZE      = 1800
TILE_OVERLAP   = 0.15

def cosine_distance(a, b) -> float:
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


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


def tiled_face_detection(image, tile_size = 1800, overlap = 0.15):

    h, w = image.shape[:2]
    stride = int(tile_size * (1.0 - overlap))
    
    all_detections = []
    
    y_coords = list(range(0, max(1, h - tile_size + stride), stride))
    x_coords = list(range(0, max(1, w - tile_size + stride), stride))
    
    if y_coords and y_coords[-1] + tile_size < h: y_coords.append(h - tile_size)
    if x_coords and x_coords[-1] + tile_size < w: x_coords.append(w - tile_size)
    
    if not y_coords: y_coords = [0]
    if not x_coords: x_coords = [0]

    num_tiles = len(x_coords) * len(y_coords)

    for ty in y_coords:
        for tx in x_coords:
            y_end = min(ty + tile_size, h)
            x_end = min(tx + tile_size, w)
            tile = image[ty:y_end, tx:x_end]
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, tile)
            
            try:
                detections = DeepFace.extract_faces(
                    img_path=tmp_path,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=True,
                )
            except Exception:
                detections = []
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            if not detections:
                continue
                
            for det in detections:
                if det.get("confidence", 0.0) < 0.1:
                    continue
                    
                fa = det["facial_area"]
                fa["x"] += tx
                fa["y"] += ty
                all_detections.append(det)

    if not all_detections:
        return []

    boxes = []
    scores = []
    for det in all_detections:
        fa = det["facial_area"]
        boxes.append([fa["x"], fa["y"], fa["w"], fa["h"]])
        scores.append(det.get("confidence", 1.0))
        
    keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.3)
    
    return [all_detections[i] for i in keep_indices]


def mark_attendance(class_image_path, embeddings_file = "embeddings_dl.pkl", output_dir = "output"):

    with open(embeddings_file, "rb") as f:
        data = pickle.load(f)

    known_face_encodings = data["encodings"]
    known_face_names     = data["names"]
    unique_students      = sorted(set(known_face_names))

    original_image = cv2.imread(class_image_path)
    if original_image is None:
        print(f"Error loading image: {class_image_path}")
        return None

    h0, w0 = original_image.shape[:2]
    detect_image, det_scale = resize_for_detection(original_image)
    dh, dw = detect_image.shape[:2]
    
    if USE_TILING:
        detected = tiled_face_detection(
            detect_image, 
            tile_size=TILE_SIZE, 
            overlap=TILE_OVERLAP
        )
    else:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, detect_image)
        try:
            detected = DeepFace.extract_faces(
                img_path=tmp_path,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True,
            )
        except Exception as e:
            os.remove(tmp_path)
            return None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not isinstance(detected, list):
        detected = [detected] if isinstance(detected, dict) else []

    scale_inv = 1.0 / det_scale
    valid_detections = []
    for det in detected:
        fa = det["facial_area"]

        x = int(fa["x"] * scale_inv)
        y = int(fa["y"] * scale_inv)
        w = int(fa["w"] * scale_inv)
        h = int(fa["h"] * scale_inv)

        if w < MIN_FACE_PX or h < MIN_FACE_PX:
            continue

        pad_x = int(w * BBOX_PADDING)
        pad_y = int(h * BBOX_PADDING)
        x_pad  = max(0, x - pad_x)
        y_pad  = max(0, y - pad_y)
        x2_pad = min(w0, x + w + pad_x)
        y2_pad = min(h0, y + h + pad_y)

        crop = original_image[y_pad:y2_pad, x_pad:x2_pad]
        if crop.size == 0:
            continue

        valid_detections.append({
            "crop": crop,
            "bbox": (x, y, w, h),
            "confidence": det.get("confidence", 0.0),
        })

    os.makedirs(output_dir, exist_ok=True)
    unknown_dir = os.path.join(output_dir, "unknown_faces")
    os.makedirs(unknown_dir, exist_ok=True)

    annotated_image  = original_image.copy()
    present_students: set[str] = set()
    unknown_count = 0

    for idx, det in enumerate(valid_detections):
        crop = det["crop"]
        x, y, w, h = det["bbox"]

        enhanced_crop = enhance_face_crop(crop)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as ftmp:
            face_tmp = ftmp.name
        cv2.imwrite(face_tmp, enhanced_crop)

        try:
            results = DeepFace.represent(
                img_path=face_tmp,
                model_name=EMBEDDING_MODEL,
                detector_backend="skip",
                enforce_detection=False,
            )
            face_encoding = results[0]["embedding"] if results else None
        except Exception as e:
            print(f"embedding error ({e})")
            face_encoding = None
        finally:
            if os.path.exists(face_tmp):
                os.remove(face_tmp)

        if face_encoding is None:
            continue

        best_distance    = float("inf")
        best_match_index = -1
        for i, known_enc in enumerate(known_face_encodings):
            dist = cosine_distance(known_enc, face_encoding)
            if dist < best_distance:
                best_distance    = dist
                best_match_index = i

        if best_distance <= THRESHOLD and best_match_index != -1:
            name = known_face_names[best_match_index]
            present_students.add(name)
            print(f"Face {name}  (dist={best_distance:.3f})")
        else:
            name = "Unknown"
            unknown_count += 1
            cv2.imwrite(
                os.path.join(unknown_dir, f"unknown_{unknown_count}.png"),
                enhanced_crop,
            )
            print(f"Face {idx+1}: Unknown  (best_dist={best_distance:.3f})")

        left, top     = x, y
        right, bottom = x + w, y + h
        color = (0, 200, 80) if name != "Unknown" else (0, 60, 220)
        cv2.rectangle(annotated_image, (left, top), (right, bottom), color, 2)
        cv2.rectangle(annotated_image, (left, bottom), (right, bottom + 35),
                      color, cv2.FILLED)
        cv2.putText(
            annotated_image, name,
            (left + 6, bottom + 25),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1,
        )

    out_img_path = os.path.join(
        output_dir, f"annotated_{os.path.basename(class_image_path)}")
    cv2.imwrite(out_img_path, annotated_image)
    attendance_data = [
        {"Student Name": s, "Status": "Present" if s in present_students else "Absent"}
        for s in unique_students
    ]
    df = pd.DataFrame(attendance_data)
    df["_sort"] = df["Status"].map({"Present": 0, "Absent": 1})
    df = df.sort_values(["_sort", "Student Name"]).drop(columns="_sort").reset_index(drop=True)
    csv_path = os.path.join(output_dir, "attendance.csv")
    df.to_csv(csv_path, index=False)

    result = {
        "present":              list(present_students),
        "absent":               [s for s in unique_students if s not in present_students],
        "unknown_count":        unknown_count,
        "annotated_image_path": out_img_path,
        "csv_path":             csv_path,
    }

    print(f"\nSummary — Present: {len(present_students)}  "
          f"Absent: {len(result['absent'])}  Unknown: {unknown_count}")
    return result


if __name__ == "__main__":
    mark_attendance(sys.argv[1])