import os
import cv2
import pickle
import numpy as np
import pandas as pd
from deepface import DeepFace

def cosine_distance(a, b):
    """Calculates the cosine distance between two vectors."""
    a = np.array(a)
    b = np.array(b)
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 - cos_sim

def mark_attendance(class_image_path, embeddings_file="embeddings.pkl", output_dir="output"):
    if not os.path.exists(embeddings_file):
        print(f"Error: {embeddings_file} not found. Please run the embedding script first.")
        return None

    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
        
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
    
    unique_students = sorted(list(set(known_face_names)))

    embedding_model = "Facenet512" 
    detector_backend = "retinaface"
    
    threshold = 0.30 

    annotated_image = cv2.imread(class_image_path)
    if annotated_image is None:
        print(f"Error loading image {class_image_path}")
        return None

    print("Detecting faces and extracting embeddings... This might take a moment.")
    
    try:
        extracted_faces = DeepFace.represent(
            img_path=class_image_path, 
            model_name=embedding_model, 
            detector_backend=detector_backend,
            enforce_detection=False 
        )
    except Exception as e:
        print(f"Error processing image with DeepFace: {e}")
        return None

    if not isinstance(extracted_faces, list):
        extracted_faces = [extracted_faces] if "embedding" in extracted_faces else []

    extracted_faces = [f for f in extracted_faces if f["facial_area"]["w"] > 0]

    print(f"Found {len(extracted_faces)} faces in the classroom image.")

    os.makedirs(output_dir, exist_ok=True)
    unknown_dir = os.path.join(output_dir, "unknown_faces")
    os.makedirs(unknown_dir, exist_ok=True)

    present_students = set()
    unknown_count = 0

    for face_data in extracted_faces:
        face_encoding = face_data["embedding"]
        facial_area = face_data["facial_area"]
        
        left = facial_area['x']
        top = facial_area['y']
        right = left + facial_area['w']
        bottom = top + facial_area['h']

        name = "Unknown"
        best_distance = float("inf")
        best_match_index = -1

        for i, known_encoding in enumerate(known_face_encodings):
            distance = cosine_distance(known_encoding, face_encoding)
            if distance < best_distance:
                best_distance = distance
                best_match_index = i

        if best_distance <= threshold and best_match_index != -1:
            name = known_face_names[best_match_index]
            present_students.add(name)

        if name == "Unknown":
            unknown_count += 1
            crop_top, crop_bottom = max(0, top), min(annotated_image.shape[0], bottom)
            crop_left, crop_right = max(0, left), min(annotated_image.shape[1], right)
            
            face_image_bgr = annotated_image[crop_top:crop_bottom, crop_left:crop_right]
            if face_image_bgr.size > 0:
                unknown_face_path = os.path.join(unknown_dir, f"unknown_{unknown_count}.jpg")
                cv2.imwrite(unknown_face_path, face_image_bgr)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(annotated_image, (left, top), (right, bottom), color, 2)

        cv2.rectangle(annotated_image, (left, bottom), (right, bottom + 35), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(annotated_image, name, (left + 6, bottom + 25), font, 0.7, (255, 255, 255), 1)

    base_name = os.path.basename(class_image_path)
    output_image_path = os.path.join(output_dir, f"annotated_{base_name}")
    cv2.imwrite(output_image_path, annotated_image)
    print(f"Saved annotated image to {output_image_path}")

    attendance_data = []
    for student in unique_students:
        status = "Present" if student in present_students else "Absent"
        attendance_data.append({"Student Name": student, "Status": status})

    df = pd.DataFrame(attendance_data)
    csv_path = os.path.join(output_dir, "attendance.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved attendance report to {csv_path}")

    return {
        "present": list(present_students),
        "absent": [s for s in unique_students if s not in present_students],
        "unknown_count": unknown_count,
        "annotated_image_path": output_image_path,
        "csv_path": csv_path
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        mark_attendance(img_path)
    else:
        print("Usage: python mark_attendance.py <path_to_class_image>")