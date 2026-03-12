import os
import pickle
from tqdm import tqdm
from deepface import DeepFace

dataset_dir="course_project_dataset"
output_file="embeddings_dl.pkl"
embedding_model = "Facenet512" 
detector_backend = "retinaface" 
known_face_encodings = []
known_face_names = []

students = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

for student_name in tqdm(students):
    student_dir = os.path.join(dataset_dir, student_name)
    
    for filename in os.listdir(student_dir):

        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(student_dir, filename)
            try:
                results = DeepFace.represent(
                    img_path=image_path, 
                    model_name=embedding_model, 
                    detector_backend=detector_backend,
                    enforce_detection=True 
                )
                
                if len(results) > 0:
                    embedding = results[0]["embedding"]
                    known_face_encodings.append(embedding)
                    known_face_names.append(student_name)
                    
            except ValueError or Exception as e:
                print("Error processing ", image_path)
        else:
            print("Skipping non-image file", filename)

with open(output_file, 'wb') as f:
    data = {"encodings": known_face_encodings, "names": known_face_names}
    pickle.dump(data, f)
