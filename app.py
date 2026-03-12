import streamlit as st
import os
import pandas as pd
from PIL import Image
import shutil
from mark_attendance import mark_attendance
from generate_embeddings import generate_embeddings
import time

st.set_page_config(page_title="Automated Attendance System", layout="wide")

st.title("Automated Classroom Attendance System")
st.markdown("Upload classroom images to automatically mark student attendance using face recognition.")

DATASET_DIR = "course_project_dataset"
EMBEDDINGS_FILE = "embeddings/embeddings_dl.pkl"
OUTPUT_DIR = "output"
UPLOAD_DIR = "uploads"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.sidebar.header("System Setup")

if st.sidebar.button("Load Dataset & Generate Embeddings"):
    if not os.path.exists(DATASET_DIR):
        st.sidebar.error(f"Dataset directory '{DATASET_DIR}' not found!")
    else:
        with st.spinner("Generating embeddings... This may take a while depending on the dataset size."):
            try:
                generate_embeddings(DATASET_DIR, EMBEDDINGS_FILE)
                st.sidebar.success("Embeddings generated successfully!")
            except Exception as e:
                st.sidebar.error(f"Error generating embeddings: {e}")

if os.path.exists(EMBEDDINGS_FILE):
    st.sidebar.info("Embeddings file exists. System is ready to mark attendance.")
else:
    st.sidebar.warning("No embeddings found. Please generate embeddings first.")

st.header("Evaluating Attendance")

uploaded_files = st.file_uploader("Upload Classroom Images (up to 5)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.error("Please upload a maximum of 5 images.")
    else:
        if st.button("Process Images & Mark Attendance"):
            if not os.path.exists(EMBEDDINGS_FILE):
                st.error("Please generate embeddings first before marking attendance.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_present_students = set()
                all_unknown_crops = []
                annotated_images = []
                
                if os.path.exists(OUTPUT_DIR):
                    shutil.rmtree(OUTPUT_DIR)
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                unknown_dir = os.path.join(OUTPUT_DIR, "unknown_faces")
                os.makedirs(unknown_dir, exist_ok=True)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing image {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}...")
                    
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    results = mark_attendance(file_path, EMBEDDINGS_FILE, OUTPUT_DIR)
                    if results:
                        all_present_students.update(results['present'])
                        annotated_images.append(results['annotated_image_path'])
                        
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("Processing complete! Generating final report...")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
                import pickle
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    data = pickle.load(f)
                all_students = sorted(list(set(data["names"])))
                
                final_attendance = []
                for student in all_students:
                    status = "Present" if student in all_present_students else "Absent"
                    final_attendance.append({"Student Name": student, "Status": status})
                
                df_final = pd.DataFrame(final_attendance)
                final_csv_path = os.path.join(OUTPUT_DIR, "final_attendance.csv")
                df_final.to_csv(final_csv_path, index=False)
                
                tab1, tab2, tab3 = st.tabs(["Attendance Report", "Classroom Images", "Unknown Faces"])
                
                with tab1:
                    st.subheader("Final Attendance")
                    st.dataframe(df_final, use_container_width=True)
                    
                    with open(final_csv_path, "r") as f:
                        st.download_button(
                            label="Download Attendance CSV",
                            data=f.read(),
                            file_name="final_attendance.csv",
                            mime="text/csv"
                        )
                        
                with tab2:
                    st.subheader("Annotated Images")
                    for img_path in annotated_images:
                        if os.path.exists(img_path):
                            st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
                            
                with tab3:
                    st.subheader("Unknown Individuals Detected")
                    unknown_crops = []
                    if os.path.exists(unknown_dir):
                        unknown_crops = [os.path.join(unknown_dir, f) for f in os.listdir(unknown_dir) if f.endswith(('.jpg', '.png'))]
                    
                    if not unknown_crops:
                        st.success("No unknown individuals detected in the classroom images.")
                    else:
                        st.warning(f"{len(unknown_crops)} unknown face(s) detected.")
                        cols = st.columns(5)
                        for i, crop_path in enumerate(unknown_crops):
                            img = Image.open(crop_path)
                            cols[i % 5].image(img, caption=f"Unknown {i+1}")
