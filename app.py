import streamlit as st
import os
import pandas as pd
from PIL import Image, ImageOps
import shutil
from mark_attendance import mark_attendance
import time
from supabase import create_client, Client

def mark_present_callback(crop_path, selected_student):
    if not os.path.exists(crop_path):
        return
    res = st.session_state.results
    new_name = f"{selected_student}.png"
    target_path = os.path.join(res['identified_dir'], new_name)
    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(crop_path, target_path)
    
    idx = res['df_final'].index[res['df_final']['Student Name'] == selected_student].tolist()
    if idx:
        if res['df_final'].at[idx[0], 'Status'] == 'Absent':
            res['df_final'].at[idx[0], 'Status'] = 'Present'
            res['n_present'] += 1
            res['n_absent'] -= 1
            
    res['n_unknown'] -= 1
    res['df_final']["_sort"] = res['df_final']["Status"].map({"Present": 0, "Absent": 1})
    res['df_final'] = res['df_final'].sort_values(["_sort", "Student Name"]).drop(columns="_sort").reset_index(drop=True)
    res['df_final'].to_csv(res['final_csv_path'], index=False)
    st.session_state.results = res

def unmark_present_callback(crop_path, student_name):
    if not os.path.exists(crop_path):
        return
    res = st.session_state.results
    ts = int(time.time() * 1000)
    new_name = f"unknown_{ts}_{student_name}.png"
    target_path = os.path.join(res['unknown_dir'], new_name)
    os.rename(crop_path, target_path)
    
    idx = res['df_final'].index[res['df_final']['Student Name'] == student_name].tolist()
    if idx:
        if res['df_final'].at[idx[0], 'Status'] == 'Present':
            res['df_final'].at[idx[0], 'Status'] = 'Absent'
            res['n_present'] -= 1
            res['n_absent'] += 1
            
    res['n_unknown'] += 1
    res['df_final']["_sort"] = res['df_final']["Status"].map({"Present": 0, "Absent": 1})
    res['df_final'] = res['df_final'].sort_values(["_sort", "Student Name"]).drop(columns="_sort").reset_index(drop=True)
    res['df_final'].to_csv(res['final_csv_path'], index=False)
    st.session_state.results = res

def get_reference_image(student_name, dataset_dir):
    student_path = os.path.join(dataset_dir, student_name)
    if os.path.exists(student_path):
        images = [f for f in os.listdir(student_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            return os.path.join(student_path, sorted(images)[0])
    return None

SUPABASE_URL = 'https://jjmqimjinuykhusquyxd.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpqbXFpbWppbnV5a2h1c3F1eXhkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQxODE4ODgsImV4cCI6MjA4OTc1Nzg4OH0.xiv1q7xuOM6hSM2KCsfQP2mjU9iyvabMp2DhRVhuVT8'
Client = create_client(SUPABASE_URL, SUPABASE_KEY) 
st.set_page_config(page_title="Automated Attendance System", layout="wide")

st.title("Automated Classroom Attendance System")
st.markdown("Upload classroom images to automatically mark student attendance using face recognition.")

DATASET_DIR = "course_project_dataset"
EMBEDDINGS_FILE = "embeddings/embeddings.pkl"
OUTPUT_DIR = "output"
UPLOAD_DIR = "uploads"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.sidebar.header("System Setup")

if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None

st.sidebar.subheader("Cloud Database Sync")
if st.sidebar.button("Download Embeddings from Cloud"):
    try:
        res = Client.storage.from_("embeddings").download("embeddings_dl.pkl")
        with open(EMBEDDINGS_FILE, 'wb') as f:
            f.write(res)
        st.sidebar.success("Successfully downloaded embeddings from cloud!")
    except Exception as e:
        st.sidebar.error(f"Failed to download: {e}")

if st.sidebar.button("Upload Embeddings to Cloud"):
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                Client.storage.from_("embeddings").upload("embeddings_dl.pkl", f, {"upsert": "true"})
            st.sidebar.success("Successfully updated cloud embeddings!")
        except Exception as e:
            st.sidebar.error(f"Failed to upload: {e}")
    else:
        st.sidebar.warning("No local embeddings found to upload.")

if st.sidebar.button("Load Dataset & Generate Embeddings"):
    if not os.path.exists(DATASET_DIR):
        st.sidebar.error(f"Dataset directory '{DATASET_DIR}' not found!")

if os.path.exists(EMBEDDINGS_FILE):
    st.sidebar.info("Embeddings file exists. System is ready to mark attendance.")
else:
    st.sidebar.warning("No embeddings found. Please generate embeddings first.")

st.header("Evaluating Attendance")

uploaded_files = st.file_uploader("Upload Classroom Images (up to 5)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    if not st.session_state.processing:
        if st.button("Process Images & Mark Attendance"):
            if not os.path.exists(EMBEDDINGS_FILE):
                st.error("Please generate embeddings first before marking attendance.")
            else:
                st.session_state.results = None
                st.session_state.processing = True
                st.rerun()
    else:
        if st.button("Stop Processing", type="primary"):
            st.session_state.processing = False
            st.rerun()

    if st.session_state.processing:
        if not os.path.exists(EMBEDDINGS_FILE):
             st.error("Please generate embeddings first before marking attendance.")
             st.session_state.processing = False
             st.rerun()
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
            identified_dir = os.path.join(OUTPUT_DIR, "identified_faces")
            os.makedirs(unknown_dir, exist_ok=True)
            os.makedirs(identified_dir, exist_ok=True)
            
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

            df_final["_sort"] = df_final["Status"].map({"Present": 0, "Absent": 1})
            df_final = df_final.sort_values(["_sort", "Student Name"]).drop(columns="_sort").reset_index(drop=True)
            final_csv_path = os.path.join(OUTPUT_DIR, "final_attendance.csv")
            df_final.to_csv(final_csv_path, index=False)
            
            n_present = len(all_present_students)
            n_total   = len(all_students)
            n_absent  = n_total - n_present
            n_unknown = sum(
                1 for f in os.listdir(unknown_dir)
                if f.endswith(('.jpg', '.png'))
            ) if os.path.exists(unknown_dir) else 0

            st.session_state.results = {
                "n_present": n_present,
                "n_absent": n_absent,
                "n_unknown": n_unknown,
                "n_total": n_total,
                "df_final": df_final,
                "final_csv_path": final_csv_path,
                "annotated_images": annotated_images,
                "identified_dir": identified_dir,
                "unknown_dir": unknown_dir
            }
            
            st.session_state.processing = False
            st.rerun()

if st.session_state.results:
    res = st.session_state.results
    st.markdown("### Attendance Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Present",  res["n_present"],  delta=None)
    c2.metric("Absent",   res["n_absent"],   delta=None)
    c3.metric("Unknown",  res["n_unknown"],  delta=None)
    c4.metric("Total",    res["n_total"],    delta=None)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Attendance Report", "Classroom Images", "Identified Faces", "Unknown Faces"])

    with tab1:
        st.subheader("Final Attendance")
        st.dataframe(res["df_final"], width='stretch')
        
        if os.path.exists(res["final_csv_path"]):
            with open(res["final_csv_path"], "r") as f:
                st.download_button(
                    label="Download Attendance CSV",
                    data=f.read(),
                    file_name="final_attendance.csv",
                    mime="text/csv"
                )

    with tab2:
        st.subheader("Annotated Images")
        for img_path in res["annotated_images"]:
            if os.path.exists(img_path):
                st.image(img_path, caption=os.path.basename(img_path), width='content')

    with tab3:
        st.subheader("Successfully Identified Faces")
        identified_crops = []
        if os.path.exists(res["identified_dir"]):
            identified_crops = [os.path.join(res["identified_dir"], f) for f in os.listdir(res["identified_dir"]) if f.endswith(('.jpg', '.png'))]
        
        if not identified_crops:
            st.info("No identified faces to display.")
        else:
            st.success(f"{len(identified_crops)} student(s) identified.")
            cols = st.columns(3)
            for i, crop_path in enumerate(identified_crops):
                with cols[i % 3]:
                    student_name = os.path.splitext(os.path.basename(crop_path))[0]
                    ref_image_path = get_reference_image(student_name, DATASET_DIR)
                    
                    head_col1, head_col2 = st.columns(2, vertical_alignment="center")
                    with head_col1:
                        st.write(f"**{student_name}**")
                    with head_col2:
                        if st.button("Unmark", key=f"unmark_{crop_path}", width='stretch'):
                            unmark_present_callback(crop_path, student_name)
                            st.rerun()
                            
                    c1, c2 = st.columns(2)
                    c1.image(ImageOps.exif_transpose(Image.open(crop_path)).resize((200, 200)), caption="Image from Classroom", width='stretch')
                    if ref_image_path:
                        c2.image(ImageOps.exif_transpose(Image.open(ref_image_path)).resize((200, 200)), caption="Original Submitted Image", width='stretch')
                    else:
                        c2.info("No reference found")
                        
                    st.divider()

    with tab4:
        st.subheader("Unknown Individuals Detected")
        unknown_crops = []
        if os.path.exists(res["unknown_dir"]):
            unknown_crops = [os.path.join(res["unknown_dir"], f) for f in os.listdir(res["unknown_dir"]) if f.endswith(('.jpg', '.png'))]
        
        if not unknown_crops:
            st.success("No unknown individuals detected in the classroom images.")
        else:
            st.warning(f"{len(unknown_crops)} unknown face(s) detected.")
            cols = st.columns(3)
            for i, crop_path in enumerate(unknown_crops):
                with cols[i % 3]:
                    basename = os.path.splitext(os.path.basename(crop_path))[0]
                    parts = basename.split('_')
                    closest_name = parts[-1] if len(parts) > 2 else "None"
                    ref_image_path = get_reference_image(closest_name, DATASET_DIR)
                    
                    st.write(f"**Unknown** (Nearest: {closest_name})")
                    c1, c2 = st.columns(2)
                    c1.image(ImageOps.exif_transpose(Image.open(crop_path)).resize((200, 200)), caption="Image from Classroom", width='stretch')
                    if ref_image_path:
                        c2.image(ImageOps.exif_transpose(Image.open(ref_image_path)).resize((200, 200)), caption="Original Submitted Image", width='stretch')
                    else:
                        c2.info("No reference found")
                        
                    all_students = res['df_final']['Student Name'].tolist()
                    default_idx = all_students.index(closest_name) if closest_name in all_students else 0
                    
                    action_c1, action_c2 = st.columns(2)
                    with action_c1:
                        selected_student = st.selectbox("Select Student", all_students, index=default_idx, key=f"select_{crop_path}", label_visibility="collapsed")
                    with action_c2:
                        if st.button("Mark", width='stretch', key=f"mark_{crop_path}"):
                            mark_present_callback(crop_path, selected_student)
                            st.rerun()
                        
                    st.divider()

