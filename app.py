import streamlit as st
import os
import pandas as pd
from PIL import Image, ImageOps
import shutil
from mark_attendance import mark_attendance, cluster_faces
import time

def mark_present_callback(person_dir, selected_student):
    if not os.path.exists(person_dir):
        return
    res = st.session_state.results
    new_name = selected_student
    target_path = os.path.join(res['identified_dir'], new_name)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.rename(person_dir, target_path)
    
    idx = res['df_final'].index[res['df_final']['Student Name'] == selected_student].tolist()
    if idx:
        if res['df_final'].at[idx[0], 'Status'] == 'A':
            res['df_final'].at[idx[0], 'Status'] = 'P'
            res['n_present'] += 1
            res['n_absent'] -= 1
            
    res['n_unknown'] -= 1
    res['df_final'] = res['df_final'].sort_values("Student Name").reset_index(drop=True)
    res['df_final'].to_csv(res['final_csv_path'], index=False)
    st.session_state.results = res

def unmark_present_callback(person_dir, student_name):
    if not os.path.exists(person_dir):
        return
    res = st.session_state.results
    ts = int(time.time() * 1000)
    new_name = f"unknown_{ts}_{student_name}"
    target_path = os.path.join(res['unknown_dir'], new_name)
    os.rename(person_dir, target_path)
    
    idx = res['df_final'].index[res['df_final']['Student Name'] == student_name].tolist()
    if idx:
        if res['df_final'].at[idx[0], 'Status'] == 'P':
            res['df_final'].at[idx[0], 'Status'] = 'A'
            res['n_present'] -= 1
            res['n_absent'] += 1
            
    res['n_unknown'] += 1
    res['df_final'] = res['df_final'].sort_values("Student Name").reset_index(drop=True)
    res['df_final'].to_csv(res['final_csv_path'], index=False)
    st.session_state.results = res

def get_reference_image(student_name, dataset_dir):
    student_path = os.path.join(dataset_dir, student_name)
    if os.path.exists(student_path):
        images = [f for f in os.listdir(student_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            return os.path.join(student_path, sorted(images)[0])
    return None

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
            annotated_images = []
            
            if os.path.exists(OUTPUT_DIR):
                shutil.rmtree(OUTPUT_DIR)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            unknown_dir = os.path.join(OUTPUT_DIR, "unknown_faces")
            identified_dir = os.path.join(OUTPUT_DIR, "identified_faces")
            os.makedirs(unknown_dir, exist_ok=True)
            os.makedirs(identified_dir, exist_ok=True)
            
            file_paths = []
            for idx, uploaded_file in enumerate(uploaded_files):
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
                progress_bar.progress(0.1 + 0.4 * (idx + 1) / len(uploaded_files))
                
            status_text.text("Detecting faces, clustering, and marking attendance across all images (this may take a while)...")
            
            results = mark_attendance(file_paths, EMBEDDINGS_FILE, OUTPUT_DIR)
            progress_bar.progress(0.9)
            if results:
                all_present_students.update(results['present'])
                annotated_images.extend(results.get('annotated_image_paths', []))
            
            status_text.text("Processing complete! Generating final report...")
            progress_bar.progress(1.0)
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            import pickle
            with open(EMBEDDINGS_FILE, 'rb') as f:
                data = pickle.load(f)
            all_students = sorted(list(set(data["names"])))
            
            final_attendance = []
            for student in all_students:
                status = "P" if student in all_present_students else "A"
                final_attendance.append({"Student Name": student, "Status": status})
            
            df_final = pd.DataFrame(final_attendance)

            df_final = df_final.sort_values("Student Name").reset_index(drop=True)
            final_csv_path = os.path.join(OUTPUT_DIR, "final_attendance.csv")
            df_final.to_csv(final_csv_path, index=False)
            
            n_present = len(all_present_students)
            n_total   = len(all_students)
            n_absent  = n_total - n_present
            n_unknown = sum(
                1 for f in os.listdir(unknown_dir)
                if os.path.isdir(os.path.join(unknown_dir, f))
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
        identified_persons = []
        if os.path.exists(res["identified_dir"]):
            identified_persons = [os.path.join(res["identified_dir"], f) for f in os.listdir(res["identified_dir"]) if os.path.isdir(os.path.join(res["identified_dir"], f))]
        
        if not identified_persons:
            st.info("No identified faces to display.")
        else:
            st.success(f"{len(identified_persons)} student(s) identified.")
            for i in range(0, len(identified_persons), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(identified_persons):
                        person_dir = identified_persons[i + j]
                        with cols[j]:
                            with st.container(border=True):
                                student_name = os.path.basename(person_dir)
                                person_crops = sorted([os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))])
                                main_crop = person_crops[0] if person_crops else None
                                if not main_crop:
                                    continue
                                    
                                ref_image_path = get_reference_image(student_name, DATASET_DIR)
                                
                                head_col1, head_col2 = st.columns(2, vertical_alignment="center")
                                with head_col1:
                                    st.write(f"**{student_name}**")
                                with head_col2:
                                    if st.button("Unmark", key=f"unmark_{person_dir}", width='stretch'):
                                        unmark_present_callback(person_dir, student_name)
                                        st.rerun()
                                        
                                c1, c2 = st.columns(2)
                                c1.image(ImageOps.exif_transpose(Image.open(main_crop)).resize((200, 200)), caption="Image from Classroom", width='stretch')
                                if ref_image_path:
                                    c2.image(ImageOps.exif_transpose(Image.open(ref_image_path)).resize((200, 200)), caption="Original Submitted Image", width='stretch')
                                else:
                                    c2.info("No reference found")
                                    
                                if len(person_crops) > 1:
                                    with st.expander(f"View all {len(person_crops)} clustered images"):
                                        display_images = [ImageOps.exif_transpose(Image.open(c)).resize((100, 100)) for c in person_crops]
                                        st.image(display_images, width=80)

    with tab4:
        st.subheader("Unknown Individuals Detected")
        unknown_persons = []
        if os.path.exists(res["unknown_dir"]):
            unknown_persons = [os.path.join(res["unknown_dir"], f) for f in os.listdir(res["unknown_dir"]) if os.path.isdir(os.path.join(res["unknown_dir"], f))]
        
        if not unknown_persons:
            st.success("No unknown individuals detected in the classroom images.")
        else:
            st.warning(f"{len(unknown_persons)} unknown face(s) detected.")
            for i in range(0, len(unknown_persons), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(unknown_persons):
                        person_dir = unknown_persons[i + j]
                        with cols[j]:
                            with st.container(border=True):
                                basename = os.path.basename(person_dir)
                                parts = basename.split('_')
                                closest_name = parts[-1] if len(parts) > 2 else "None"
                                
                                person_crops = sorted([os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))])
                                main_crop = person_crops[0] if person_crops else None
                                if not main_crop:
                                    continue
                                    
                                ref_image_path = get_reference_image(closest_name, DATASET_DIR)
                                
                                st.write(f"**Unknown** (Nearest: {closest_name})")
                                c1, c2 = st.columns(2)
                                c1.image(ImageOps.exif_transpose(Image.open(main_crop)).resize((200, 200)), caption="Image from Classroom", width='stretch')
                                if ref_image_path:
                                    c2.image(ImageOps.exif_transpose(Image.open(ref_image_path)).resize((200, 200)), caption="Original Submitted Image", width='stretch')
                                else:
                                    c2.info("No reference found")
                                    
                                if len(person_crops) > 1:
                                    with st.expander(f"View all {len(person_crops)} clustered images"):
                                        display_images = [ImageOps.exif_transpose(Image.open(c)).resize((100, 100)) for c in person_crops]
                                        st.image(display_images, width=80)
                                    
                                all_students = res['df_final']['Student Name'].tolist()
                                default_idx = all_students.index(closest_name) if closest_name in all_students else 0
                                
                                action_c1, action_c2 = st.columns(2)
                                with action_c1:
                                    selected_student = st.selectbox("Select Student", all_students, index=default_idx, key=f"select_{person_dir}", label_visibility="collapsed")
                                with action_c2:
                                    if st.button("Mark", width='stretch', key=f"mark_{person_dir}"):
                                        mark_present_callback(person_dir, selected_student)
                                        st.rerun()

