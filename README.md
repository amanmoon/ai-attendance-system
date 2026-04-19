# Automated Classroom Attendance System

An automated attendance tracking system powered by Deep Face Recognition and Streamlit. This project processes a dataset of student images to generate facial embeddings and later compares them against a classroom image to automatically mark attendance.

## Features

- **Face Registration & Embeddings**: Extracts facial features from student images using **InsightFace** (`buffalo_l` model for detection and representation) and stores them as `.pkl` embeddings.
- **Automated Attendance Marking**: Analyzes uploaded classroom images, extracts faces, and matches them against stored student embeddings using Cosine Distance.
- **Interactive Web Interface**: A user-friendly **Streamlit** dashboard for:
  - Generating and loading student datasets.
  - Uploading multiple classroom images to continuously evaluate attendance.
  - Viewing summarized Final Attendance reports with an option to download as a CSV.
  - Previewing annotated classroom images (with bounding boxes and names directly drawn).
  - Reviewing "Unknown" faces detected in the classroom.

## Prerequisites

- Python 3.8+
- The project dependencies listed in `requirements.txt`.

## Installation

1. Clone or download this repository.
2. Create and activate a Virtual Environment (Recommended):

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Algorithm Flow & Key Features

1. **Tiled Face Detection**: Splits large classroom images into overlapping tiles to improve detection of small or occluded faces (`buffalo_l`).
2. **Non-Maximum Suppression (NMS)**: Filters out duplicate bounding boxes generated at tile boundaries.
3. **Face Clustering**: Groups faces based on embedding distances. **Why it's useful**: Helps remove unwanted matching anomalies by combining multiple angles of the same person into one robust identity, reducing false positives.
4. **Face Matching & Classification**: Compares clustered embeddings against the known dataset. Strict calibration thresholds ensure unverified faces are marked as "Unknown".

## Dataset Structure

For the system to recognize students, you need to structure your dataset inside the `course_project_dataset` folder. Create a subfolder for each student using their name, and place their face images (e.g., from different angles) inside.

```text
Automated-Attendance-Management-System/
├── course_project_dataset/
│   ├── Alice/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   ├── Bob/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   └── ...
```

## How to Run

You can interact with the system entirely through the Streamlit App:

```bash
streamlit run app.py
```

### Streamlit Application Usage

1. **System Setup**: On the sidebar, click the "Load Dataset & Generate Embeddings" button. This will analyze the images in the `course_project_dataset/` folder and generate the embeddings file (`embeddings/embeddings_dl.pkl`). Note: This may take some time depending on hardware speed and dataset size.
2. **Evaluating Attendance**: Upload up to 5 classroom images (in `.jpg`, `.jpeg`, or `.png` format).
3. **Process Images**: Click "Process Images & Mark Attendance" to run the recognition pipeline.
4. **View Results**: Check the tabs to see:
   - **Attendance Report**: View and download the attendance CSV report (`Present`/`Absent`).
   - **Classroom Images**: View annotated versions of your uploaded images.
   - **Unknown Faces**: View cropped faces of people who weren't recognized.

### Command Line Interfacing (Optional)

You can also bypass the Streamlit app to independently generate embeddings or process an image from the terminal:

- Generate Embeddings: `python generate_embeddings.py`
- Mark Attendance for a single image: `python mark_attendance.py <path_to_class_image.jpg>`

## Project Structure

- `app.py`: The main Streamlit web application.
- `generate_embeddings.py`: Script dedicated to extracting faces from the `course_project_dataset` and generating mathematical embedding representations (stored in `.pkl` format).
- `mark_attendance.py`: Core logic for matching extracted faces from a classroom image strictly against the saved embeddings utilizing cosine-distance calculations.
- `requirements.txt`: Python package dependency list.
- `embeddings/`: Folder to store the generated embeddings.
- `output/`: Folder to save temporary output results like CSV files, annotated frames, and unknown face crops.
- `uploads/`: Folder used to store temporarily uploaded classroom images from Streamlit.
