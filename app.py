import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import time
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="PCB Component Detector",
    page_icon="🔌",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------
custom_css = """
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    background-size: cover;
}

[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
}

.title {
    color: white;
    text-align: center;
    font-size: 48px;
    font-weight: 900;
    margin-top: -20px;
}
.subtitle {
    color: #e0e0e0;
    text-align: center;
    font-size: 20px;
    margin-bottom: 40px;
}

.glass-card {
    background: rgba(255, 255, 255, 0.12);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.25);
    margin-bottom: 25px;
    backdrop-filter: blur(12px);
}

img {
    border-radius: 15px !important;
}

.label-badge {
    background: rgba(255,255,255,0.15);
    padding: 10px 18px;
    border-radius: 12px;
    color: white;
    font-size: 16px;
    margin: 5px 0px;
    font-weight: 600;
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("<div class='title'>🔌 PCB COMPONENT DETECTOR</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Objects Detection • Batch Upload • Dashboard • Webcam • Video</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)



# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Controls")

mode = st.sidebar.radio(
    "Choose Mode",
    ["Single Image Scan", "Batch Image Scan", "Webcam Detection", "Video File Detection"]
)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
st.sidebar.markdown("---")
st.sidebar.info("""👤 Built by Batch 50 \n
Under Guidance of 
\nMrs.K.Jayanthi""")


# ###########################################
# 🔢 COMPONENT COUNT HELPER
# ###########################################
def get_component_counts(results):
    counts = {}
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = results[0].names[cls]
        counts[label] = counts.get(label, 0) + 1
    return counts


# ==================================================================
# 1️⃣ SINGLE IMAGE DETECTION
# ==================================================================
if mode == "Single Image Scan":

    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        # Input image
        with col1:
            st.subheader("📥 Input Image")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_column_width=True)

        # Prediction
        with st.spinner("Detecting components..."):
            results = model.predict(image, conf=conf_threshold)
            result_img = results[0].plot()

        # Output
        with col2:
            st.subheader("📤 Output Image")
            st.image(result_img, use_column_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Dashboard Section
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📊 Component Count Dashboard")

        counts = get_component_counts(results)

        if counts:
            df = pd.DataFrame({'Component': list(counts.keys()), 'Count': list(counts.values())})
            fig = px.bar(df, x='Component', y='Count', text='Count')
            st.plotly_chart(fig)

            for comp, cnt in counts.items():
                st.markdown(f"<div class='label-badge'>{comp} — {cnt}</div>", unsafe_allow_html=True)
        else:
            st.warning("No components detected.")

        st.markdown("</div>", unsafe_allow_html=True)



# ==================================================================
# 2️⃣ BATCH IMAGE UPLOAD — MULTIPLE IMAGES
# ==================================================================
elif mode == "Batch Image Scan":

    uploaded_files = st.sidebar.file_uploader(
        "Upload Multiple Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader("🖼️ Batch Results")

        for img_file in uploaded_files:

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.write(f"📌 **Image:** {img_file.name}")

            image = Image.open(img_file).convert("RGB")

            # Run YOLO
            results = model.predict(image, conf=conf_threshold)
            result_img = results[0].plot()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Input Image")
                st.image(image, use_column_width=True)

            with col2:
                st.subheader("Output Image")
                st.image(result_img, use_column_width=True)

            # Component count
            st.subheader("Component Summary")

            counts = get_component_counts(results)
            if counts:
                for comp, cnt in counts.items():
                    st.markdown(f"<div class='label-badge'>{comp} — {cnt}</div>", unsafe_allow_html=True)
            else:
                st.warning("No components detected.")

            st.markdown("</div>", unsafe_allow_html=True)



# ==================================================================
# 3️⃣ WEBCAM DETECTION
# ==================================================================
elif mode == "Webcam Detection":

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📷 Webcam Detection")

    start = st.button("Start Webcam")

    if start:
        FRAME = st.empty()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam error.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.flip(rgb,1)
            results = model.predict(rgb, conf=conf_threshold, verbose=False)
            out_frame = results[0].plot()

            FRAME.image(out_frame, channels="RGB")

        cap.release()

    st.markdown("</div>", unsafe_allow_html=True)



# ==================================================================
# 4️⃣ VIDEO DETECTION
# ==================================================================
elif mode == "Video File Detection":

    uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🎞 Video Detection")

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        st.video(temp_file.name)

        FRAME = st.empty()
        cap = cv2.VideoCapture(temp_file.name)

        with st.spinner("Detecting components..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(rgb, conf=conf_threshold, verbose=False)
                out_frame = results[0].plot()

                FRAME.image(out_frame, channels="RGB")
                time.sleep(0.02)

        cap.release()
        st.success("Video detection completed!")
        st.markdown("</div>", unsafe_allow_html=True)
