import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Page configuration and CSS
st.set_page_config(
    page_title="Cat & Dog Detector",
    page_icon="🐱",
    layout="wide"
)

# Custom CSS for layout, padding, and button states
st.markdown("""
    <style>
        /* Remove top blank space in main content */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        /* Button sizing */
        div.stButton > button {
            min-height: 100px;
            border-radius: 8px;
        }

        /* Standard font size for Next / Previous buttons */
        div.stButton > button p {
            font-size: 17px !important;
            font-weight: bold !important;
        }

        /* Bigger font size for 'Run Detection' button in sidebar */
        section[data-testid="stSidebar"] div.stButton > button p,
        button[data-testid="stBaseButton-primary"] p {
            font-size: 26px !important;
            font-weight: bold !important;
        }

        /* Ensure summary box matches button height */
        div.stAlert {
            min-height: 100px;
            display: flex;
            justify-content: center;
            flex-direction: column;
        }
    </style>
""", unsafe_allow_html=True)

# Load pre-trained model (cached)
@st.cache_resource
def load_model():
    MODEL_PATH = "weights/YoloCatDog.pt"
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Could not find '{MODEL_PATH}'. Please ensure you have placed your weights file in the 'weights' folder."
        )
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

# Configuration and helpers
MIN_BOX_AREA_RATIO = 0.01
IOU_SUPPRESSION_THRES = 0.50

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2 - b1)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0

# Detection logic
def process_images(uploaded_files, conf_thresh):
    results_output = []

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        image_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_h, img_w = image_cv2.shape[:2]
        img_area = img_h * img_w

        results = model(img, conf=conf_thresh, verbose=False)[0]

        raw_detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_area = (x2 - x1) * (y2 - y1)

            if conf < conf_thresh or box_area < img_area * MIN_BOX_AREA_RATIO:
                continue

            raw_detections.append(
                {"label": label, "conf": conf, "box": (x1, y1, x2, y2)}
            )

        final_detections = []
        for det in sorted(raw_detections, key=lambda x: x["conf"], reverse=True):
            if all(calculate_iou(det["box"], sel["box"]) <= IOU_SUPPRESSION_THRES for sel in final_detections):
                final_detections.append(det)

        cat_count = dog_count = 0
        cat_conf, dog_conf = [], []

        for det in final_detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            conf = det["conf"]

            if label == "cat":
                cat_count += 1
                cat_conf.append(conf * 100)
                color = (255, 50, 0)
            elif label == "dog":
                dog_count += 1
                dog_conf.append(conf * 100)
                color = (255, 0, 255)
            else:
                continue

            cv2.rectangle(image_cv2, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image_cv2,
                f"{label} ({conf * 100:.1f}%)",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        result_img = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

        summary = f"**Detected {cat_count} Cat(s) and {dog_count} Dog(s).**\n"
        if cat_conf:
            summary += f"\n- Average Cat Confidence: {sum(cat_conf) / len(cat_conf):.2f}%"
        if dog_conf:
            summary += f"\n- Average Dog Confidence: {sum(dog_conf) / len(dog_conf):.2f}%"

        results_output.append({"image": result_img, "summary": summary, "name": file.name})

    return results_output

# Streamlit GUI
if "processed_results" not in st.session_state:
    st.session_state.processed_results = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

st.title("Cat & Dog Object Detector")
st.markdown("Upload images, tweak the confidence threshold, and view your YOLO detections.")

with st.sidebar:
    st.header("⚙️ Settings")
    conf_thresh = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.50, step=0.05)

    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

    if st.button("Run Detection", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing images..."):
                st.session_state.processed_results = process_images(uploaded_files, conf_thresh)
                st.session_state.current_index = 0
        else:
            st.warning("Please upload at least one image.")

if st.session_state.processed_results:
    results = st.session_state.processed_results
    idx = st.session_state.current_index
    current_result = results[idx]

    # Center layout and render content
    spacer_left, center_col, spacer_right = st.columns([1, 2, 1])

    with center_col:
        # Image
        st.image(current_result["image"], use_container_width=True)

        # Image counter text
        st.markdown(
            f"<div style='text-align: center; font-size: 14px; color: gray; margin-top: 10px; margin-bottom: 10px;'>"
            f"Image {idx + 1} of {len(results)}<br>{current_result['name']}"
            f"</div>",
            unsafe_allow_html=True
        )

        # [Prev Button] [Summary] [Next Button]
        nav_prev, summary_col, nav_next = st.columns([1, 3, 1])

        with nav_prev:
            if st.button("Previous image", use_container_width=True, disabled=(idx == 0)):
                st.session_state.current_index -= 1
                st.rerun()

        with summary_col:
            st.info(current_result["summary"])

        with nav_next:
            if st.button("Next image", use_container_width=True, disabled=(idx == len(results) - 1)):
                st.session_state.current_index += 1
                st.rerun()

else:
    st.info("Upload your images in the sidebar and click **Run Detection** to get started.")