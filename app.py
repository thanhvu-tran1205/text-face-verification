import streamlit as st
from deepface import DeepFace
import numpy as np
import cv2
import base64

st.set_page_config(page_title="DeepFace Clean UI", layout="wide")

# ================= 1. HÀM DATA & CONFIG =================
def get_image_base64(uploaded_file):
    try:
        bytes_data = uploaded_file.getvalue()
        base64_str = base64.b64encode(bytes_data).decode()
        return f"data:image/png;base64,{base64_str}"
    except:
        return ""

def array_to_base64(img_array):
    try:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        _, buffer = cv2.imencode('.jpg', img_array)
        base64_str = base64.b64encode(buffer).decode()
        return f"data:image/jpeg;base64,{base64_str}"
    except:
        return ""

MODEL_DEFAULTS = {
    "VGG-Face": 0.68,
    "Facenet": 0.40,
    "ArcFace": 0.68,
}

DETECTOR_BACKENDS = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'retinaface', 'mediapipe', 'yolov8n', 'yolov8m', 
    'yolov8l', 'yolov11n', 'yolov11s', 'yolov11m',
    'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m',
    'yolov12l', 'yunet', 'centerface',
]

DEFAULT_DETECTOR_BACKEND = DETECTOR_BACKENDS[0]

# ================= 2. CSS STYLING (ĐÃ SỬA) =================
st.markdown("""
<style>
    /* 1. KHUNG CHỨA ẢNH (FILLED) */
    .square-box {
        width: 100%; aspect-ratio: 1 / 1;
        border-radius: 10px;
        background-position: center center; 
        background-size: contain; 
        background-repeat: no-repeat;
        border: 2px solid #ddd;
        margin-bottom: 0px;
        /* Không còn ::after hay overlay text nào ở đây */
    }

    /* 2. KHUNG CHỜ (EMPTY - PLACEHOLDER) */
    .square-placeholder {
        width: 100%; aspect-ratio: 1 / 1;
        border-radius: 10px; 
        background-color: #f8f9fa; 
        border: 2px dashed #ccc;
        margin-bottom: 0px;
        
        /* Căn giữa text */
        display: flex; 
        align-items: center; 
        justify-content: center; 
        
        /* Style cho text bên trong */
        color: #aaa; 
        font-weight: bold;
        font-size: 0.9rem;
    }

    /* Các phần khác giữ nguyên */
    .upload-card { background: white; padding: 15px; border-radius: 12px; border: 1px solid #e0e0e0; margin-bottom: 15px; border-left: 5px solid #3498db; }
    .process-container { background: white; padding: 15px; border-radius: 12px; border: 1px solid #e0e0e0; border-top: 5px solid #f1c40f; }
    .output-container { background: white; padding: 15px; border-radius: 12px; border: 1px solid #e0e0e0; border-top: 5px solid #2ecc71; min-height: 250px; text-align: center; }
    
    .result-badge { padding: 8px; border-radius: 6px; font-weight: bold; color: white; display: block; width: 100%; margin-top: 10px; margin-bottom: 10px; }
    .match { background-color: #27ae60; }
    .diff { background-color: #c0392b; }
    .waiting { background-color: #95a5a6; opacity: 0.8; }

    .svg-container { display: flex; align-items: center; justify-content: center; height: 100%; opacity: 0.4; }
    .block-container { padding-top: 1rem; }
    div[data-testid="stExpander"] details { border-color: #eee; background-color: #f8f9fa; border-radius: 8px; }
    .param-row { display: flex; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px dashed #eee; padding-bottom: 4px; font-size: 0.9rem; }
    .param-label { font-weight: 600; color: #555; }
    .param-value { color: #2980b9; font-family: monospace; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("Face Verification Demo")

# ================= 3. LAYOUT =================
c_input, c_arrow1, c_process, c_arrow2, c_output = st.columns([1.2, 0.5, 2.0, 0.5, 1.2])

img1_cv2 = None
img2_cv2 = None
btn_verify = False

# === PHASE 1: INPUT ===
with c_input:
    st.markdown("##### 1. Input")
    f1 = st.file_uploader("Image 1", type=['jpg', 'png'], key="f1", label_visibility="collapsed")
    f2 = st.file_uploader("Image 2", type=['jpg', 'png'], key="f2", label_visibility="collapsed")

# === ARROWS 1 ===
with c_arrow1:
    st.markdown("""<div class="svg-container" style="height: 300px;"><svg width="100%" height="100%" viewBox="0 0 50 100" preserveAspectRatio="none"><line x1="0" y1="50" x2="40" y2="50" style="stroke:#bbb;stroke-width:2" /><polygon points="45,50 35,45 35,55" style="fill:#bbb" /></svg></div>""", unsafe_allow_html=True)

# === PHASE 2: MODEL CONFIGURATION ===
with c_process:
    st.markdown("##### 2. Model Configuration")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if f1:
            img_url = get_image_base64(f1)
            # Có ảnh -> Chỉ hiện box, không có text bên trong
            st.markdown(f'<div class="square-box" style="background-image: url(\'{img_url}\');"></div>', unsafe_allow_html=True)
            file_bytes = np.asarray(bytearray(f1.getvalue()), dtype=np.uint8)
            img1_cv2 = cv2.imdecode(file_bytes, 1)
            img1_cv2 = cv2.cvtColor(img1_cv2, cv2.COLOR_BGR2RGB)
        else:
            # Không ảnh -> Hiện text "Image 1" bên trong div
            st.markdown('<div class="square-placeholder">Image 1</div>', unsafe_allow_html=True)

    with col_p2:
        if f2:
            img_url = get_image_base64(f2)
            st.markdown(f'<div class="square-box" style="background-image: url(\'{img_url}\');"></div>', unsafe_allow_html=True)
            file_bytes = np.asarray(bytearray(f2.getvalue()), dtype=np.uint8)
            img2_cv2 = cv2.imdecode(file_bytes, 1)
            img2_cv2 = cv2.cvtColor(img2_cv2, cv2.COLOR_BGR2RGB)
        else:
            st.markdown('<div class="square-placeholder">Image 2</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    selected_model = st.selectbox("Select Model", list(MODEL_DEFAULTS.keys()))
    current_threshold = MODEL_DEFAULTS.get(selected_model, 0.40)
    
    # Checkbox để quản lý enforce_detection
    enforce_detection = st.checkbox(
        "Enforce Face Detection", 
        value=False,
        help="If enabled, will raise error if no face is detected. If disabled, will continue even without face detection."
    )
    
    with st.expander("Model Details", expanded=False):
        st.markdown(f"""
        <div class="param-row"><span class="param-label">Backend Detector:</span><span class="param-value">opencv</span></div>
        <div class="param-row"><span class="param-label">Distance Metric:</span><span class="param-value">cosine</span></div>
        <div class="param-row" style="border:none;"><span class="param-label">Threshold:</span><span class="param-value">{current_threshold}</span></div>
        """, unsafe_allow_html=True)
    
    st.write("") 
    btn_verify = st.button("RUN VERIFICATION", type="primary", use_container_width=True)

# === ARROWS 2 ===
with c_arrow2:
    st.markdown("""<div class="svg-container" style="height: 300px;"><svg width="100%" height="100%" viewBox="0 0 50 100" preserveAspectRatio="none"><line x1="0" y1="50" x2="40" y2="50" style="stroke:#bbb;stroke-width:2" /><polygon points="45,50 35,45 35,55" style="fill:#bbb" /></svg></div>""", unsafe_allow_html=True)

# === PHASE 3: RESULTS ===
with c_output:
    st.markdown("##### 3. Results")
    
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        ph_align1 = st.empty()
    with c_f2:
        ph_align2 = st.empty()
    
    ph_badge = st.empty()

    # KHỞI TẠO: Hiển thị text "Align 1/2" khi chưa có ảnh
    ph_align1.markdown('<div class="square-placeholder">Align 1</div>', unsafe_allow_html=True)
    ph_align2.markdown('<div class="square-placeholder">Align 2</div>', unsafe_allow_html=True)

    action_area = st.empty()
    
    if btn_verify and img1_cv2 is not None and img2_cv2 is not None:
        with action_area.container():
            try:
                ph_badge.markdown('<div class="result-badge waiting">Processing...</div>', unsafe_allow_html=True)

                res = DeepFace.verify(img1_cv2, img2_cv2, model_name=selected_model, enforce_detection=enforce_detection, detector_backend=DEFAULT_DETECTOR_BACKEND)

                faces1 = DeepFace.extract_faces(img1_cv2, enforce_detection=enforce_detection, detector_backend=DEFAULT_DETECTOR_BACKEND)
                faces2 = DeepFace.extract_faces(img2_cv2, enforce_detection=enforce_detection, detector_backend=DEFAULT_DETECTOR_BACKEND)
                
                # KHI CẬP NHẬT ẢNH: Chỉ render box có ảnh, không có text
                if len(faces1) > 0:
                    b64_1 = array_to_base64(faces1[0]["face"])
                    ph_align1.markdown(f'<div class="square-box" style="background-image: url(\'{b64_1}\');"></div>', unsafe_allow_html=True)
                
                if len(faces2) > 0:
                    b64_2 = array_to_base64(faces2[0]["face"])
                    ph_align2.markdown(f'<div class="square-box" style="background-image: url(\'{b64_2}\');"></div>', unsafe_allow_html=True)

                if res['verified']:
                    ph_badge.markdown('<div class="result-badge match">✅ MATCHED</div>', unsafe_allow_html=True)
                else:
                    ph_badge.markdown('<div class="result-badge diff">❌ DIFFERENT</div>', unsafe_allow_html=True)
                
                st.metric("Distance", f"{res['distance']:.4f}")
                
                # Hiển thị toàn bộ object res
                with st.expander("📋 Full Result Object", expanded=False):
                    st.json(res)

            except Exception as e:
                st.error(f"Can't extract faces from images")

    st.markdown('</div>', unsafe_allow_html=True)