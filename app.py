"""
Construction Site PPE (Personal Protective Equipment) Detection Streamlit Web Application
Powered by Ultralytics YOLO11x (Extra Large model)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import io
import os

from config import (
    APP_TITLE, APP_SUBTITLE, APP_ICON, DEFAULT_MODEL, AVAILABLE_MODELS,
    DEFAULT_CONFIDENCE, DEFAULT_IOU, PPE_CLASSES, CUSTOM_CSS
)
from utils import (
    draw_bounding_boxes, compute_metrics, generate_audio_alert_html,
    create_demo_construction_images
)
from yolo_detector import PPEDetector

# 1. Page Configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS Theme
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "detection_logs" not in st.session_state:
    st.session_state.detection_logs = []
if "detector_instance" not in st.session_state:
    st.session_state.detector_instance = None
if "current_model_path" not in st.session_state:
    st.session_state.current_model_path = DEFAULT_MODEL


# Helper function to cache & load model
@st.cache_resource
def load_yolo_model(model_name_or_path):
    return PPEDetector(model_name_or_path)


# 3. Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/worker-with-road-block.png", width=70)
    st.title("⚙️ Control Panel")
    
    st.markdown("---")
    st.subheader("🤖 AI Model Selection")
    
    selected_model_label = st.selectbox(
        "Choose YOLO Model Architecture:",
        list(AVAILABLE_MODELS.keys()),
        index=0,
        help="YOLO11x provides highest detection precision for small objects like goggles and safety clips."
    )
    selected_model_file = AVAILABLE_MODELS[selected_model_label]
    
    # Custom Weight File Upload Option
    custom_weights_file = st.file_uploader(
        "Upload Custom Fine-Tuned (.pt) Model:",
        type=["pt"],
        help="Upload custom YOLO11 fine-tuned PPE weights if available."
    )
    
    if custom_weights_file is not None:
        custom_path = os.path.join(".", custom_weights_file.name)
        with open(custom_path, "wb") as f:
            f.write(custom_weights_file.getbuffer())
        active_model_path = custom_path
        st.success(f"Loaded custom weights: `{custom_weights_file.name}`")
    else:
        active_model_path = selected_model_file

    st.markdown("---")
    st.subheader("🎯 Detection Parameters")
    
    conf_thresh = st.slider(
        "Confidence Threshold:",
        min_value=0.05,
        max_value=1.00,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
        help="Minimum confidence score required for detection."
    )
    
    iou_thresh = st.slider(
        "NMS IoU Threshold:",
        min_value=0.10,
        max_value=1.00,
        value=DEFAULT_IOU,
        step=0.05,
        help="Overlap threshold for Non-Maximum Suppression."
    )

    st.markdown("---")
    st.subheader("🛡️ Filter PPE Classes")
    
    all_class_keys = list(PPE_CLASSES.keys())
    selected_classes = st.multiselect(
        "Display Selected Classes:",
        options=all_class_keys,
        default=all_class_keys,
        help="Filter specific PPE gear or violation alerts."
    )

    st.markdown("---")
    st.subheader("🔊 Safety Alert Settings")
    enable_audio_alert = st.toggle("Enable Audio Siren on Safety Violation", value=True)


# 2. Header Banner Section (Dynamic)
st.markdown(
    f"""
    <div class="header-banner">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="header-title">{APP_ICON} {APP_TITLE}</h1>
                <p class="header-subtitle">{APP_SUBTITLE}</p>
            </div>
            <div style="text-align: right;">
                <span class="badge-safe">🟢 SYSTEM ONLINE</span><br/>
                <span style="font-size: 0.8rem; color: #9ca3af; font-family: monospace;">Model: {os.path.basename(active_model_path)}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Load Detector Model
try:
    detector = load_yolo_model(active_model_path)
except Exception as e:
    st.error(f"Error loading model {active_model_path}: {e}")
    detector = load_yolo_model("yolo11x.pt")


# 4. Multi-Tab Navigation Layout
tab_image, tab_video, tab_webcam, tab_analytics = st.tabs([
    "📷 Image Analytics",
    "🎥 Video Stream",
    "📹 Live Webcam",
    "📊 Analytics & Reports"
])

demo_images = create_demo_construction_images()


# ==========================================
# TAB 1: IMAGE PPE ANALYTICS
# ==========================================
with tab_image:
    st.subheader("📷 Construction Site Image Analysis")
    
    col_input, col_demo = st.columns([2, 1])
    with col_input:
        uploaded_image = st.file_uploader(
            "Upload Construction Site Image:",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_uploader"
        )
    with col_demo:
        st.markdown("**or Try Demo Construction Images:**")
        selected_demo = st.selectbox(
            "Choose Sample Construction Image:",
            ["-- Select Demo Image --"] + list(demo_images.keys())
        )

    # Determine image source
    input_pil_img = None
    image_name = "Uploaded_Image.jpg"
    
    if uploaded_image is not None:
        input_pil_img = Image.open(uploaded_image).convert("RGB")
        image_name = uploaded_image.name
    elif selected_demo != "-- Select Demo Image --":
        input_pil_img = demo_images[selected_demo]
        image_name = selected_demo

    if input_pil_img is not None:
        st.markdown("---")
        
        with st.spinner(f"Running YOLO11x Inference ({active_model_path})..."):
            t_start = time.time()
            detections = detector.predict(
                image_input=input_pil_img,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                selected_classes=selected_classes
            )
            inference_time = (time.time() - t_start) * 1000

            # Compute safety metrics
            metrics = compute_metrics(detections)
            processed_img = draw_bounding_boxes(input_pil_img, detections)
            
            # Log detections to session state
            for det in detections:
                st.session_state.detection_logs.append({
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Source": image_name,
                    "Detected Class": det['class_name'],
                    "Status": "⚠️ VIOLATION" if det['is_violation'] else "✅ COMPLIANT",
                    "Confidence": f"{det['confidence']:.1%}",
                    "Bounding Box": str(det['box'])
                })

        # KPI Dashboard Cards
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Workers Detected</div>
                    <div class="metric-val" style="color: #38bdf8;">{metrics['worker_count']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Safety Compliance</div>
                    <div class="metric-val" style="color: {'#34d399' if metrics['compliance_pct'] >= 80 else '#f87171'};">{metrics['compliance_pct']}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Helmets Detected</div>
                    <div class="metric-val" style="color: #34d399;">{metrics['hardhat_count']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Vests Detected</div>
                    <div class="metric-val" style="color: #059669;">{metrics['vest_count']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Violations</div>
                    <div class="metric-val" style="color: {'#ef4444' if metrics['total_violations'] > 0 else '#34d399'};">{metrics['total_violations']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Trigger Audio Alert if violations exist
        if metrics['total_violations'] > 0 and enable_audio_alert:
            st.markdown(generate_audio_alert_html(), unsafe_allow_html=True)
            st.error(f"🚨 ALERT: {metrics['total_violations']} Safety Violation(s) Detected on Site!")

        st.markdown("<br/>", unsafe_allow_html=True)

        # Side-by-Side Image Views
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.subheader("📸 Original Site Image")
            st.image(input_pil_img, use_container_width=True)
        with img_col2:
            st.subheader(f"🛡️ YOLO11x Safety Overlay ({inference_time:.1f} ms)")
            st.image(processed_img, use_container_width=True)

        # Bounding Box Crop Gallery for Violations
        if metrics['total_violations'] > 0:
            st.subheader("⚠️ Detected Safety Violation Snippets")
            violation_dets = [d for d in detections if d.get('is_violation', False)]
            
            crop_cols = st.columns(min(4, len(violation_dets)))
            img_np = np.array(input_pil_img)
            
            for idx, det in enumerate(violation_dets):
                x1, y1, x2, y2 = det['box']
                crop = img_np[max(0, y1):min(img_np.shape[0], y2), max(0, x1):min(img_np.shape[1], x2)]
                
                with crop_cols[idx % len(crop_cols)]:
                    st.image(crop, caption=f"{det['class_name']} ({det['confidence']:.0%})", use_container_width=True)
    else:
        st.info("👋 Upload a construction site image or select a sample image above to view real-time PPE detection.")


# ==========================================
# TAB 2: VIDEO STREAM DETECTION
# ==========================================
with tab_video:
    st.subheader("🎥 Construction Site Video Stream Analysis")
    
    uploaded_video = st.file_uploader(
        "Upload Video File (MP4, AVI, MOV):",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )

    if uploaded_video is not None:
        # Save video to temporary file
        temp_video_path = os.path.join(".", "temp_input_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success(f"Loaded video file: `{uploaded_video.name}`")
        
        start_video_btn = st.button("🚀 Process & Detect Video Stream", type="primary")
        
        if start_video_btn:
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            
            st_frame = st.empty()
            st_progress = st.progress(0)
            st_metrics_placeholder = st.empty()
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                # Run detection every 2nd frame for smooth speed
                if frame_idx % 2 == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    
                    dets = detector.predict(
                        image_input=pil_frame,
                        conf_threshold=conf_thresh,
                        iou_threshold=iou_thresh,
                        selected_classes=selected_classes
                    )
                    
                    annotated_pil = draw_bounding_boxes(pil_frame, dets)
                    v_metrics = compute_metrics(dets)
                    
                    st_frame.image(annotated_pil, caption=f"Frame {frame_idx}/{total_frames} | Workers: {v_metrics['worker_count']} | Violations: {v_metrics['total_violations']}", use_container_width=True)
                    st_progress.progress(min(1.0, frame_idx / total_frames))
                    
            cap.release()
            st.success("✅ Video Stream Detection Completed!")
    else:
        st.info("📹 Upload an MP4 video file from your construction camera to run frame-by-frame safety compliance tracking.")


# ==========================================
# TAB 3: LIVE WEBCAM FEED
# ==========================================
with tab_webcam:
    st.subheader("📹 Live Camera PPE Compliance Check")
    st.markdown("Take a live photo or connect local camera feed for immediate PPE verification.")
    
    camera_photo = st.camera_input("Capture Live Worker Snapshot")
    
    if camera_photo is not None:
        cam_pil = Image.open(camera_photo).convert("RGB")
        
        with st.spinner("Analyzing camera feed with YOLO11x..."):
            cam_dets = detector.predict(
                image_input=cam_pil,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                selected_classes=selected_classes
            )
            
            cam_processed = draw_bounding_boxes(cam_pil, cam_dets)
            cam_metrics = compute_metrics(cam_dets)
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📸 Camera Capture")
                st.image(cam_pil, use_container_width=True)
            with c2:
                st.subheader("🛡️ PPE Safety Analysis")
                st.image(cam_processed, use_container_width=True)
                
            if cam_metrics['total_violations'] > 0:
                st.error(f"🚨 VIOLATION ALERT: {cam_metrics['total_violations']} worker(s) missing required PPE!")
                if enable_audio_alert:
                    st.markdown(generate_audio_alert_html(), unsafe_allow_html=True)
            else:
                st.success("✅ Safety Approved! All workers are compliant with site safety rules.")


# ==========================================
# TAB 4: ANALYTICS & REPORTS
# ==========================================
with tab_analytics:
    st.subheader("📊 Site Safety Compliance Dashboard & Reports")
    
    if len(st.session_state.detection_logs) > 0:
        df_logs = pd.DataFrame(st.session_state.detection_logs)
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### 🎯 Detection Class Distribution")
            class_counts = df_logs["Detected Class"].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]
            
            fig_pie = px.pie(
                class_counts,
                names="Class",
                values="Count",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            st.markdown("#### ⚠️ Compliance vs Violation Ratio")
            status_counts = df_logs["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            
            fig_bar = px.bar(
                status_counts,
                x="Status",
                y="Count",
                color="Status",
                color_discrete_map={"✅ COMPLIANT": "#34d399", "⚠️ VIOLATION": "#f87171"}
            )
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Safety Incident Logs Table")
        st.dataframe(df_logs, use_container_width=True)
        
        # Download CSV export
        csv_buffer = io.StringIO()
        df_logs.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Safety Incident Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"PPE_Safety_Report_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("📊 No detection events logged yet. Upload an image or video in Tab 1 / Tab 2 to generate safety reports.")
