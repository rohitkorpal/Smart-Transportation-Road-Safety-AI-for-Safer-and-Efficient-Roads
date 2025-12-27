"""
Streamlit Dashboard for Accident Detection System
Live dashboard with video feed, alerts, and statistics
"""
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import tempfile
import os
import json

# Import system components
from main import AccidentDetectionSystem

# Page config
st.set_page_config(
    page_title="Accident Detection AI System",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 20px;
        }
        .alert-box {
            background-color: #ff4444;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #1e3c72;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Header
st.markdown("""
    <div class="main-header">
        <h1>🚦 Smart Traffic & Accident Detection AI</h1>
        <p>Real-time Road Safety Monitoring System</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Input source selection
    input_option = st.radio(
        "Input Source",
        ["Upload Video", "Upload Image (JPEG)", "RTSP Stream", "File Path"]
    )
    
    video_source = None
    image_source = None
    
    if input_option == "Upload Video":
        uploaded_file = st.file_uploader(
            "Upload video file",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_upload"
        )
        if uploaded_file:
            # Save to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_source = tfile.name
    
    elif input_option == "Upload Image (JPEG)":
        uploaded_image = st.file_uploader(
            "Upload image file (JPEG/JPG)",
            type=["jpg", "jpeg", "png"],
            key="image_upload"
        )
        if uploaded_image:
            # Save to temporary file
            suffix = ".jpg" if uploaded_image.name.lower().endswith(('.jpg', '.jpeg')) else ".png"
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tfile.write(uploaded_image.read())
            image_source = tfile.name
    
    elif input_option == "RTSP Stream":
        rtsp_url = st.text_input("RTSP URL", value="rtsp://example.com/stream")
        if rtsp_url:
            video_source = rtsp_url
    
    elif input_option == "File Path":
        file_path = st.text_input("Video/Image File Path")
        if file_path and os.path.exists(file_path):
            # Check if it's an image or video
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_source = file_path
            else:
                video_source = file_path
    
    # Options
    st.subheader("Detection Options")
    enable_pose = st.checkbox("Enable Human Detection (Phase-2)", value=False)
    enable_fire = st.checkbox("Enable Fire Detection (Phase-3)", value=False)
    fps = st.slider("FPS", min_value=15, max_value=60, value=30)
    
    # Start button
    if st.button("🚀 Start Detection", type="primary"):
        if video_source or image_source:
            with st.spinner("Initializing system..."):
                try:
                    # Reset state
                    if 'video_cap' in st.session_state:
                        if st.session_state.video_cap is not None:
                            st.session_state.video_cap.release()
                        del st.session_state.video_cap
                    if 'image_processed' in st.session_state:
                        del st.session_state.image_processed
                    
                    # Store source type
                    st.session_state.is_image = image_source is not None
                    st.session_state.source_path = image_source if image_source else video_source
                    
                    if image_source:
                        # For images, create system with dummy video source
                        st.session_state.system = AccidentDetectionSystem(
                            video_source="dummy",
                            fps=fps,
                            enable_pose=enable_pose,
                            enable_fire=enable_fire
                        )
                        # Mark as static image for collision detection
                        st.session_state.system.is_static_image = True
                    else:
                        st.session_state.system = AccidentDetectionSystem(
                            video_source=video_source,
                            fps=fps,
                            enable_pose=enable_pose,
                            enable_fire=enable_fire
                        )
                    st.session_state.video_processed = True
                    st.success("System initialized!")
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("Please provide a video or image source")

# Main content
if st.session_state.system and st.session_state.video_processed:
    # Create columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Check if processing image or video
        if st.session_state.get('is_image', False):
            st.subheader("🖼️ Image Detection")
            image_placeholder = st.empty()
            
            # Process image
            if 'image_processed' not in st.session_state:
                try:
                    # Read image
                    frame = cv2.imread(st.session_state.source_path)
                    if frame is not None:
                        # Process frame
                        tracked_vehicles, alerts = st.session_state.system.process_frame(frame)
                        
                        # Draw detections
                        frame = st.session_state.system.draw_detections(frame, tracked_vehicles, alerts)
                        
                        # Update alerts
                        if alerts:
                            st.session_state.alerts.extend(alerts)
                        
                        # Convert to RGB for Streamlit
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image_placeholder.image(frame_rgb, channels="RGB", width='stretch')
                        
                        st.session_state.image_processed = True
                        st.success("✅ Image processed successfully!")
                    else:
                        st.error("Could not read image file")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                # Re-display processed image
                frame = cv2.imread(st.session_state.source_path)
                if frame is not None:
                    tracked_vehicles, alerts = st.session_state.system.process_frame(frame)
                    frame = st.session_state.system.draw_detections(frame, tracked_vehicles, alerts)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_placeholder.image(frame_rgb, channels="RGB", width='stretch')
        else:
            # Video processing
            st.subheader("📹 Live Video Feed")
            video_placeholder = st.empty()
            
            # Process video frame by frame
            if 'video_cap' not in st.session_state:
                st.session_state.video_cap = cv2.VideoCapture(st.session_state.source_path)
                st.session_state.frame_count = 0
                st.session_state.video_finished = False
            
            cap = st.session_state.video_cap
            
            if cap.isOpened() and not st.session_state.video_finished:
                ret, frame = cap.read()
                
                if ret:
                    # Process frame
                    tracked_vehicles, alerts = st.session_state.system.process_frame(frame)
                    
                    # Draw detections
                    frame = st.session_state.system.draw_detections(frame, tracked_vehicles, alerts)
                    
                    # Update alerts
                    if alerts:
                        st.session_state.alerts.extend(alerts)
                    
                    # Convert to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", width='stretch')
                    
                    st.session_state.frame_count += 1
                    
                    # Limit processing for demo or check if video ended
                    if st.session_state.frame_count >= 100:  # Process first 100 frames
                        st.session_state.video_finished = True
                        cap.release()
                        st.info("✅ Video processing complete! Processed 100 frames.")
                else:
                    st.session_state.video_finished = True
                    cap.release()
                    st.info("✅ Video processing complete!")
            elif st.session_state.video_finished:
                st.info("✅ Video processing complete!")
            else:
                st.error("Could not open video source")
    
    with col2:
        st.subheader("📊 Statistics")
        
        # Get alert summary
        if st.session_state.system:
            summary = st.session_state.system.alert_manager.get_alert_summary()
            
            st.metric("Total Alerts", summary['total_alerts'])
            st.metric("Vehicles Tracked", len(st.session_state.system.track_history))
            
            # Alert counts
            st.subheader("Alert Breakdown")
            for alert_type, count in summary['alert_counts'].items():
                st.metric(alert_type.replace("_", " ").title(), count)
        
        # Recent alerts
        st.subheader("🚨 Recent Alerts")
        if st.session_state.alerts:
            for alert in st.session_state.alerts[-10:]:  # Last 10 alerts
                alert_type = alert[0]
                st.markdown(f"""
                    <div class="alert-box">
                        <strong>{alert_type.replace('_', ' ').title()}</strong>
                        <br>{datetime.now().strftime('%H:%M:%S')}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No alerts yet")
        
        # Download logs
        if st.button("📥 Download Alert Logs"):
            log_dir = st.session_state.system.alert_manager.log_dir
            if os.path.exists(log_dir):
                # Create zip of logs
                import zipfile
                zip_path = "alerts_logs.zip"
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file in os.listdir(log_dir):
                        zipf.write(os.path.join(log_dir, file), file)
                
                with open(zip_path, 'rb') as f:
                    st.download_button(
                        "Download",
                        f.read(),
                        file_name="alerts_logs.zip",
                        mime="application/zip"
                    )
else:
    st.info("👈 Please configure and start the system from the sidebar")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Smart Traffic & Accident Detection AI System | Microsoft Hackathon 2025</p>
        <p>Built with YOLOv8, DeepSORT, and Streamlit</p>
    </div>
""", unsafe_allow_html=True)

