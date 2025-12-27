import logging
# silence the Streamlit ScriptRunContext warnings
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
# optionally silence all streamlit runtime warnings
# logging.getLogger("streamlit").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext!")
import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from datetime import datetime
import numpy as np



# Load YOLO model
yolo_model = YOLO("yolov8m.pt")  
vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Streamlit Page Config
st.set_page_config(page_title="Hogwarts Flow Master", layout="wide")

# Magical Sparkle Effect + Fonts + CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=MedievalSharp&display=swap');

        .stApp {
            background-image: url(https://i.redd.it/esoi3wkswf6c1.png);
            background-size: cover;
            background-attachment: fixed;
            font-family: 'MedievalSharp', cursive;
            color: #f5f2d0;
        }

        .sparkles::before {
            content: '✨';
            animation: sparkle 1.5s infinite;
            font-size: 20px;
            margin-right: 5px;
        }

        @keyframes sparkle {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }

        .header-box {
            background-color: rgba(0, 0, 0, 0.75);
            padding: 30px;
            border: 4px solid #d4af37;
            border-radius: 20px;
            box-shadow: 0 0 25px #d4af37;
            color: #f5f2d0;
            text-align: center;
        }

        .lane-box {
            background: linear-gradient(135deg, #1a1a40 0%, #000000 100%);
            border: 2px solid #f5d142;
            color: #f5f2d0;
            font-size: 18px;
            border-radius: 15px;
            box-shadow: 0 0 20px #c9a52c, 0 0 30px #ffc107;
            padding: 16px;
            transition: all 0.3s ease-in-out;
        }

        .lane-box:hover {
            transform: scale(1.05);
            box-shadow: 0 0 30px #ffe082, 0 0 40px #ffd740;
        }

        .section-title {
            font-size: 36px;
            font-weight: bold;
            color: #ffd700;
            text-shadow: 2px 2px 6px black;
            margin-top: 40px;
            text-align: center;
        }

        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            font-size: 16px;
            color: #f5f2d0;
            background: rgba(0,0,0,0.85);
            padding: 14px;
            border-top: 1px solid #d4af37;
        }

        .stButton>button {
            background-color: #5d3a00;
            color: #f5f2d0;
            border: 3px solid #d4af37;
            border-radius: 12px;
            font-size: 20px;
            font-weight: bold;
            padding: 12px 24px;
            box-shadow: 0 0 12px #d4af37;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #7a5000;
            box-shadow: 0 0 24px #ffcc00;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Header
with st.container():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    st.markdown(f"""
        <div class='header-box'>
            <h1 class='sparkles'>✨ Hogwarts Flow Master ✨</h1>
            <p style='margin:0;font-size:18px;'>Real-time traffic control by spellcraft and logic</p>
            <p style='font-size:16px;margin-top:10px;'>🕰️ Current Time: <b>{current_time}</b></p>
        </div>
    """, unsafe_allow_html=True)

# Upload Section
lane_names = ["North", "East", "South", "West"]
uploaded_videos = [None] * 4
lane_colors = ["#B71C1C", "#F57F17", "#1B5E20", "#0D47A1"]

st.markdown("<div class='section-title'>🧙 Present Your Lane Observations</div>", unsafe_allow_html=True)

cols = st.columns(4)
for i in range(4):
    with cols[i]:
        st.markdown(f"<div style='background-color:{lane_colors[i]};padding:15px;border-radius:10px;text-align:center;'>" \
                    f"<b style='color:white'>{lane_names[i]} Lane</b>", unsafe_allow_html=True)
        uploaded_videos[i] = st.file_uploader(
                        f"Upload video for lane {i+1}",
                        type=["mp4", "avi", "mov"],
                        key=f"lane{i+1}",
                        label_visibility="collapsed"
                    )

# Magic Button
if st.button("🔮 Cast Signal Optimizing Spell"):
    if None in uploaded_videos:
        st.warning("⚠️ Please upload all 4 lane videos to proceed with the spell.")
    else:
        st.info("✨ Enchanting lanes... please wait while we divine the green light.")

        temp_paths = []
        for file in uploaded_videos:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(file.read())
                temp_paths.append(tmp.name)

        def count_vehicles(frame):
            frame = cv2.resize(frame, (640, 360))
            results = yolo_model(frame)
            count = 0
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in vehicle_ids:
                        count += 1
            return count

        counts = []
        for path in temp_paths:
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            if not ret:
                counts.append(0)
            else:
                counts.append(count_vehicles(frame))
            cap.release()

        green_index = counts.index(max(counts))

        st.markdown("<div class='section-title'>🧙 Green Light Chosen by Magic!</div>", unsafe_allow_html=True)
        result_cols = st.columns(4)
        for i in range(4):
            with result_cols[i]:
                signal_status = "🟢 GO" if i == green_index else "🔴 STOP"
                st.markdown(f"""
                    <div class='lane-box'>
                        <b>{lane_names[i]} Lane</b><br>
                        Signal: <b>{signal_status}</b><br>
                        Vehicles: <b>{counts[i]}</b>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='footer'>Made by <b>Hackwarts Founders 🧙‍♂️</b> | Triwizardathon 2025</div>",
    unsafe_allow_html=True
)
