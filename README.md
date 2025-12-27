# 🚦 Smart AI Traffic & Accident Detection System

An AI-powered Smart-City road safety and traffic-optimization platform that detects vehicle accidents in real-time, monitors risky events, and automatically manages traffic signals based on congestion — using only CCTV cameras.

👤 **Author:** Rohit Korpal

---

## 🌍 Overview
This project uses advanced computer vision (YOLO + DeepSORT) to detect crashes, risky driving, stalled vehicles, and potential hazards while also dynamically allocating green-light time based on traffic congestion. It is built for smart-city deployment to improve emergency response and reduce traffic delays.

---

## 🧠 Features
- 🚘 Vehicle Accident Detection (crash, multi-car chain crash, skid, sudden stop, wrong-way driving)
- 🔥 Post-Accident Fire & Smoke Detection *(Phase-3)*
- 👥 Human Fall / Injury Detection *(Phase-2)*
- 🚦 Traffic Light Optimizer – counts vehicles & dynamically adjusts green timing
- 🆔 DeepSORT ID Tracking – maintains unique ID for each vehicle
- 📢 Alerts – prints emergency notifications (SMS/Email optional)
- 🎥 Supports MP4 files or CCTV camera RTSP

---

## 🧪 Detection Scenarios
### Phase-1 (MVP – Vehicle AI)
- Collision / Crash detection
- Chain crash (3+ vehicles)
- Wrong-way driving
- Sudden stop / stalled vehicle
- Vehicle skidding / loss of control
- Debris on road

### Phase-2 (Extended Human Safety)
- Person falling from bike
- Pedestrian hit
- Human lying on road (post-impact)

### Phase-3 (Post-Accident Fire Events)
- Smoke detection
- Fire on vehicle
- Explosion prediction (concept)

---
## 🏗 System Architecture (Flow)
CCTV Feed / Video File
↓
YOLO Object Detector
↓
DeepSORT Vehicle Tracker
↓
Crash / Risk Logic Engine
↓
Alerts → (Console / SMS / Email)
↓
Traffic Optimizer → Smart-Signal Control

## 📂 Folder Structure
Accident_AI/
│── main.py
│── detectors/
│── logic/
│── models/
│── videos/
│── requirements.txt
│── README.md

## 🚦 Traffic Optimization Logic
System counts vehicles per lane → detects congestion → gives more green light time to busy road → auto-switches red when flow is clear.

Example:
| Lane | Vehicles | Green Time |
|------|----------|------------|
| East-West | 18 | 35 sec |
| North-South | 7 | 15 sec |

---

## 🏁 Future Enhancements
- Ambulance GPS routing
- Mobile app accident alerts
- Database logging & analytics dashboard
- Azure cloud deployment

---

## ⭐ Contribute
Fork → Add improvements → Submit PR  
Give ⭐ if this repo helped you!
