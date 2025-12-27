"""
Human Pose Detector using YOLOv8-pose
Detects humans and their pose keypoints for fall detection
"""
import os
from ultralytics import YOLO
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class PoseDetector:
    def __init__(self, model_name="yolov8n-pose.pt", conf_threshold=0.25):
        """
        Initialize pose detector
        
        Args:
            model_name: YOLO pose model file name
            conf_threshold: Confidence threshold (lowered to 0.25 for better detection)
        """
        self.model_path = os.path.join(MODEL_DIR, model_name)
        self.conf_threshold = conf_threshold
        
        # Person class ID in COCO
        self.person_class = 0
        
        # Download model if not exists
        if not os.path.exists(self.model_path):
            print(f"📥 Downloading {model_name}...")
            self.model = YOLO(model_name)
            self.model.save(self.model_path)
        else:
            self.model = YOLO(self.model_path)
        
        print(f"✅ Pose Detector loaded: {model_name}")
    
    def detect(self, frame):
        """
        Detect humans and their poses
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections: [(x1, y1, x2, y2, conf, keypoints), ...]
            keypoints: numpy array of shape (17, 3) - 17 keypoints with (x, y, confidence)
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        detections = []
        if results.keypoints is not None:
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                if cls_id == self.person_class:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Get keypoints for this detection
                    keypoints = results.keypoints.data[i].cpu().numpy() if results.keypoints.data is not None else None
                    
                    detections.append((x1, y1, x2, y2, conf, keypoints))
        
        return detections
    
    def get_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox[:4]
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_keypoint(self, keypoints, idx):
        """
        Get specific keypoint by index
        Keypoint indices: 0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear,
                         5-left_shoulder, 6-right_shoulder, 7-left_elbow, 8-right_elbow,
                         9-left_wrist, 10-right_wrist, 11-left_hip, 12-right_hip,
                         13-left_knee, 14-right_knee, 15-left_ankle, 16-right_ankle
        """
        if keypoints is None or idx >= len(keypoints):
            return None
        return keypoints[idx] if keypoints[idx][2] > 0.5 else None  # confidence > 0.5
