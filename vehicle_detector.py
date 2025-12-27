"""
Vehicle Detector using YOLOv8
Detects cars, motorcycles, buses, trucks, and other vehicles
"""
import os
from ultralytics import YOLO
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class VehicleDetector:
    def __init__(self, model_name="yolov8m.pt", conf_threshold=0.5):
        """
        Initialize vehicle detector
        
        Args:
            model_name: YOLO model file name
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = os.path.join(MODEL_DIR, model_name)
        self.conf_threshold = conf_threshold
        
        # Vehicle class IDs in COCO dataset
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [2, 3, 5, 7]
        
        # Download model if not exists
        if not os.path.exists(self.model_path):
            print(f"📥 Downloading {model_name}...")
            self.model = YOLO(model_name)
            self.model.save(self.model_path)
        else:
            self.model = YOLO(self.model_path)
        
        print(f"✅ Vehicle Detector loaded: {model_name}")
    
    def detect(self, frame):
        """
        Detect vehicles in frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            List of detections: [(x1, y1, x2, y2, conf, cls_id), ...]
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf, cls_id))
        
        return detections
    
    def get_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox[:4]
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_area(self, bbox):
        """Get area of bounding box"""
        x1, y1, x2, y2 = bbox[:4]
        return (x2 - x1) * (y2 - y1)
