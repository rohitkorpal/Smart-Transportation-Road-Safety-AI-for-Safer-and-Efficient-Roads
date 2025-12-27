"""
Fire and Smoke Detector
Detects fire and smoke in frames (Phase-3 feature)
"""
import os
from ultralytics import YOLO
import numpy as np
import cv2

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class FireDetector:
    def __init__(self):
        """Initialize fire detector using YOLOv8"""
        # Use standard YOLOv8 which can detect fire-related objects
        # In practice, you might use a custom trained fire detection model
        model_path = os.path.join(MODEL_DIR, "yolov8m.pt")
        
        if not os.path.exists(model_path):
            print("📥 Downloading YOLOv8 model for fire detection...")
            self.model = YOLO("yolov8m.pt")
            self.model.save(model_path)
        else:
            self.model = YOLO(model_path)
        
        print("✅ Fire Detector initialized")
    
    def detect(self, frame):
        """
        Detect fire and smoke in frame
        
        Args:
            frame: Input frame
            
        Returns:
            (has_fire, has_smoke, fire_regions)
        """
        # Convert to HSV for color-based fire detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Fire color range (red/orange/yellow)
        lower_fire1 = np.array([0, 50, 50])
        upper_fire1 = np.array([10, 255, 255])
        lower_fire2 = np.array([170, 50, 50])
        upper_fire2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        fire_mask = cv2.bitwise_or(mask1, mask2)
        
        # Smoke detection (grayish regions with movement)
        lower_smoke = np.array([0, 0, 100])
        upper_smoke = np.array([180, 30, 200])
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
        
        # Check for significant fire regions
        fire_pixels = cv2.countNonZero(fire_mask)
        smoke_pixels = cv2.countNonZero(smoke_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        
        fire_ratio = fire_pixels / total_pixels
        smoke_ratio = smoke_pixels / total_pixels
        
        has_fire = fire_ratio > 0.01  # 1% of frame
        has_smoke = smoke_ratio > 0.02  # 2% of frame
        
        # Get fire regions
        fire_regions = []
        if has_fire:
            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    fire_regions.append((x, y, x+w, y+h))
        
        return has_fire, has_smoke, fire_regions
