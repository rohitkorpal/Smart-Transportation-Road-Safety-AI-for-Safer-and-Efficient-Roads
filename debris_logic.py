"""
Debris Detection Logic
Detects debris/car parts on road (V7 scenario)
"""
import numpy as np
import cv2

class DebrisDetector:
    def __init__(self, min_area=2000, aspect_ratio_range=(0.2, 5.0), min_frames_static=5):
        """
        Initialize debris detector
        
        Args:
            min_area: Minimum area for debris detection (increased to reduce false positives)
            aspect_ratio_range: Valid aspect ratio range for debris
            min_frames_static: Minimum frames object must be static to be considered debris
        """
        self.min_area = min_area
        self.aspect_ratio_range = aspect_ratio_range
        self.min_frames_static = min_frames_static
        self.static_objects = {}  # Track static objects: {bbox: frame_count}
        print("✅ Debris Detector initialized")
    
    def detect_debris(self, frame, tracked_vehicles, vehicle_detections):
        """
        Detect debris/objects on road
        
        Args:
            frame: Current frame
            tracked_vehicles: List of tracked vehicles
            vehicle_detections: All vehicle detections (including untracked)
            
        Returns:
            List of debris regions: [(x1, y1, x2, y2), ...]
        """
        # Get bounding boxes of all tracked vehicles
        vehicle_boxes = []
        for vehicle in tracked_vehicles:
            vehicle_boxes.append(vehicle[:4])
        
        debris_regions = []
        
        # More conservative approach: only detect in very lower part of frame (road surface)
        h, w = frame.shape[:2]
        road_region = frame[int(h * 0.75):, :]  # Lower 25% of frame only
        
        # Convert to grayscale
        gray = cv2.cvtColor(road_region, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold (better than simple threshold)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_objects = {}
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:  # Increased threshold
                continue
            
            x, y, w_cont, h_cont = cv2.boundingRect(contour)
            
            # Adjust y coordinate (we cropped the frame)
            y += int(h * 0.75)
            
            # Check aspect ratio
            aspect_ratio = w_cont / h_cont if h_cont > 0 else 0
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            
            # Check if this region overlaps with any vehicle
            debris_box = (x, y, x + w_cont, y + h_cont)
            is_vehicle = False
            for v_box in vehicle_boxes:
                if self._boxes_overlap(debris_box, v_box):
                    is_vehicle = True
                    break
            
            if not is_vehicle:
                # Check if this object has been static for enough frames
                # Find similar object in previous frames
                found_match = False
                for prev_box, frame_count in self.static_objects.items():
                    if self._boxes_similar(debris_box, prev_box, threshold=30):
                        # Object is still there, increment count
                        current_objects[debris_box] = frame_count + 1
                        found_match = True
                        break
                
                if not found_match:
                    # New object, start tracking
                    current_objects[debris_box] = 1
        
        # Update static objects
        self.static_objects = current_objects
        
        # Only return objects that have been static for minimum frames
        for debris_box, frame_count in self.static_objects.items():
            if frame_count >= self.min_frames_static:
                debris_regions.append(debris_box)
        
        return debris_regions
    
    def _boxes_similar(self, box1, box2, threshold=30):
        """Check if two boxes are similar (within threshold distance)"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        center1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
        center2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance < threshold
    
    def _boxes_overlap(self, box1, box2):
        """Check if two bounding boxes overlap"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
