"""
Crash Detection Logic
Detects vehicle collisions (V1, V3 scenarios)
"""
import numpy as np
import math

class CrashDetector:
    def __init__(self, collision_distance=50, speed_threshold=10, image_iou_threshold=0.05):
        """
        Initialize crash detector
        
        Args:
            collision_distance: Minimum distance for collision (pixels)
            speed_threshold: Minimum speed before collision to trigger alert
            image_iou_threshold: IoU threshold for static image collision detection
        """
        self.collision_distance = collision_distance
        self.speed_threshold = speed_threshold
        self.image_iou_threshold = image_iou_threshold
        self.crash_history = {}  # Track ID -> crash state
        print("✅ Crash Detector initialized")
    
    def calculate_distance(self, bbox1, bbox2):
        """Calculate distance between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
        
        # Get centers
        center1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
        center2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)
        
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect_crash(self, tracked_vehicles, track_speeds, track_history, is_static_image=False):
        """
        Detect vehicle collisions
        
        Args:
            tracked_vehicles: List of (x1, y1, x2, y2, track_id, conf, cls_id)
            track_speeds: Dict {track_id: speed} or {track_id: [speeds]}
            track_history: Dict {track_id: [positions]}
            is_static_image: If True, uses static image detection (no speed requirement)
            
        Returns:
            List of crash events: [(track_id1, track_id2, crash_type), ...]
        """
        crashes = []
        processed_pairs = set()
        
        for i, vehicle1 in enumerate(tracked_vehicles):
            track_id1 = vehicle1[4]
            # Handle both list and single value for speeds
            speed_data1 = track_speeds.get(track_id1, 0)
            if isinstance(speed_data1, list):
                speed1 = speed_data1[-1] if speed_data1 else 0
            else:
                speed1 = speed_data1
            
            for j, vehicle2 in enumerate(tracked_vehicles[i+1:], start=i+1):
                track_id2 = vehicle2[4]
                speed_data2 = track_speeds.get(track_id2, 0)
                if isinstance(speed_data2, list):
                    speed2 = speed_data2[-1] if speed_data2 else 0
                else:
                    speed2 = speed_data2
                
                # Avoid duplicate pairs
                pair = tuple(sorted([track_id1, track_id2]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                # Calculate distance and IoU
                distance = self.calculate_distance(vehicle1, vehicle2)
                iou = self.calculate_iou(vehicle1, vehicle2)
                
                # Check for collision
                collision_detected = False
                
                if is_static_image:
                    # For static images: use IoU and distance only (no speed requirement)
                    # More lenient thresholds for images - detect if vehicles are close or overlapping
                    if iou > self.image_iou_threshold or distance < self.collision_distance * 2:
                        collision_detected = True
                else:
                    # For video: require speed + proximity
                    if distance < self.collision_distance or iou > 0.1:
                        if speed1 > self.speed_threshold or speed2 > self.speed_threshold:
                            collision_detected = True
                
                if collision_detected:
                    # Determine crash type
                    crash_type = self._classify_crash_type(vehicle1, vehicle2, track_history, is_static_image)
                    crashes.append((track_id1, track_id2, crash_type))
        
        return crashes
    
    def _classify_crash_type(self, vehicle1, vehicle2, track_history, is_static_image=False):
        """
        Classify crash type: rear-end, side-impact, or head-on
        
        Args:
            vehicle1, vehicle2: Vehicle bounding boxes
            track_history: Track position history
            is_static_image: If True, uses position-based classification
            
        Returns:
            Crash type string
        """
        track_id1 = vehicle1[4]
        track_id2 = vehicle2[4]
        
        if is_static_image:
            # For static images, classify based on relative positions
            x1_1, y1_1, x2_1, y2_1 = vehicle1[:4]
            x1_2, y1_2, x2_2, y2_2 = vehicle2[:4]
            
            # Get centers
            center1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
            center2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)
            
            # Calculate relative positions
            dx = center2[0] - center1[0]
            dy = center2[1] - center1[1]
            
            # Classify based on relative position
            if abs(dx) > abs(dy) * 2:  # More horizontal
                return "side-impact"
            elif abs(dy) > abs(dx) * 2:  # More vertical
                return "rear-end"
            else:
                return "collision"
        
        # Get movement directions (for video)
        dir1 = self._get_direction(track_history.get(track_id1, []))
        dir2 = self._get_direction(track_history.get(track_id2, []))
        
        if dir1 is None or dir2 is None:
            return "collision"
        
        # Calculate angle between directions
        angle = self._angle_between_vectors(dir1, dir2)
        
        # Classify based on angle
        if angle < 30:  # Similar directions (rear-end)
            return "rear-end"
        elif angle > 150:  # Opposite directions (head-on)
            return "head-on"
        else:  # Perpendicular (side-impact)
            return "side-impact"
    
    def _get_direction(self, history):
        """Get movement direction vector from history"""
        if len(history) < 2:
            return None
        prev = history[-2]
        curr = history[-1]
        return (curr[0] - prev[0], curr[1] - prev[1])
    
    def _angle_between_vectors(self, v1, v2):
        """Calculate angle between two vectors in degrees"""
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        angle = math.degrees(math.acos(cos_angle))
        return angle
