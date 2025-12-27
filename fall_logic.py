"""
Human Fall Detection Logic
Detects person falling or thrown from bike (H1, H2 scenarios)
"""
import numpy as np

class FallDetector:
    def __init__(self, height_ratio_threshold=0.3, keypoint_confidence=0.5):
        """
        Initialize fall detector
        
        Args:
            height_ratio_threshold: Maximum height/width ratio for fallen person
            keypoint_confidence: Minimum confidence for keypoints
        """
        self.height_ratio_threshold = height_ratio_threshold
        self.keypoint_confidence = keypoint_confidence
        self.person_history = {}  # track_id -> [heights, positions]
        print("✅ Fall Detector initialized")
    
    def detect_fall(self, pose_detections, track_history=None):
        """
        Detect person falling based on pose keypoints
        
        Args:
            pose_detections: List of pose detections from PoseDetector
            track_history: Optional track history for temporal analysis
            
        Returns:
            List of fall events: [(bbox, fall_type), ...]
        """
        fall_events = []
        
        for detection in pose_detections:
            x1, y1, x2, y2, conf, keypoints = detection
            
            if keypoints is None:
                continue
            
            # Check if person is lying down (horizontal)
            fall_type = self._analyze_pose(keypoints, (x1, y1, x2, y2))
            
            if fall_type:
                fall_events.append(((x1, y1, x2, y2), fall_type))
        
        return fall_events
    
    def _analyze_pose(self, keypoints, bbox):
        """
        Analyze pose to determine if person has fallen
        
        Returns:
            'fall' if fallen, 'lying' if lying down, None otherwise
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Get key keypoints
        # Keypoint indices: 5-left_shoulder, 6-right_shoulder, 11-left_hip, 12-right_hip
        # Check if keypoints exist and have valid confidence
        left_shoulder = None
        right_shoulder = None
        left_hip = None
        right_hip = None
        
        if keypoints is not None and len(keypoints) > 12:
            # Safely extract keypoints with confidence check
            # Convert to float to ensure scalar comparison
            if float(keypoints[5][2]) > self.keypoint_confidence:
                left_shoulder = keypoints[5]
            if float(keypoints[6][2]) > self.keypoint_confidence:
                right_shoulder = keypoints[6]
            if float(keypoints[11][2]) > self.keypoint_confidence:
                left_hip = keypoints[11]
            if float(keypoints[12][2]) > self.keypoint_confidence:
                right_hip = keypoints[12]
        
        # Check if person is horizontal (lying down)
        if left_shoulder is not None and right_shoulder is not None and left_hip is not None and right_hip is not None:
            # Calculate body orientation
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            
            # If shoulders and hips are at similar height, person is horizontal
            height_diff = abs(shoulder_center_y - hip_center_y)
            body_width = abs(left_shoulder[0] - right_shoulder[0])
            
            if height_diff < body_width * 0.3:  # Person is horizontal
                # Check if height/width ratio indicates lying
                if width > 0:
                    aspect_ratio = height / width
                    if aspect_ratio < self.height_ratio_threshold:
                        return 'lying'
        
        # Check bounding box aspect ratio (fallen person is usually wider than tall)
        if width > 0:
            aspect_ratio = height / width
            if aspect_ratio < self.height_ratio_threshold:
                return 'fall'
        
        return None
    
    def detect_pedestrian_hit(self, pose_detections, tracked_vehicles, crash_events):
        """
        Detect pedestrian hit by vehicle (H2 scenario)
        
        Args:
            pose_detections: List of pose detections
            tracked_vehicles: List of tracked vehicles
            crash_events: List of crash events
            
        Returns:
            List of pedestrian hit events
        """
        pedestrian_hits = []
        
        for pose_det in pose_detections:
            person_bbox = pose_det[:4]
            person_center = ((person_bbox[0] + person_bbox[2]) // 2,
                           (person_bbox[1] + person_bbox[3]) // 2)
            
            # Check if person is near any vehicle involved in crash
            for vehicle in tracked_vehicles:
                vehicle_bbox = vehicle[:4]
                vehicle_center = ((vehicle_bbox[0] + vehicle_bbox[2]) // 2,
                                (vehicle_bbox[1] + vehicle_bbox[3]) // 2)
                
                # Calculate distance
                distance = np.sqrt((person_center[0] - vehicle_center[0])**2 +
                                 (person_center[1] - vehicle_center[1])**2)
                
                # If person is very close to vehicle and there's a crash
                if distance < 50:  # Threshold
                    for crash in crash_events:
                        if vehicle[4] in [crash[0], crash[1]]:
                            pedestrian_hits.append((person_bbox, vehicle[4]))
                            break
        
        return pedestrian_hits
