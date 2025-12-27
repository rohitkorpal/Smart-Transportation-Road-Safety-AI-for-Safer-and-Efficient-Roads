"""
DeepSORT Tracker for vehicle tracking across frames
Falls back to simple IoU-based tracking if DeepSORT is not available
"""
import numpy as np

try:
    from deep_sort_realtime import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("⚠️ DeepSORT not available, using simple IoU tracking")

class VehicleTracker:
    def __init__(self, max_age=30, n_init=3):
        """
        Initialize tracker (DeepSORT if available, else simple tracking)
        
        Args:
            max_age: Maximum frames to keep track without detection
            n_init: Number of consecutive detections needed to start tracking
        """
        self.max_age = max_age
        self.n_init = n_init
        self.track_history = {}  # Track ID -> list of positions
        self.track_speeds = {}   # Track ID -> list of speeds
        self.next_id = 0
        self.tracks = {}  # track_id -> {bbox, age, hits, conf, cls_id}
        
        if DEEPSORT_AVAILABLE:
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=0.7,
                max_cosine_distance=0.2,
                nn_budget=None
            )
            print("✅ DeepSORT Tracker initialized")
        else:
            self.tracker = None
            print("✅ Simple IoU Tracker initialized")
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _simple_track(self, detections):
        """Simple IoU-based tracking fallback"""
        tracked_objects = []
        
        # Update existing tracks
        for track_id, track in list(self.tracks.items()):
            track['age'] += 1
            best_iou = 0
            best_det_idx = -1
            
            for idx, det in enumerate(detections):
                iou = self._calculate_iou(track['bbox'], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx
            
            if best_iou > 0.3 and best_det_idx >= 0:  # Match found
                det = detections[best_det_idx]
                track['bbox'] = det[:4]
                track['hits'] += 1
                track['age'] = 0
                track['conf'] = det[4]
                track['cls_id'] = det[5]
                detections.pop(best_det_idx)
                
                # Update history
                center = ((det[0] + det[2]) // 2, (det[1] + det[3]) // 2)
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append(center)
                if len(self.track_history[track_id]) > 10:
                    self.track_history[track_id] = self.track_history[track_id][-10:]
                
                # Calculate speed
                if len(self.track_history[track_id]) >= 2:
                    prev = self.track_history[track_id][-2]
                    curr = self.track_history[track_id][-1]
                    speed = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                    if track_id not in self.track_speeds:
                        self.track_speeds[track_id] = []
                    self.track_speeds[track_id].append(speed)
                    if len(self.track_speeds[track_id]) > 5:
                        self.track_speeds[track_id] = self.track_speeds[track_id][-5:]
                
                x1, y1, x2, y2 = det[:4]
                tracked_objects.append((x1, y1, x2, y2, track_id, det[4], det[5]))
            elif track['age'] > self.max_age:
                # Remove old tracks
                del self.tracks[track_id]
        
        # Create new tracks for unmatched detections
        for det in detections:
            if self.tracker is None:  # Only for simple tracking
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'bbox': det[:4],
                    'age': 0,
                    'hits': 1,
                    'conf': det[4],
                    'cls_id': det[5]
                }
                x1, y1, x2, y2 = det[:4]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                self.track_history[track_id] = [center]
                tracked_objects.append((x1, y1, x2, y2, track_id, det[4], det[5]))
        
        return tracked_objects
    
    def update(self, detections, frame):
        """
        Update tracker with new detections
        
        Args:
            detections: List of (x1, y1, x2, y2, conf, cls_id)
            frame: Current frame for feature extraction
            
        Returns:
            List of tracked objects: [(x1, y1, x2, y2, track_id, conf, cls_id), ...]
        """
        if self.tracker is None:
            return self._simple_track(detections)
        
        # Use DeepSORT
        if not detections:
            tracks = self.tracker.update_tracks([], frame=frame)
        else:
            # Convert to format: [(x1, y1, x2, y2, conf, cls_id), ...]
            dets = []
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                dets.append(([x1, y1, x2, y2], conf, cls_id))
            
            tracks = self.tracker.update_tracks(dets, frame=frame)
        
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Get class ID and confidence from track state if available
            cls_id = getattr(track, 'class_id', 2)  # Default to car
            conf = getattr(track, 'confidence', 0.5)  # Default confidence
            
            # Update track history
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(center)
            
            # Keep only last 10 positions
            if len(self.track_history[track_id]) > 10:
                self.track_history[track_id] = self.track_history[track_id][-10:]
            
            # Calculate speed
            if len(self.track_history[track_id]) >= 2:
                prev_pos = self.track_history[track_id][-2]
                curr_pos = self.track_history[track_id][-1]
                speed = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                              (curr_pos[1] - prev_pos[1])**2)
                
                if track_id not in self.track_speeds:
                    self.track_speeds[track_id] = []
                self.track_speeds[track_id].append(speed)
                
                # Keep only last 5 speeds
                if len(self.track_speeds[track_id]) > 5:
                    self.track_speeds[track_id] = self.track_speeds[track_id][-5:]
            
            tracked_objects.append((x1, y1, x2, y2, track_id, conf, cls_id))
        
        return tracked_objects
    
    def get_track_history(self, track_id):
        """Get position history for a track"""
        return self.track_history.get(track_id, [])
    
    def get_track_speed(self, track_id):
        """Get average speed for a track"""
        speeds = self.track_speeds.get(track_id, [])
        return np.mean(speeds) if speeds else 0.0
    
    def get_track_direction(self, track_id):
        """Get movement direction vector for a track"""
        history = self.get_track_history(track_id)
        if len(history) < 2:
            return None
        prev = history[-2]
        curr = history[-1]
        return (curr[0] - prev[0], curr[1] - prev[1])
