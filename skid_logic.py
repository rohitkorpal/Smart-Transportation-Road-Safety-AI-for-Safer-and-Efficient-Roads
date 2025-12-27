"""
Skidding / Loss of Control Detection Logic
Detects rapid direction change + speed drop (V5 scenario)
"""
import numpy as np
import math

class SkidDetector:
    def __init__(self, angle_threshold=45, speed_drop_threshold=0.5):
        """
        Initialize skid detector
        
        Args:
            angle_threshold: Minimum angle change in degrees to detect skid
            speed_drop_threshold: Minimum speed drop ratio (0-1)
        """
        self.angle_threshold = angle_threshold
        self.speed_drop_threshold = speed_drop_threshold
        print("✅ Skid Detector initialized")
    
    def calculate_angle_change(self, dir1, dir2):
        """Calculate angle change between two direction vectors in degrees"""
        if dir1 is None or dir2 is None:
            return 0
        
        # Normalize vectors
        mag1 = np.sqrt(dir1[0]**2 + dir1[1]**2)
        mag2 = np.sqrt(dir2[0]**2 + dir2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        dir1_norm = (dir1[0] / mag1, dir1[1] / mag1)
        dir2_norm = (dir2[0] / mag2, dir2[1] / mag2)
        
        # Calculate angle using dot product
        dot = dir1_norm[0] * dir2_norm[0] + dir1_norm[1] * dir2_norm[1]
        dot = max(-1, min(1, dot))  # Clamp
        angle = math.degrees(math.acos(dot))
        
        return angle
    
    def detect_skid(self, tracked_vehicles, track_history, track_speeds):
        """
        Detect skidding/loss of control events
        
        Args:
            tracked_vehicles: List of tracked vehicles
            track_history: Dict of track position histories
            track_speeds: Dict of track speeds
            
        Returns:
            List of skidding track IDs
        """
        skidding_vehicles = []
        
        for vehicle in tracked_vehicles:
            track_id = vehicle[4]
            history = track_history.get(track_id, [])
            
            if len(history) < 3:
                continue
            
            # Get recent directions
            dir1 = self._get_direction(history, -3, -2)  # Previous direction
            dir2 = self._get_direction(history, -2, -1)  # Current direction
            
            if dir1 is None or dir2 is None:
                continue
            
            # Calculate angle change
            angle_change = self.calculate_angle_change(dir1, dir2)
            
            # Get speed history
            speeds = track_speeds.get(track_id, [])
            if len(speeds) < 2:
                continue
            
            # Check for speed drop
            prev_speed = speeds[-2] if len(speeds) >= 2 else speeds[-1]
            curr_speed = speeds[-1]
            
            if prev_speed > 0:
                speed_drop_ratio = (prev_speed - curr_speed) / prev_speed
            else:
                speed_drop_ratio = 0
            
            # Detect skid: rapid direction change + speed drop
            if angle_change > self.angle_threshold and speed_drop_ratio > self.speed_drop_threshold:
                skidding_vehicles.append(track_id)
        
        return skidding_vehicles
    
    def _get_direction(self, history, idx1, idx2):
        """Get direction vector between two history points"""
        if len(history) < abs(idx2) + 1:
            return None
        
        try:
            prev = history[idx1]
            curr = history[idx2]
            return (curr[0] - prev[0], curr[1] - prev[1])
        except IndexError:
            return None
