"""
Stationary Vehicle Detection Logic
Detects sudden complete stops (V6 scenario)
"""
import numpy as np

class StationaryDetector:
    def __init__(self, stop_threshold=2, time_threshold=1):
        """
        Initialize stationary detector
        
        Args:
            stop_threshold: Maximum speed to consider as stopped (pixels/frame)
            time_threshold: Minimum frames to be stationary (seconds * fps)
        """
        self.stop_threshold = stop_threshold
        self.time_threshold = time_threshold
        self.stationary_timers = {}  # track_id -> frames_stationary
        print("✅ Stationary Detector initialized")
    
    def detect_stationary(self, tracked_vehicles, track_speeds, fps=30):
        """
        Detect vehicles that have suddenly stopped
        
        Args:
            tracked_vehicles: List of tracked vehicles
            track_speeds: Dict of track speeds
            fps: Frames per second (for time calculation)
            
        Returns:
            List of stationary track IDs
        """
        stationary_vehicles = []
        time_frames = int(self.time_threshold * fps)
        
        for vehicle in tracked_vehicles:
            track_id = vehicle[4]
            speed = track_speeds.get(track_id, 0)
            
            # Check if vehicle is moving slowly
            if speed < self.stop_threshold:
                # Increment stationary timer
                if track_id not in self.stationary_timers:
                    self.stationary_timers[track_id] = 0
                self.stationary_timers[track_id] += 1
                
                # Check if stationary long enough
                if self.stationary_timers[track_id] >= time_frames:
                    # Check if vehicle was moving before (sudden stop)
                    if self.stationary_timers[track_id] == time_frames:
                        stationary_vehicles.append(track_id)
            else:
                # Vehicle is moving, reset timer
                if track_id in self.stationary_timers:
                    del self.stationary_timers[track_id]
        
        return stationary_vehicles
