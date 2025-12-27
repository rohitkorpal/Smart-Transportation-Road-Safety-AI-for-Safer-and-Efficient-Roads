"""
Motion Prediction Logic
Predicts potential crashes before they happen (V8 scenario)
"""
import numpy as np
import math

class MotionPredictor:
    def __init__(self, prediction_horizon=10, collision_threshold=30):
        """
        Initialize motion predictor
        
        Args:
            prediction_horizon: Number of frames to predict ahead
            collision_threshold: Distance threshold for predicted collision
        """
        self.prediction_horizon = prediction_horizon
        self.collision_threshold = collision_threshold
        print("✅ Motion Predictor initialized")
    
    def predict_collision(self, tracked_vehicles, track_history, track_speeds):
        """
        Predict potential collisions based on current trajectories
        
        Args:
            tracked_vehicles: List of tracked vehicles
            track_history: Dict of track position histories
            track_speeds: Dict of track speeds
            
        Returns:
            List of predicted collision pairs: [(track_id1, track_id2, time_to_collision), ...]
        """
        predicted_collisions = []
        
        for i, vehicle1 in enumerate(tracked_vehicles):
            track_id1 = vehicle1[4]
            trajectory1 = self._predict_trajectory(track_id1, track_history, track_speeds)
            
            if trajectory1 is None:
                continue
            
            for j, vehicle2 in enumerate(tracked_vehicles[i+1:], start=i+1):
                track_id2 = vehicle2[4]
                trajectory2 = self._predict_trajectory(track_id2, track_history, track_speeds)
                
                if trajectory2 is None:
                    continue
                
                # Check for intersection in predicted trajectories
                collision_frame = self._check_trajectory_intersection(
                    trajectory1, trajectory2, vehicle1, vehicle2
                )
                
                if collision_frame is not None and collision_frame <= self.prediction_horizon:
                    predicted_collisions.append((track_id1, track_id2, collision_frame))
        
        return predicted_collisions
    
    def _predict_trajectory(self, track_id, track_history, track_speeds):
        """
        Predict future trajectory for a track
        
        Returns:
            List of predicted positions: [(x, y), ...]
        """
        history = track_history.get(track_id, [])
        if len(history) < 2:
            return None
        
        # Get current position and velocity
        curr_pos = history[-1]
        prev_pos = history[-2]
        
        velocity = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
        
        # Predict future positions (assuming constant velocity)
        trajectory = []
        for t in range(1, self.prediction_horizon + 1):
            future_pos = (
                curr_pos[0] + velocity[0] * t,
                curr_pos[1] + velocity[1] * t
            )
            trajectory.append(future_pos)
        
        return trajectory
    
    def _check_trajectory_intersection(self, traj1, traj2, bbox1, bbox2):
        """
        Check if two trajectories will intersect
        
        Returns:
            Frame number of predicted collision, or None
        """
        # Get vehicle dimensions
        w1 = (bbox1[2] - bbox1[0]) / 2
        h1 = (bbox1[3] - bbox1[1]) / 2
        w2 = (bbox2[2] - bbox2[0]) / 2
        h2 = (bbox2[3] - bbox2[1]) / 2
        
        # Check each predicted position
        for frame_idx, (pos1, pos2) in enumerate(zip(traj1, traj2)):
            distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            
            # Check if vehicles will be too close
            min_distance = max(w1 + w2, h1 + h2)
            if distance < min_distance + self.collision_threshold:
                return frame_idx + 1  # +1 because frame_idx is 0-based
        
        return None

