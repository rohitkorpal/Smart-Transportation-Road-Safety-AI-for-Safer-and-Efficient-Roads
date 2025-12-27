"""
Wrong-Way Driving Detection Logic
Detects vehicles moving opposite to traffic flow (V4 scenario)
"""
import numpy as np

class WrongWayDetector:
    def __init__(self, direction_threshold=0.7):
        """
        Initialize wrong-way detector
        
        Args:
            direction_threshold: Threshold for direction similarity (0-1)
        """
        self.direction_threshold = direction_threshold
        self.expected_direction = None  # Will be learned from traffic flow
        self.direction_history = []  # History of vehicle directions
        print("✅ Wrong-Way Detector initialized")
    
    def learn_traffic_flow(self, tracked_vehicles, track_history):
        """
        Learn expected traffic flow direction from majority of vehicles
        
        Args:
            tracked_vehicles: List of tracked vehicles
            track_history: Dict of track histories
        """
        if len(tracked_vehicles) < 3:
            return  # Need enough vehicles to learn flow
        
        directions = []
        for vehicle in tracked_vehicles:
            track_id = vehicle[4]
            history = track_history.get(track_id, [])
            if len(history) >= 2:
                prev = history[-2]
                curr = history[-1]
                direction = (curr[0] - prev[0], curr[1] - prev[1])
                # Normalize direction
                mag = np.sqrt(direction[0]**2 + direction[1]**2)
                if mag > 0:
                    direction = (direction[0] / mag, direction[1] / mag)
                    directions.append(direction)
        
        if directions:
            # Calculate average direction
            avg_dir = np.mean(directions, axis=0)
            mag = np.sqrt(avg_dir[0]**2 + avg_dir[1]**2)
            if mag > 0:
                self.expected_direction = (avg_dir[0] / mag, avg_dir[1] / mag)
                self.direction_history.append(self.expected_direction)
                
                # Keep only last 10 directions
                if len(self.direction_history) > 10:
                    self.direction_history = self.direction_history[-10:]
    
    def detect_wrong_way(self, tracked_vehicles, track_history):
        """
        Detect vehicles moving in wrong direction
        
        Args:
            tracked_vehicles: List of tracked vehicles
            track_history: Dict of track histories
            
        Returns:
            List of wrong-way track IDs
        """
        if self.expected_direction is None:
            # Learn flow first
            self.learn_traffic_flow(tracked_vehicles, track_history)
            return []
        
        wrong_way_vehicles = []
        
        for vehicle in tracked_vehicles:
            track_id = vehicle[4]
            history = track_history.get(track_id, [])
            
            if len(history) < 2:
                continue
            
            # Get vehicle direction
            prev = history[-2]
            curr = history[-1]
            vehicle_dir = (curr[0] - prev[0], curr[1] - prev[1])
            
            # Normalize
            mag = np.sqrt(vehicle_dir[0]**2 + vehicle_dir[1]**2)
            if mag < 1:  # Vehicle not moving
                continue
            
            vehicle_dir = (vehicle_dir[0] / mag, vehicle_dir[1] / mag)
            
            # Calculate similarity (dot product)
            similarity = (vehicle_dir[0] * self.expected_direction[0] + 
                         vehicle_dir[1] * self.expected_direction[1])
            
            # If similarity is negative, vehicle is going opposite direction
            if similarity < -self.direction_threshold:
                wrong_way_vehicles.append(track_id)
        
        return wrong_way_vehicles
