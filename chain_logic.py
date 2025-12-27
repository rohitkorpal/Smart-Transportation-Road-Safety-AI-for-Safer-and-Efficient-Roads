"""
Chain Crash Detection Logic
Detects multi-vehicle chain collisions (V2 scenario)
"""
import numpy as np

class ChainCrashDetector:
    def __init__(self, chain_threshold=3, proximity_threshold=80):
        """
        Initialize chain crash detector
        
        Args:
            chain_threshold: Minimum number of vehicles for chain crash
            proximity_threshold: Maximum distance between vehicles in chain
        """
        self.chain_threshold = chain_threshold
        self.proximity_threshold = proximity_threshold
        print("✅ Chain Crash Detector initialized")
    
    def calculate_distance(self, bbox1, bbox2):
        """Calculate distance between centers of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
        
        center1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
        center2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def detect_chain_crash(self, tracked_vehicles, crash_events):
        """
        Detect chain crashes (3+ vehicles in collision)
        
        Args:
            tracked_vehicles: List of tracked vehicles
            crash_events: List of detected crashes from crash_logic
            
        Returns:
            List of chain crash groups: [[track_id1, track_id2, ...], ...]
        """
        if len(crash_events) < 2:
            return []
        
        # Build graph of connected vehicles
        connections = {}
        for track_id1, track_id2, _ in crash_events:
            if track_id1 not in connections:
                connections[track_id1] = set()
            if track_id2 not in connections:
                connections[track_id2] = set()
            connections[track_id1].add(track_id2)
            connections[track_id2].add(track_id1)
        
        # Find connected components (chain groups)
        visited = set()
        chain_groups = []
        
        def dfs(node, group):
            if node in visited:
                return
            visited.add(node)
            group.add(node)
            for neighbor in connections.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, group)
        
        for track_id in connections:
            if track_id not in visited:
                group = set()
                dfs(track_id, group)
                if len(group) >= self.chain_threshold:
                    chain_groups.append(list(group))
        
        return chain_groups
    
    def detect_proximity_chain(self, tracked_vehicles):
        """
        Detect potential chain crash based on proximity
        (Vehicles close together in sequence)
        
        Args:
            tracked_vehicles: List of tracked vehicles
            
        Returns:
            List of potential chain groups
        """
        if len(tracked_vehicles) < self.chain_threshold:
            return []
        
        chain_groups = []
        
        # Check all combinations
        for i in range(len(tracked_vehicles)):
            group = [tracked_vehicles[i][4]]  # track_id
            for j in range(i + 1, len(tracked_vehicles)):
                distance = self.calculate_distance(tracked_vehicles[i], tracked_vehicles[j])
                if distance < self.proximity_threshold:
                    group.append(tracked_vehicles[j][4])
            
            if len(group) >= self.chain_threshold:
                chain_groups.append(group)
        
        return chain_groups
