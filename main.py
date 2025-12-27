"""
Main Accident Detection System
Real-time road safety monitoring with AI-based accident detection
"""
import os
import cv2
import numpy as np
from datetime import datetime
import argparse

# Import detectors
from detectors.vehicle_detector import VehicleDetector
from detectors.tracker_deepsort import VehicleTracker
from detectors.pose_detector import PoseDetector
from detectors.fire_detector import FireDetector

# Import logic modules
from logic.crash_logic import CrashDetector
from logic.chain_logic import ChainCrashDetector
from logic.wrong_way_logic import WrongWayDetector
from logic.skid_logic import SkidDetector
from logic.stationary_logic import StationaryDetector
from logic.debris_logic import DebrisDetector
from logic.fall_logic import FallDetector
from logic.motion_prediction import MotionPredictor
from logic.alert_manager import AlertManager

class AccidentDetectionSystem:
    def __init__(self, video_source, fps=30, enable_pose=False, enable_fire=False):
        """
        Initialize the accident detection system
        
        Args:
            video_source: Path to video file or RTSP URL
            fps: Frames per second (for time calculations)
            enable_pose: Enable human pose detection (Phase-2)
            enable_fire: Enable fire detection (Phase-3)
        """
        self.video_source = video_source
        self.fps = fps
        self.frame_count = 0
        
        # Initialize detectors
        print("🔧 Initializing detectors...")
        self.vehicle_detector = VehicleDetector()
        self.tracker = VehicleTracker()
        self.alert_manager = AlertManager()
        
        if enable_pose:
            self.pose_detector = PoseDetector()
        else:
            self.pose_detector = None
        
        if enable_fire:
            self.fire_detector = FireDetector()
        else:
            self.fire_detector = None
        
        # Initialize logic modules
        print("🔧 Initializing logic modules...")
        self.crash_detector = CrashDetector()
        self.chain_detector = ChainCrashDetector()
        self.wrong_way_detector = WrongWayDetector()
        self.skid_detector = SkidDetector()
        self.stationary_detector = StationaryDetector(time_threshold=1.0)
        self.debris_detector = DebrisDetector()
        self.fall_detector = FallDetector()
        self.motion_predictor = MotionPredictor()
        
        # Tracking data
        self.track_history = {}
        self.track_speeds = {}
        self.processed_alerts = set()  # To avoid duplicate alerts
        self.current_pose_detections = []  # Store current frame pose detections
        self.is_static_image = False  # Flag to indicate if processing static image
        
        print("✅ Accident Detection System initialized")
    
    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1
        
        # Detect vehicles
        vehicle_detections = self.vehicle_detector.detect(frame)
        
        # Track vehicles
        tracked_vehicles = self.tracker.update(vehicle_detections, frame)
        
        # Update tracking data
        for vehicle in tracked_vehicles:
            track_id = vehicle[4]
            center = self.vehicle_detector.get_center(vehicle)
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(center)
            
            # Keep only last 20 positions
            if len(self.track_history[track_id]) > 20:
                self.track_history[track_id] = self.track_history[track_id][-20:]
            
            # Calculate speed
            if len(self.track_history[track_id]) >= 2:
                prev = self.track_history[track_id][-2]
                curr = self.track_history[track_id][-1]
                speed = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                # Store speed as list (for history tracking)
                if track_id not in self.track_speeds:
                    self.track_speeds[track_id] = []
                self.track_speeds[track_id].append(float(speed))  # Convert to float
                # Keep only last 10 speeds
                if len(self.track_speeds[track_id]) > 10:
                    self.track_speeds[track_id] = self.track_speeds[track_id][-10:]
        
        # Phase-1: Vehicle-related detections
        alerts = []
        
        # V1, V3: Crash detection
        crashes = self.crash_detector.detect_crash(
            tracked_vehicles, self.track_speeds, self.track_history, 
            is_static_image=self.is_static_image
        )
        for track_id1, track_id2, crash_type in crashes:
            alert_key = f"crash_{track_id1}_{track_id2}"
            if alert_key not in self.processed_alerts:
                self.alert_manager.send_alert(
                    "collision",
                    f"🚨 Vehicle Collision Detected ({crash_type})",
                    severity="critical",
                    metadata={"track_id1": track_id1, "track_id2": track_id2, "type": crash_type}
                )
                self.processed_alerts.add(alert_key)
                alerts.append(("collision", track_id1, track_id2))
        
        # V2: Chain crash
        chain_groups = self.chain_detector.detect_chain_crash(tracked_vehicles, crashes)
        for group in chain_groups:
            alert_key = f"chain_{'_'.join(map(str, sorted(group)))}"
            if alert_key not in self.processed_alerts:
                self.alert_manager.send_alert(
                    "chain_crash",
                    f"⚠️ Multi-Car Chain Collision ({len(group)} vehicles)",
                    severity="critical",
                    metadata={"vehicle_count": len(group), "track_ids": group}
                )
                self.processed_alerts.add(alert_key)
                alerts.append(("chain_crash", group))
        
        # V4: Wrong-way detection
        wrong_way_ids = self.wrong_way_detector.detect_wrong_way(
            tracked_vehicles, self.track_history
        )
        for track_id in wrong_way_ids:
            alert_key = f"wrong_way_{track_id}"
            if alert_key not in self.processed_alerts:
                self.alert_manager.send_alert(
                    "wrong_way",
                    f"🚧 Wrong-Way Vehicle Detected",
                    severity="high",
                    metadata={"track_id": track_id}
                )
                self.processed_alerts.add(alert_key)
                alerts.append(("wrong_way", track_id))
        
        # V5: Skid detection
        skidding_ids = self.skid_detector.detect_skid(
            tracked_vehicles, self.track_history, self.track_speeds
        )
        for track_id in skidding_ids:
            alert_key = f"skid_{track_id}"
            if alert_key not in self.processed_alerts:
                self.alert_manager.send_alert(
                    "skid",
                    f"⚠ Skid Event - Loss of Control",
                    severity="high",
                    metadata={"track_id": track_id}
                )
                self.processed_alerts.add(alert_key)
                alerts.append(("skid", track_id))
        
        # V6: Stationary vehicle
        stationary_ids = self.stationary_detector.detect_stationary(
            tracked_vehicles, self.track_speeds, self.fps
        )
        for track_id in stationary_ids:
            alert_key = f"stationary_{track_id}"
            if alert_key not in self.processed_alerts:
                self.alert_manager.send_alert(
                    "stationary",
                    f"⛔ Vehicle Stopped (Stall)",
                    severity="medium",
                    metadata={"track_id": track_id}
                )
                self.processed_alerts.add(alert_key)
                alerts.append(("stationary", track_id))
        
        # V7: Debris detection
        debris_regions = self.debris_detector.detect_debris(
            frame, tracked_vehicles, vehicle_detections
        )
        if debris_regions:
            alert_key = f"debris_{self.frame_count}"
            if alert_key not in self.processed_alerts:
                self.alert_manager.send_alert(
                    "debris",
                    f"🟠 Debris Detected — Road Block Risk",
                    severity="medium",
                    metadata={"regions": len(debris_regions)}
                )
                self.processed_alerts.add(alert_key)
                alerts.append(("debris", debris_regions))
        
        # V8: Motion prediction
        predicted_collisions = self.motion_predictor.predict_collision(
            tracked_vehicles, self.track_history, self.track_speeds
        )
        for track_id1, track_id2, time_to_collision in predicted_collisions:
            alert_key = f"prediction_{track_id1}_{track_id2}"
            if alert_key not in self.processed_alerts:
                self.alert_manager.send_alert(
                    "collision",
                    f"⚠️ Potential Collision Predicted (Pre-Alert)",
                    severity="medium",
                    metadata={"track_id1": track_id1, "track_id2": track_id2, 
                             "time_to_collision": time_to_collision}
                )
                self.processed_alerts.add(alert_key)
                alerts.append(("prediction", track_id1, track_id2))
        
        # Phase-2: Human safety (if enabled)
        if self.pose_detector:
            pose_detections = self.pose_detector.detect(frame)
            self.current_pose_detections = pose_detections  # Store for drawing
            
            # Debug: Print number of humans detected (only first few frames)
            if self.frame_count <= 5 and len(pose_detections) > 0:
                print(f"👤 Detected {len(pose_detections)} person(s) in frame {self.frame_count}")
            
            # H1: Fall detection
            fall_events = self.fall_detector.detect_fall(pose_detections)
            for bbox, fall_type in fall_events:
                alert_key = f"fall_{self.frame_count}"
                if alert_key not in self.processed_alerts:
                    self.alert_manager.send_alert(
                        "fall",
                        f"🚨 Human Fall Injury Detected",
                        severity="critical",
                        metadata={"type": fall_type}
                    )
                    self.processed_alerts.add(alert_key)
                    alerts.append(("fall", bbox))
            
            # H2: Pedestrian hit
            pedestrian_hits = self.fall_detector.detect_pedestrian_hit(
                pose_detections, tracked_vehicles, crashes
            )
            for bbox, vehicle_id in pedestrian_hits:
                alert_key = f"pedestrian_hit_{vehicle_id}"
                if alert_key not in self.processed_alerts:
                    self.alert_manager.send_alert(
                        "pedestrian_hit",
                        f"🚑 Pedestrian Crash",
                        severity="critical",
                        metadata={"vehicle_id": vehicle_id}
                    )
                    self.processed_alerts.add(alert_key)
                    alerts.append(("pedestrian_hit", bbox, vehicle_id))
        
        # Phase-3: Post-accident events (if enabled)
        if self.fire_detector:
            has_fire, has_smoke, fire_regions = self.fire_detector.detect(frame)
            if has_fire:
                alert_key = f"fire_{self.frame_count}"
                if alert_key not in self.processed_alerts:
                    self.alert_manager.send_alert(
                        "fire",
                        f"🔥 Fire Detected After Crash",
                        severity="critical",
                        metadata={"regions": len(fire_regions)}
                    )
                    self.processed_alerts.add(alert_key)
                    alerts.append(("fire", fire_regions))
            
            if has_smoke:
                alert_key = f"smoke_{self.frame_count}"
                if alert_key not in self.processed_alerts:
                    self.alert_manager.send_alert(
                        "smoke",
                        f"💨 Smoke Detected",
                        severity="high",
                        metadata={}
                    )
                    self.processed_alerts.add(alert_key)
                    alerts.append(("smoke", None))
        
        return tracked_vehicles, alerts
    
    def draw_detections(self, frame, tracked_vehicles, alerts):
        """Draw detections and alerts on frame"""
        # Draw vehicle bounding boxes
        for vehicle in tracked_vehicles:
            x1, y1, x2, y2, track_id, conf, cls_id = vehicle
            
            # Color based on class
            colors = {
                2: (0, 255, 0),    # car - green
                3: (255, 0, 0),    # motorcycle - blue
                5: (0, 0, 255),    # bus - red
                7: (255, 255, 0)   # truck - cyan
            }
            color = colors.get(cls_id, (255, 255, 255))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw human/pedestrian detections (if pose detection is enabled)
        if self.pose_detector and hasattr(self, 'current_pose_detections'):
            for pose_det in self.current_pose_detections:
                x1, y1, x2, y2, conf, keypoints = pose_det
                # Draw human bounding box in purple/cyan color
                human_color = (255, 0, 255)  # Magenta/Purple
                cv2.rectangle(frame, (x1, y1), (x2, y2), human_color, 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, human_color, 2)
                
                # Optionally draw keypoints (skeleton)
                if keypoints is not None:
                    # Draw key points
                    for kp in keypoints:
                        if len(kp) >= 3 and kp[2] > 0.5:  # confidence > 0.5
                            x, y = int(kp[0]), int(kp[1])
                            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Yellow dots
        
        # Draw alert indicators
        alert_colors = {
            "collision": (0, 0, 255),
            "chain_crash": (0, 0, 255),
            "wrong_way": (0, 165, 255),
            "skid": (0, 255, 255),
            "stationary": (128, 128, 128),
            "debris": (0, 165, 255),
            "fall": (255, 0, 255),
            "pedestrian_hit": (255, 0, 255),
            "fire": (0, 0, 255),
            "smoke": (128, 128, 128)
        }
        
        for alert_type, *data in alerts:
            if alert_type in ["collision", "chain_crash"]:
                # Highlight vehicles involved
                if alert_type == "collision" and len(data) >= 2:
                    for vehicle in tracked_vehicles:
                        if vehicle[4] in [data[0], data[1]]:
                            x1, y1, x2, y2 = vehicle[:4]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                elif alert_type == "chain_crash":
                    for vehicle in tracked_vehicles:
                        if vehicle[4] in data[0]:
                            x1, y1, x2, y2 = vehicle[:4]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Draw alert text
            color = alert_colors.get(alert_type, (255, 255, 255))
            alert_text = alert_type.replace("_", " ").title()
            cv2.putText(frame, f"ALERT: {alert_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw frame info
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, show_video=True):
        """Run the accident detection system"""
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source: {self.video_source}")
            return
        
        print(f"🎥 Processing video: {self.video_source}")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("📌 Video finished")
                break
            
            # Process frame
            tracked_vehicles, alerts = self.process_frame(frame)
            
            # Draw detections
            frame = self.draw_detections(frame, tracked_vehicles, alerts)
            
            if show_video:
                cv2.imshow("Accident Detection AI System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        summary = self.alert_manager.get_alert_summary()
        print("\n📊 Alert Summary:")
        print(f"Total Alerts: {summary['total_alerts']}")
        print("Alert Counts by Type:")
        for alert_type, count in summary['alert_counts'].items():
            print(f"  {alert_type}: {count}")
        
        print(f"\n✅ Processing complete. Alerts logged to: {self.alert_manager.log_dir}")


def main():
    parser = argparse.ArgumentParser(description="Accident Detection AI System")
    parser.add_argument("--video", type=str, required=True,
                       help="Path to video file or RTSP URL")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second (default: 30)")
    parser.add_argument("--pose", action="store_true",
                       help="Enable human pose detection (Phase-2)")
    parser.add_argument("--fire", action="store_true",
                       help="Enable fire detection (Phase-3)")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable video display (headless mode)")
    
    args = parser.parse_args()
    
    # Initialize system
    system = AccidentDetectionSystem(
        video_source=args.video,
        fps=args.fps,
        enable_pose=args.pose,
        enable_fire=args.fire
    )
    
    # Run system
    system.run(show_video=not args.no_display)


if __name__ == "__main__":
    main()
