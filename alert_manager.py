"""
Alert Manager
Handles all alerts and notifications (SMS, Email, IoT, Logging)
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

class AlertManager:
    def __init__(self, log_dir="logs"):
        """
        Initialize alert manager
        
        Args:
            log_dir: Directory to store alert logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.alert_history = []
        self.alert_counts = {}  # alert_type -> count
        
        # Twilio SMS configuration (optional)
        self.twilio_enabled = False
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_from = os.getenv("TWILIO_PHONE_NUMBER")
        self.twilio_to = os.getenv("ALERT_PHONE_NUMBER")
        
        if self.twilio_sid and self.twilio_token:
            try:
                from twilio.rest import Client
                self.twilio_client = Client(self.twilio_sid, self.twilio_token)
                self.twilio_enabled = True
                print("✅ Twilio SMS enabled")
            except ImportError:
                print("⚠️ Twilio not installed. Install with: pip install twilio")
        
        print("✅ Alert Manager initialized")
    
    def send_alert(self, alert_type: str, message: str, severity: str = "high", 
                   location: Optional[Dict] = None, metadata: Optional[Dict] = None):
        """
        Send alert through all configured channels
        
        Args:
            alert_type: Type of alert (e.g., "collision", "wrong_way")
            message: Alert message
            severity: Alert severity ("low", "medium", "high", "critical")
            location: Optional location data
            metadata: Optional additional metadata
        """
        timestamp = datetime.now().isoformat()
        
        alert_data = {
            "timestamp": timestamp,
            "type": alert_type,
            "message": message,
            "severity": severity,
            "location": location or {},
            "metadata": metadata or {}
        }
        
        # Log alert
        self._log_alert(alert_data)
        
        # Add to history
        self.alert_history.append(alert_data)
        
        # Update counts
        self.alert_counts[alert_type] = self.alert_counts.get(alert_type, 0) + 1
        
        # Print alert
        emoji_map = {
            "collision": "🚨",
            "chain_crash": "⚠️",
            "wrong_way": "🚧",
            "skid": "⚠",
            "stationary": "⛔",
            "debris": "🟠",
            "fall": "🚨",
            "pedestrian_hit": "🚑",
            "person_down": "⚰",
            "fire": "🔥",
            "smoke": "💨"
        }
        emoji = emoji_map.get(alert_type, "⚠️")
        print(f"{emoji} ALERT [{severity.upper()}]: {message}")
        
        # Send SMS if enabled and critical
        if self.twilio_enabled and severity in ["high", "critical"]:
            self._send_sms(message)
        
        # Keep only last 1000 alerts in memory
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        return obj
    
    def _log_alert(self, alert_data: Dict):
        """Log alert to file"""
        log_file = os.path.join(self.log_dir, f"alerts_{datetime.now().strftime('%Y-%m-%d')}.jsonl")
        
        # Convert numpy types to native Python types
        alert_data_clean = self._convert_numpy_types(alert_data)
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert_data_clean, ensure_ascii=False) + "\n")
    
    def _send_sms(self, message: str):
        """Send SMS via Twilio"""
        if not self.twilio_enabled or not self.twilio_to:
            return
        
        try:
            self.twilio_client.messages.create(
                body=f"🚨 Accident Alert: {message}",
                from_=self.twilio_from,
                to=self.twilio_to
            )
            print(f"📱 SMS sent to {self.twilio_to}")
        except Exception as e:
            print(f"⚠️ Failed to send SMS: {e}")
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts"""
        return {
            "total_alerts": len(self.alert_history),
            "alert_counts": self.alert_counts.copy(),
            "recent_alerts": self.alert_history[-10:] if self.alert_history else []
        }
    
    def get_alerts_by_type(self, alert_type: str) -> List[Dict]:
        """Get all alerts of a specific type"""
        return [alert for alert in self.alert_history if alert["type"] == alert_type]
    
    def clear_history(self):
        """Clear alert history (keep logs)"""
        self.alert_history = []
        self.alert_counts = {}
