"""
Fatigue Detection Service
Analyzes micro-expressions, typing patterns, and bio-signals to detect fatigue
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import asyncio
from datetime import datetime, timedelta
import json
import base64
from collections import deque

from neuroalign.utils.config import settings


class FatigueDetector:
    """Main fatigue detection service"""
    
    def __init__(self):
        """Initialize fatigue detector with ML models and tracking"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Tracking variables
        self.blink_history = deque(maxlen=60)  # 1 minute of data at 1 FPS
        self.typing_history = deque(maxlen=1000)  # Last 1000 keystrokes
        self.fatigue_scores = deque(maxlen=100)  # Last 100 fatigue scores
        
        # Fatigue thresholds
        self.blink_rate_threshold = settings.BLINK_RATE_THRESHOLD
        self.typing_hesitation_threshold = settings.TYPING_HESITATION_THRESHOLD
        self.fatigue_score_threshold = settings.FATIGUE_SCORE_THRESHOLD
        
        # Eye aspect ratio calculation constants
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = 2
        
        # Initialize tracking
        self.counter = 0
        self.total_blinks = 0
        self.last_blink_time = datetime.now()
        
        print("🧠 Fatigue Detector initialized successfully")
    
    async def analyze_frame(self, frame_data: str) -> Dict[str, float]:
        """
        Analyze webcam frame for fatigue indicators
        
        Args:
            frame_data: Base64 encoded image data
            
        Returns:
            Dictionary with fatigue scores and metrics
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Extract fatigue indicators
                blink_rate = self._calculate_blink_rate(landmarks)
                eye_closure = self._calculate_eye_closure(landmarks)
                facial_tension = self._calculate_facial_tension(landmarks)
                smile_frequency = self._calculate_smile_frequency(landmarks)
                
                # Calculate facial fatigue score
                facial_fatigue = self._calculate_facial_fatigue_score(
                    blink_rate, eye_closure, facial_tension, smile_frequency
                )
                
                # Update tracking
                self.blink_history.append(blink_rate)
                
                return {
                    "facial_fatigue_score": facial_fatigue,
                    "blink_rate": blink_rate,
                    "eye_closure_duration": eye_closure,
                    "facial_tension_score": facial_tension,
                    "smile_frequency": smile_frequency,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "facial_fatigue_score": 0.0,
                "blink_rate": 0.0,
                "eye_closure_duration": 0.0,
                "facial_tension_score": 0.0,
                "smile_frequency": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error analyzing frame: {e}")
            return {
                "facial_fatigue_score": 0.0,
                "blink_rate": 0.0,
                "eye_closure_duration": 0.0,
                "facial_tension_score": 0.0,
                "smile_frequency": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_typing(self, typing_data: Dict) -> Dict[str, float]:
        """
        Analyze typing patterns for fatigue indicators
        
        Args:
            typing_data: Dictionary with typing events and timestamps
            
        Returns:
            Dictionary with typing fatigue metrics
        """
        try:
            # Extract typing events
            keystrokes = typing_data.get("keystrokes", [])
            backspaces = typing_data.get("backspaces", [])
            hesitations = typing_data.get("hesitations", [])
            
            # Calculate typing metrics
            typing_speed = self._calculate_typing_speed(keystrokes)
            hesitation_score = self._calculate_hesitation_score(hesitations)
            backspace_frequency = self._calculate_backspace_frequency(backspaces, keystrokes)
            rhythm_variance = self._calculate_rhythm_variance(keystrokes)
            
            # Calculate typing fatigue score
            typing_fatigue = self._calculate_typing_fatigue_score(
                typing_speed, hesitation_score, backspace_frequency, rhythm_variance
            )
            
            # Update typing history
            self.typing_history.append({
                "timestamp": datetime.now(),
                "typing_speed": typing_speed,
                "hesitation_score": hesitation_score,
                "backspace_frequency": backspace_frequency,
                "rhythm_variance": rhythm_variance
            })
            
            return {
                "typing_fatigue_score": typing_fatigue,
                "typing_speed": typing_speed,
                "hesitation_score": hesitation_score,
                "backspace_frequency": backspace_frequency,
                "rhythm_variance": rhythm_variance,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error analyzing typing: {e}")
            return {
                "typing_fatigue_score": 0.0,
                "typing_speed": 0.0,
                "hesitation_score": 0.0,
                "backspace_frequency": 0.0,
                "rhythm_variance": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_overall_fatigue_score(self, user_id: int) -> Dict[str, float]:
        """
        Calculate overall fatigue score combining all data sources
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with overall fatigue assessment
        """
        try:
            # Get recent fatigue data
            recent_facial = list(self.blink_history)[-10:] if self.blink_history else []
            recent_typing = list(self.typing_history)[-10:] if self.typing_history else []
            
            # Calculate weighted fatigue scores
            facial_weight = 0.4
            typing_weight = 0.3
            historical_weight = 0.3
            
            # Current facial fatigue
            current_facial = np.mean(recent_facial) if recent_facial else 0.0
            
            # Current typing fatigue
            current_typing = np.mean([t["typing_speed"] for t in recent_typing]) if recent_typing else 0.0
            
            # Historical fatigue (from database)
            historical_fatigue = await self._get_historical_fatigue(user_id)
            
            # Calculate overall score
            overall_fatigue = (
                facial_weight * current_facial +
                typing_weight * current_typing +
                historical_weight * historical_fatigue
            )
            
            # Determine fatigue level
            fatigue_level = self._classify_fatigue_level(overall_fatigue)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(overall_fatigue, fatigue_level)
            
            return {
                "overall_fatigue_score": overall_fatigue,
                "facial_fatigue_score": current_facial,
                "typing_fatigue_score": current_typing,
                "historical_fatigue_score": historical_fatigue,
                "fatigue_level": fatigue_level,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error calculating overall fatigue: {e}")
            return {
                "overall_fatigue_score": 0.0,
                "facial_fatigue_score": 0.0,
                "typing_fatigue_score": 0.0,
                "historical_fatigue_score": 0.0,
                "fatigue_level": "unknown",
                "recommendations": [],
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_blink_rate(self, landmarks) -> float:
        """Calculate blink rate from facial landmarks"""
        try:
            # Extract eye landmarks
            left_eye = [
                landmarks.landmark[33], landmarks.landmark[7], landmarks.landmark[163],
                landmarks.landmark[144], landmarks.landmark[145], landmarks.landmark[153],
                landmarks.landmark[154], landmarks.landmark[155], landmarks.landmark[133],
                landmarks.landmark[173], landmarks.landmark[157], landmarks.landmark[158],
                landmarks.landmark[159], landmarks.landmark[160], landmarks.landmark[161],
                landmarks.landmark[246]
            ]
            
            right_eye = [
                landmarks.landmark[362], landmarks.landmark[382], landmarks.landmark[381],
                landmarks.landmark[380], landmarks.landmark[374], landmarks.landmark[373],
                landmarks.landmark[390], landmarks.landmark[249], landmarks.landmark[263],
                landmarks.landmark[466], landmarks.landmark[388], landmarks.landmark[387],
                landmarks.landmark[386], landmarks.landmark[385], landmarks.landmark[384],
                landmarks.landmark[398]
            ]
            
            # Calculate eye aspect ratios
            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            
            # Average EAR
            ear = (left_ear + right_ear) / 2.0
            
            # Detect blink
            if ear < self.EYE_AR_THRESH:
                self.counter += 1
            else:
                if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                    self.total_blinks += 1
                self.counter = 0
            
            # Calculate blink rate (blinks per second)
            current_time = datetime.now()
            time_diff = (current_time - self.last_blink_time).total_seconds()
            
            if time_diff >= 60:  # Update every minute
                blink_rate = self.total_blinks / time_diff
                self.total_blinks = 0
                self.last_blink_time = current_time
                return blink_rate
            
            return 0.0
            
        except Exception as e:
            print(f"❌ Error calculating blink rate: {e}")
            return 0.0
    
    def _eye_aspect_ratio(self, eye_landmarks) -> float:
        """Calculate eye aspect ratio"""
        try:
            # Vertical distances
            A = np.linalg.norm([eye_landmarks[1].x - eye_landmarks[5].x, 
                               eye_landmarks[1].y - eye_landmarks[5].y])
            B = np.linalg.norm([eye_landmarks[2].x - eye_landmarks[4].x,
                               eye_landmarks[2].y - eye_landmarks[4].y])
            
            # Horizontal distance
            C = np.linalg.norm([eye_landmarks[0].x - eye_landmarks[3].x,
                               eye_landmarks[0].y - eye_landmarks[3].y])
            
            # Eye aspect ratio
            ear = (A + B) / (2.0 * C)
            return ear
            
        except Exception as e:
            print(f"❌ Error calculating eye aspect ratio: {e}")
            return 0.0
    
    def _calculate_eye_closure(self, landmarks) -> float:
        """Calculate average eye closure duration"""
        # Simplified implementation - in practice, this would track eye closure over time
        return 0.1  # Placeholder
    
    def _calculate_facial_tension(self, landmarks) -> float:
        """Calculate facial tension score"""
        # Simplified implementation - would analyze muscle tension around eyes, mouth
        return 0.3  # Placeholder
    
    def _calculate_smile_frequency(self, landmarks) -> float:
        """Calculate smile frequency"""
        # Simplified implementation - would track smile detection over time
        return 0.2  # Placeholder
    
    def _calculate_facial_fatigue_score(self, blink_rate: float, eye_closure: float, 
                                      facial_tension: float, smile_frequency: float) -> float:
        """Calculate facial fatigue score from multiple indicators"""
        # Normalize blink rate (higher = more fatigue)
        normalized_blink_rate = min(blink_rate / 0.5, 1.0)  # Normalize to 0-1
        
        # Weighted combination
        fatigue_score = (
            0.4 * normalized_blink_rate +
            0.3 * eye_closure +
            0.2 * facial_tension +
            0.1 * (1.0 - smile_frequency)  # Lower smile frequency = higher fatigue
        )
        
        return min(fatigue_score, 1.0)
    
    def _calculate_typing_speed(self, keystrokes: List) -> float:
        """Calculate typing speed in characters per minute"""
        if not keystrokes or len(keystrokes) < 2:
            return 0.0
        
        # Calculate time span
        start_time = keystrokes[0]["timestamp"]
        end_time = keystrokes[-1]["timestamp"]
        duration = (end_time - start_time).total_seconds() / 60.0  # Convert to minutes
        
        if duration <= 0:
            return 0.0
        
        # Calculate characters per minute
        char_count = len([k for k in keystrokes if k["type"] == "keypress"])
        return char_count / duration
    
    def _calculate_hesitation_score(self, hesitations: List) -> float:
        """Calculate hesitation score from typing hesitations"""
        if not hesitations:
            return 0.0
        
        # Count hesitations longer than threshold
        long_hesitations = [h for h in hesitations if h["duration"] > self.typing_hesitation_threshold]
        return len(long_hesitations) / len(hesitations)
    
    def _calculate_backspace_frequency(self, backspaces: List, keystrokes: List) -> float:
        """Calculate backspace frequency"""
        if not keystrokes:
            return 0.0
        
        return len(backspaces) / len(keystrokes)
    
    def _calculate_rhythm_variance(self, keystrokes: List) -> float:
        """Calculate variance in typing rhythm"""
        if len(keystrokes) < 3:
            return 0.0
        
        # Calculate intervals between keystrokes
        intervals = []
        for i in range(1, len(keystrokes)):
            interval = (keystrokes[i]["timestamp"] - keystrokes[i-1]["timestamp"]).total_seconds()
            intervals.append(interval)
        
        # Calculate variance
        if intervals:
            return np.var(intervals)
        return 0.0
    
    def _calculate_typing_fatigue_score(self, typing_speed: float, hesitation_score: float,
                                      backspace_frequency: float, rhythm_variance: float) -> float:
        """Calculate typing fatigue score from multiple indicators"""
        # Normalize metrics (higher values = more fatigue)
        normalized_hesitation = hesitation_score
        normalized_backspace = min(backspace_frequency * 10, 1.0)  # Scale up backspace frequency
        normalized_rhythm = min(rhythm_variance / 0.1, 1.0)  # Normalize rhythm variance
        normalized_speed = 1.0 - min(typing_speed / 200, 1.0)  # Lower speed = higher fatigue
        
        # Weighted combination
        fatigue_score = (
            0.3 * normalized_hesitation +
            0.3 * normalized_backspace +
            0.2 * normalized_rhythm +
            0.2 * normalized_speed
        )
        
        return min(fatigue_score, 1.0)
    
    async def _get_historical_fatigue(self, user_id: int) -> float:
        """Get historical fatigue score from database"""
        # This would query the database for recent fatigue records
        # For now, return a placeholder
        return 0.3
    
    def _classify_fatigue_level(self, fatigue_score: float) -> str:
        """Classify fatigue level based on score"""
        if fatigue_score < 0.3:
            return "low"
        elif fatigue_score < 0.6:
            return "moderate"
        elif fatigue_score < 0.8:
            return "high"
        else:
            return "critical"
    
    def _generate_recommendations(self, fatigue_score: float, fatigue_level: str) -> List[str]:
        """Generate recommendations based on fatigue level"""
        recommendations = []
        
        if fatigue_level == "low":
            recommendations.extend([
                "Continue with current activity",
                "Maintain good posture",
                "Take regular short breaks"
            ])
        elif fatigue_level == "moderate":
            recommendations.extend([
                "Take a 5-minute break",
                "Do some light stretching",
                "Hydrate and have a healthy snack"
            ])
        elif fatigue_level == "high":
            recommendations.extend([
                "Take a 15-minute break",
                "Step away from the screen",
                "Consider switching to less demanding tasks",
                "Practice deep breathing exercises"
            ])
        elif fatigue_level == "critical":
            recommendations.extend([
                "Stop current activity immediately",
                "Take a 30-minute break",
                "Consider ending work session",
                "Get some rest or sleep",
                "Contact supervisor if needed"
            ])
        
        return recommendations