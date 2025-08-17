"""
Bio-Rhythm Analyzer Service
Analyzes wearable device data and predicts energy levels for smart scheduling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from collections import deque
import json

from neuroalign.utils.config import settings


class BioRhythmAnalyzer:
    """Bio-rhythm analysis and energy prediction service"""
    
    def __init__(self):
        """Initialize bio-rhythm analyzer"""
        # Data storage
        self.heart_rate_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.sleep_history = deque(maxlen=30)  # 30 days of sleep data
        self.activity_history = deque(maxlen=1440)  # 24 hours of activity data
        self.energy_predictions = deque(maxlen=24)  # 24 hours of predictions
        
        # Energy prediction model parameters
        self.energy_prediction_horizon = settings.ENERGY_PREDICTION_HORIZON
        self.sleep_cycle_length = settings.SLEEP_CYCLE_LENGTH
        self.optimal_energy_window = settings.OPTIMAL_ENERGY_WINDOW
        
        # Baseline values (would be personalized per user)
        self.baseline_heart_rate = 70.0
        self.baseline_sleep_duration = 8.0  # hours
        self.baseline_activity_level = 10000  # steps per day
        
        print("⏰ Bio-Rhythm Analyzer initialized successfully")
    
    async def predict_energy(self, bio_signals: Dict) -> Dict[str, float]:
        """
        Predict energy levels based on bio-signals
        
        Args:
            bio_signals: Dictionary containing heart rate, sleep, activity data
            
        Returns:
            Dictionary with energy predictions and confidence scores
        """
        try:
            # Extract bio-signals
            heart_rate = bio_signals.get("heart_rate")
            sleep_duration = bio_signals.get("sleep_duration")
            sleep_quality = bio_signals.get("sleep_quality")
            steps_count = bio_signals.get("steps_count")
            stress_level = bio_signals.get("stress_level")
            
            # Update historical data
            if heart_rate:
                self.heart_rate_history.append({
                    "timestamp": datetime.now(),
                    "heart_rate": heart_rate
                })
            
            if sleep_duration:
                self.sleep_history.append({
                    "timestamp": datetime.now(),
                    "duration": sleep_duration,
                    "quality": sleep_quality or 0.7
                })
            
            if steps_count:
                self.activity_history.append({
                    "timestamp": datetime.now(),
                    "steps": steps_count
                })
            
            # Calculate energy predictions
            current_energy = self._calculate_current_energy(
                heart_rate, sleep_duration, sleep_quality, steps_count, stress_level
            )
            
            # Predict future energy levels
            future_energy = self._predict_future_energy(current_energy)
            
            # Calculate optimal scheduling windows
            optimal_windows = self._calculate_optimal_windows(future_energy)
            
            # Generate recommendations
            recommendations = self._generate_energy_recommendations(
                current_energy, future_energy, optimal_windows
            )
            
            return {
                "current_energy_level": current_energy,
                "future_energy_prediction": future_energy,
                "optimal_windows": optimal_windows,
                "recommendations": recommendations,
                "confidence_score": self._calculate_confidence_score(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error predicting energy: {e}")
            return {
                "current_energy_level": 0.5,
                "future_energy_prediction": [0.5] * 24,
                "optimal_windows": [],
                "recommendations": ["Unable to analyze bio-rhythms"],
                "confidence_score": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def optimize_schedule(self, schedule_data: Dict, user_id: int) -> Dict:
        """
        Optimize schedule based on predicted energy levels
        
        Args:
            schedule_data: Dictionary with schedule information
            user_id: User identifier
            
        Returns:
            Dictionary with optimized schedule recommendations
        """
        try:
            # Get current energy prediction
            current_time = datetime.now()
            hour = current_time.hour
            
            # Get energy prediction for the day
            energy_prediction = await self._get_daily_energy_prediction(user_id)
            
            # Analyze schedule items
            schedule_items = schedule_data.get("items", [])
            optimized_items = []
            
            for item in schedule_items:
                # Get original timing
                original_start = datetime.fromisoformat(item["start_time"])
                original_end = datetime.fromisoformat(item["end_time"])
                duration = (original_end - original_start).total_seconds() / 3600  # hours
                
                # Calculate energy requirement
                energy_requirement = item.get("energy_requirement", 0.5)
                priority = item.get("priority", 3)
                complexity = item.get("complexity", 0.5)
                
                # Find optimal time slot
                optimal_start = self._find_optimal_timeslot(
                    energy_prediction, energy_requirement, duration, priority, complexity
                )
                
                # Calculate fatigue risk
                fatigue_risk = self._calculate_fatigue_risk(
                    energy_prediction, optimal_start, duration, energy_requirement
                )
                
                optimized_items.append({
                    "original_start": item["start_time"],
                    "original_end": item["end_time"],
                    "recommended_start": optimal_start.isoformat(),
                    "recommended_end": (optimal_start + timedelta(hours=duration)).isoformat(),
                    "energy_prediction": energy_prediction[optimal_start.hour],
                    "fatigue_risk": fatigue_risk,
                    "optimization_score": self._calculate_optimization_score(
                        energy_prediction[optimal_start.hour], fatigue_risk, priority
                    )
                })
            
            # Sort by optimization score
            optimized_items.sort(key=lambda x: x["optimization_score"], reverse=True)
            
            return {
                "original_schedule": schedule_items,
                "optimized_schedule": optimized_items,
                "energy_prediction": energy_prediction,
                "optimization_summary": self._generate_optimization_summary(optimized_items),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error optimizing schedule: {e}")
            return {
                "original_schedule": schedule_data.get("items", []),
                "optimized_schedule": [],
                "energy_prediction": [0.5] * 24,
                "optimization_summary": "Unable to optimize schedule",
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_current_energy(self, heart_rate: Optional[float], 
                                sleep_duration: Optional[float],
                                sleep_quality: Optional[float],
                                steps_count: Optional[int],
                                stress_level: Optional[float]) -> float:
        """Calculate current energy level from bio-signals"""
        energy_components = []
        
        # Heart rate component (normalized)
        if heart_rate:
            hr_energy = max(0, 1 - abs(heart_rate - self.baseline_heart_rate) / 50)
            energy_components.append(hr_energy)
        
        # Sleep component
        if sleep_duration:
            sleep_energy = min(1.0, sleep_duration / self.baseline_sleep_duration)
            if sleep_quality:
                sleep_energy *= sleep_quality
            energy_components.append(sleep_energy)
        
        # Activity component
        if steps_count:
            activity_energy = min(1.0, steps_count / self.baseline_activity_level)
            energy_components.append(activity_energy)
        
        # Stress component (inverse relationship)
        if stress_level:
            stress_energy = 1.0 - stress_level
            energy_components.append(stress_energy)
        
        # Calculate weighted average
        if energy_components:
            weights = [0.3, 0.4, 0.2, 0.1]  # HR, Sleep, Activity, Stress
            return np.average(energy_components[:len(weights)], weights=weights[:len(energy_components)])
        
        return 0.5  # Default energy level
    
    def _predict_future_energy(self, current_energy: float) -> List[float]:
        """Predict energy levels for the next 24 hours"""
        predictions = []
        current_hour = datetime.now().hour
        
        for hour in range(24):
            # Base circadian rhythm (simplified)
            circadian_factor = self._calculate_circadian_factor(hour)
            
            # Sleep debt factor
            sleep_debt_factor = self._calculate_sleep_debt_factor()
            
            # Activity factor
            activity_factor = self._calculate_activity_factor(hour)
            
            # Combine factors
            predicted_energy = (
                current_energy * 0.3 +
                circadian_factor * 0.4 +
                sleep_debt_factor * 0.2 +
                activity_factor * 0.1
            )
            
            # Ensure energy is between 0 and 1
            predicted_energy = max(0.0, min(1.0, predicted_energy))
            predictions.append(predicted_energy)
        
        return predictions
    
    def _calculate_circadian_factor(self, hour: int) -> float:
        """Calculate circadian rhythm factor for given hour"""
        # Simplified circadian rhythm model
        # Peak energy around 9-11 AM, dip around 2-4 PM, low energy late night
        if 9 <= hour <= 11:
            return 0.9  # Peak energy
        elif 14 <= hour <= 16:
            return 0.4  # Afternoon dip
        elif 22 <= hour or hour <= 6:
            return 0.2  # Low energy (night)
        else:
            return 0.7  # Moderate energy
    
    def _calculate_sleep_debt_factor(self) -> float:
        """Calculate sleep debt factor from recent sleep history"""
        if not self.sleep_history:
            return 0.5
        
        # Calculate average sleep duration over last 7 days
        recent_sleep = list(self.sleep_history)[-7:]
        avg_sleep = np.mean([s["duration"] for s in recent_sleep])
        
        # Calculate sleep debt
        sleep_debt = max(0, self.baseline_sleep_duration - avg_sleep)
        sleep_debt_factor = max(0.3, 1.0 - sleep_debt / 4.0)  # Reduce energy if sleep deprived
        
        return sleep_debt_factor
    
    def _calculate_activity_factor(self, hour: int) -> float:
        """Calculate activity factor for given hour"""
        if not self.activity_history:
            return 0.5
        
        # Get activity level for this hour (simplified)
        # In practice, this would analyze historical activity patterns
        return 0.7  # Placeholder
    
    def _calculate_optimal_windows(self, energy_prediction: List[float]) -> List[Dict]:
        """Calculate optimal scheduling windows based on energy prediction"""
        optimal_windows = []
        
        # Find high energy periods (>0.7)
        high_energy_hours = [i for i, energy in enumerate(energy_prediction) if energy > 0.7]
        
        # Group consecutive hours
        if high_energy_hours:
            start_hour = high_energy_hours[0]
            end_hour = start_hour
            
            for i in range(1, len(high_energy_hours)):
                if high_energy_hours[i] == high_energy_hours[i-1] + 1:
                    end_hour = high_energy_hours[i]
                else:
                    # Add window
                    optimal_windows.append({
                        "start_hour": start_hour,
                        "end_hour": end_hour,
                        "duration": end_hour - start_hour + 1,
                        "avg_energy": np.mean(energy_prediction[start_hour:end_hour+1])
                    })
                    start_hour = high_energy_hours[i]
                    end_hour = start_hour
            
            # Add last window
            optimal_windows.append({
                "start_hour": start_hour,
                "end_hour": end_hour,
                "duration": end_hour - start_hour + 1,
                "avg_energy": np.mean(energy_prediction[start_hour:end_hour+1])
            })
        
        return optimal_windows
    
    def _generate_energy_recommendations(self, current_energy: float, 
                                       future_energy: List[float],
                                       optimal_windows: List[Dict]) -> List[str]:
        """Generate recommendations based on energy analysis"""
        recommendations = []
        
        # Current energy recommendations
        if current_energy < 0.3:
            recommendations.append("Consider taking a break or rest period")
        elif current_energy < 0.5:
            recommendations.append("Focus on lighter tasks or take a short break")
        elif current_energy > 0.8:
            recommendations.append("Great energy level! Tackle challenging tasks")
        
        # Optimal window recommendations
        if optimal_windows:
            best_window = max(optimal_windows, key=lambda x: x["avg_energy"])
            recommendations.append(
                f"Schedule important tasks between {best_window['start_hour']}:00 and {best_window['end_hour']}:00"
            )
        
        # Sleep recommendations
        if self.sleep_history:
            recent_sleep = list(self.sleep_history)[-3:]
            avg_sleep = np.mean([s["duration"] for s in recent_sleep])
            if avg_sleep < 7.0:
                recommendations.append("Consider getting more sleep to improve energy levels")
        
        return recommendations
    
    async def _get_daily_energy_prediction(self, user_id: int) -> List[float]:
        """Get daily energy prediction for user"""
        # This would query user-specific data and models
        # For now, return a generic prediction
        return [0.5 + 0.3 * np.sin(2 * np.pi * i / 24) for i in range(24)]
    
    def _find_optimal_timeslot(self, energy_prediction: List[float], 
                             energy_requirement: float, duration: float,
                             priority: int, complexity: float) -> datetime:
        """Find optimal time slot for a task"""
        current_time = datetime.now()
        best_start_hour = current_time.hour
        best_score = -1
        
        # Search for best time slot
        for hour in range(24):
            # Check if slot has enough energy
            slot_energy = energy_prediction[hour]
            if slot_energy < energy_requirement:
                continue
            
            # Calculate score based on energy, priority, and complexity
            score = slot_energy * priority * (1 + complexity)
            
            if score > best_score:
                best_score = score
                best_start_hour = hour
        
        # Create datetime for recommended start time
        recommended_time = current_time.replace(
            hour=best_start_hour, minute=0, second=0, microsecond=0
        )
        
        # If recommended time is in the past, move to next day
        if recommended_time <= current_time:
            recommended_time += timedelta(days=1)
        
        return recommended_time
    
    def _calculate_fatigue_risk(self, energy_prediction: List[float], 
                              start_time: datetime, duration: float,
                              energy_requirement: float) -> float:
        """Calculate fatigue risk for a task"""
        start_hour = start_time.hour
        end_hour = int((start_time + timedelta(hours=duration)).hour)
        
        # Calculate average energy during task
        task_energy = np.mean(energy_prediction[start_hour:end_hour+1])
        
        # Fatigue risk is inverse to available energy
        fatigue_risk = max(0, energy_requirement - task_energy)
        
        return min(1.0, fatigue_risk)
    
    def _calculate_optimization_score(self, energy_level: float, 
                                    fatigue_risk: float, priority: int) -> float:
        """Calculate optimization score for a schedule item"""
        # Higher energy, lower fatigue risk, and higher priority = better score
        score = energy_level * (1 - fatigue_risk) * priority
        return score
    
    def _generate_optimization_summary(self, optimized_items: List[Dict]) -> str:
        """Generate summary of schedule optimization"""
        if not optimized_items:
            return "No schedule items to optimize"
        
        total_items = len(optimized_items)
        high_priority_items = len([item for item in optimized_items if item.get("priority", 0) >= 4])
        avg_energy = np.mean([item["energy_prediction"] for item in optimized_items])
        avg_fatigue_risk = np.mean([item["fatigue_risk"] for item in optimized_items])
        
        return f"Optimized {total_items} items ({high_priority_items} high priority). " \
               f"Average energy: {avg_energy:.2f}, Fatigue risk: {avg_fatigue_risk:.2f}"
    
    def _calculate_confidence_score(self) -> float:
        """Calculate confidence score for predictions"""
        # Base confidence on amount of available data
        hr_data_points = len(self.heart_rate_history)
        sleep_data_points = len(self.sleep_history)
        activity_data_points = len(self.activity_history)
        
        # More data = higher confidence
        confidence = min(1.0, (hr_data_points + sleep_data_points + activity_data_points) / 100)
        
        return confidence