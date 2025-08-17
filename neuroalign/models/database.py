"""
Database models for NeuroAlign
"""

from datetime import datetime, time
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    Text, ForeignKey, JSON, Time, Date
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from pydantic import BaseModel

Base = declarative_base()


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fatigue_records = relationship("FatigueRecord", back_populates="user")
    bio_rhythm_records = relationship("BioRhythmRecord", back_populates="user")
    schedules = relationship("Schedule", back_populates="user")
    wearable_devices = relationship("WearableDevice", back_populates="user")


class FatigueRecord(Base):
    """Fatigue detection records"""
    __tablename__ = "fatigue_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Fatigue scores (0-1 scale)
    overall_fatigue_score = Column(Float)
    facial_fatigue_score = Column(Float)
    typing_fatigue_score = Column(Float)
    
    # Micro-expression data
    blink_rate = Column(Float)  # Blinks per second
    eye_closure_duration = Column(Float)  # Average eye closure time
    facial_tension_score = Column(Float)  # 0-1 scale
    smile_frequency = Column(Float)  # Smiles per minute
    
    # Typing dynamics
    typing_speed = Column(Float)  # Characters per minute
    hesitation_count = Column(Integer)  # Number of hesitations
    backspace_count = Column(Integer)  # Number of backspaces
    typing_rhythm_variance = Column(Float)  # Variance in typing intervals
    
    # Additional metadata
    session_duration = Column(Float)  # Minutes
    activity_type = Column(String)  # "typing", "meeting", "general"
    notes = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="fatigue_records")


class BioRhythmRecord(Base):
    """Bio-rhythm and wearable device data"""
    __tablename__ = "bio_rhythm_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Heart rate data
    heart_rate = Column(Float)
    heart_rate_variability = Column(Float)
    resting_heart_rate = Column(Float)
    
    # Sleep data
    sleep_duration = Column(Float)  # Hours
    sleep_quality = Column(Float)  # 0-1 scale
    deep_sleep_percentage = Column(Float)
    rem_sleep_percentage = Column(Float)
    sleep_efficiency = Column(Float)
    
    # Activity data
    steps_count = Column(Integer)
    calories_burned = Column(Float)
    active_minutes = Column(Integer)
    
    # Energy prediction
    predicted_energy_level = Column(Float)  # 0-1 scale
    energy_confidence = Column(Float)  # 0-1 scale
    
    # Additional bio-signals
    skin_temperature = Column(Float, nullable=True)
    stress_level = Column(Float, nullable=True)  # 0-1 scale
    oxygen_saturation = Column(Float, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="bio_rhythm_records")


class Schedule(Base):
    """Smart scheduling data"""
    __tablename__ = "schedules"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Schedule details
    title = Column(String)
    description = Column(Text, nullable=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    date = Column(Date)
    
    # Task characteristics
    task_type = Column(String)  # "meeting", "work", "break", "exercise"
    priority = Column(Integer)  # 1-5 scale
    energy_requirement = Column(Float)  # 0-1 scale
    complexity = Column(Float)  # 0-1 scale
    
    # AI recommendations
    recommended_start_time = Column(DateTime, nullable=True)
    recommended_end_time = Column(DateTime, nullable=True)
    energy_prediction = Column(Float, nullable=True)
    fatigue_risk = Column(Float, nullable=True)  # 0-1 scale
    
    # Status
    is_completed = Column(Boolean, default=False)
    is_optimized = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="schedules")


class WearableDevice(Base):
    """Wearable device connections"""
    __tablename__ = "wearable_devices"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Device information
    device_type = Column(String)  # "fitbit", "garmin", "apple_watch"
    device_name = Column(String)
    device_id = Column(String)
    
    # API credentials
    access_token = Column(String, nullable=True)
    refresh_token = Column(String, nullable=True)
    token_expires_at = Column(DateTime, nullable=True)
    
    # Connection status
    is_connected = Column(Boolean, default=False)
    last_sync = Column(DateTime, nullable=True)
    
    # Settings
    sync_frequency = Column(Integer, default=300)  # Seconds
    enabled_features = Column(JSON, default=list)
    
    # Relationships
    user = relationship("User", back_populates="wearable_devices")


class Notification(Base):
    """System notifications and alerts"""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Notification details
    title = Column(String)
    message = Column(Text)
    notification_type = Column(String)  # "fatigue_alert", "energy_prediction", "schedule_optimization"
    severity = Column(String)  # "low", "medium", "high", "critical"
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    read_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    
    # Action data
    action_required = Column(Boolean, default=False)
    action_url = Column(String, nullable=True)
    metadata = Column(JSON, nullable=True)


# Pydantic models for API
class UserCreate(BaseModel):
    email: str
    username: str
    full_name: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class FatigueRecordCreate(BaseModel):
    overall_fatigue_score: float
    facial_fatigue_score: Optional[float] = None
    typing_fatigue_score: Optional[float] = None
    blink_rate: Optional[float] = None
    typing_speed: Optional[float] = None
    activity_type: str = "general"
    notes: Optional[str] = None


class FatigueRecordResponse(BaseModel):
    id: int
    user_id: int
    timestamp: datetime
    overall_fatigue_score: float
    facial_fatigue_score: Optional[float]
    typing_fatigue_score: Optional[float]
    blink_rate: Optional[float]
    typing_speed: Optional[float]
    activity_type: str
    
    class Config:
        from_attributes = True


class BioRhythmRecordCreate(BaseModel):
    heart_rate: Optional[float] = None
    sleep_duration: Optional[float] = None
    sleep_quality: Optional[float] = None
    steps_count: Optional[int] = None
    predicted_energy_level: Optional[float] = None


class ScheduleCreate(BaseModel):
    title: str
    description: Optional[str] = None
    start_time: datetime
    end_time: datetime
    task_type: str
    priority: int
    energy_requirement: float
    complexity: float


class ScheduleResponse(BaseModel):
    id: int
    user_id: int
    title: str
    description: Optional[str]
    start_time: datetime
    end_time: datetime
    task_type: str
    priority: int
    energy_requirement: float
    complexity: float
    recommended_start_time: Optional[datetime]
    energy_prediction: Optional[float]
    fatigue_risk: Optional[float]
    is_completed: bool
    is_optimized: bool
    
    class Config:
        from_attributes = True