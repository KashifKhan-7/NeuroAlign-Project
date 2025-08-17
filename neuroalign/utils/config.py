"""
Configuration settings for NeuroAlign
"""

import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    DATABASE_URL: str = "sqlite:///./neuroalign.db"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ML Models
    MODEL_PATH: str = "neuroalign/ml_models/"
    FACE_DETECTION_MODEL: str = "haarcascade_frontalface_default.xml"
    MICRO_EXPRESSION_MODEL: str = "micro_expression_cnn.h5"
    BIO_RHYTHM_MODEL: str = "bio_rhythm_lstm.h5"
    
    # Wearable Device APIs
    FITBIT_CLIENT_ID: Optional[str] = None
    FITBIT_CLIENT_SECRET: Optional[str] = None
    GARMIN_CONSUMER_KEY: Optional[str] = None
    GARMIN_CONSUMER_SECRET: Optional[str] = None
    
    # Real-time Processing
    WEBSOCKET_PING_INTERVAL: int = 20
    WEBSOCKET_PING_TIMEOUT: int = 20
    
    # Fatigue Detection Settings
    BLINK_RATE_THRESHOLD: float = 0.3  # Blinks per second
    TYPING_HESITATION_THRESHOLD: float = 0.5  # Seconds
    FATIGUE_SCORE_THRESHOLD: float = 0.7  # 0-1 scale
    
    # Bio-rhythm Analysis
    ENERGY_PREDICTION_HORIZON: int = 24  # Hours
    SLEEP_CYCLE_LENGTH: int = 90  # Minutes
    OPTIMAL_ENERGY_WINDOW: tuple = (9, 11)  # Hours (9 AM - 11 AM)
    
    # Notification Settings
    ENABLE_NOTIFICATIONS: bool = True
    NOTIFICATION_COOLDOWN: int = 300  # Seconds
    
    # Development
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()