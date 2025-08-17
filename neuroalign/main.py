"""
NeuroAlign - AI-Powered Fatigue Detection & Smart Scheduling System
Main application entry point
"""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from neuroalign.api.routes import fatigue, scheduling, auth, dashboard
from neuroalign.services.fatigue_detector import FatigueDetector
from neuroalign.services.bio_rhythm_analyzer import BioRhythmAnalyzer
from neuroalign.services.websocket_manager import WebSocketManager
from neuroalign.utils.database import init_db, get_db
from neuroalign.utils.config import Settings

# Load environment variables
load_dotenv()

# Global settings
settings = Settings()

# WebSocket manager for real-time communication
websocket_manager = WebSocketManager()

# Global service instances
fatigue_detector = None
bio_rhythm_analyzer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 Starting NeuroAlign Fatigue Detection System...")
    
    # Initialize database
    await init_db()
    
    # Initialize ML models
    global fatigue_detector, bio_rhythm_analyzer
    fatigue_detector = FatigueDetector()
    bio_rhythm_analyzer = BioRhythmAnalyzer()
    
    print("✅ NeuroAlign system initialized successfully!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down NeuroAlign system...")


# Create FastAPI app
app = FastAPI(
    title="NeuroAlign",
    description="AI-Powered Fatigue Detection & Smart Scheduling System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="neuroalign/static"), name="static")

# Templates
templates = Jinja2Templates(directory="neuroalign/templates")

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(fatigue.router, prefix="/api/fatigue", tags=["Fatigue Detection"])
app.include_router(scheduling.router, prefix="/api/scheduling", tags=["Smart Scheduling"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])


@app.get("/")
async def root(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NeuroAlign",
        "version": "1.0.0",
        "fatigue_detector": fatigue_detector is not None,
        "bio_rhythm_analyzer": bio_rhythm_analyzer is not None
    }


@app.websocket("/ws/fatigue")
async def websocket_fatigue_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time fatigue monitoring"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Process fatigue detection
            if data.get("type") == "webcam_frame":
                fatigue_score = await fatigue_detector.analyze_frame(data["frame"])
                await websocket.send_json({
                    "type": "fatigue_score",
                    "score": fatigue_score,
                    "timestamp": data.get("timestamp")
                })
            
            elif data.get("type") == "typing_data":
                typing_fatigue = await fatigue_detector.analyze_typing(data["typing_pattern"])
                await websocket.send_json({
                    "type": "typing_fatigue",
                    "score": typing_fatigue,
                    "timestamp": data.get("timestamp")
                })
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.websocket("/ws/bio-rhythm")
async def websocket_bio_rhythm_endpoint(websocket: WebSocket):
    """WebSocket endpoint for bio-rhythm data"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "bio_data":
                energy_prediction = await bio_rhythm_analyzer.predict_energy(data["bio_signals"])
                await websocket.send_json({
                    "type": "energy_prediction",
                    "prediction": energy_prediction,
                    "timestamp": data.get("timestamp")
                })
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(
        "neuroalign.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )