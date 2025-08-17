"""
Fatigue Detection API Routes
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

from neuroalign.utils.database import get_db
from neuroalign.models.database import FatigueRecord, FatigueRecordCreate, FatigueRecordResponse
from neuroalign.services.fatigue_detector import FatigueDetector
from neuroalign.services.websocket_manager import WebSocketManager

router = APIRouter()
websocket_manager = WebSocketManager()


@router.post("/analyze-frame", response_model=Dict)
async def analyze_webcam_frame(
    frame_data: Dict,
    db: Session = Depends(get_db)
):
    """
    Analyze webcam frame for fatigue indicators
    
    Args:
        frame_data: Dictionary containing base64 encoded image data
        
    Returns:
        Dictionary with fatigue analysis results
    """
    try:
        # Initialize fatigue detector
        fatigue_detector = FatigueDetector()
        
        # Analyze frame
        frame_base64 = frame_data.get("frame")
        if not frame_base64:
            raise HTTPException(status_code=400, detail="Frame data is required")
        
        result = await fatigue_detector.analyze_frame(frame_base64)
        
        # Store result in database if user_id provided
        user_id = frame_data.get("user_id")
        if user_id:
            fatigue_record = FatigueRecord(
                user_id=user_id,
                facial_fatigue_score=result["facial_fatigue_score"],
                blink_rate=result["blink_rate"],
                eye_closure_duration=result["eye_closure_duration"],
                facial_tension_score=result["facial_tension_score"],
                smile_frequency=result["smile_frequency"],
                activity_type="webcam_analysis"
            )
            db.add(fatigue_record)
            db.commit()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing frame: {str(e)}")


@router.post("/analyze-typing", response_model=Dict)
async def analyze_typing_patterns(
    typing_data: Dict,
    db: Session = Depends(get_db)
):
    """
    Analyze typing patterns for fatigue indicators
    
    Args:
        typing_data: Dictionary containing typing events and timestamps
        
    Returns:
        Dictionary with typing fatigue analysis results
    """
    try:
        # Initialize fatigue detector
        fatigue_detector = FatigueDetector()
        
        # Analyze typing patterns
        result = await fatigue_detector.analyze_typing(typing_data)
        
        # Store result in database if user_id provided
        user_id = typing_data.get("user_id")
        if user_id:
            fatigue_record = FatigueRecord(
                user_id=user_id,
                typing_fatigue_score=result["typing_fatigue_score"],
                typing_speed=result["typing_speed"],
                hesitation_count=int(result["hesitation_score"] * 100),  # Convert to count
                backspace_count=int(result["backspace_frequency"] * 100),  # Convert to count
                typing_rhythm_variance=result["rhythm_variance"],
                activity_type="typing_analysis"
            )
            db.add(fatigue_record)
            db.commit()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing typing: {str(e)}")


@router.get("/overall-score/{user_id}", response_model=Dict)
async def get_overall_fatigue_score(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Get overall fatigue score combining all data sources
    
    Args:
        user_id: User identifier
        
    Returns:
        Dictionary with overall fatigue assessment
    """
    try:
        # Initialize fatigue detector
        fatigue_detector = FatigueDetector()
        
        # Get overall fatigue score
        result = await fatigue_detector.get_overall_fatigue_score(user_id)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating fatigue score: {str(e)}")


@router.get("/history/{user_id}", response_model=List[FatigueRecordResponse])
async def get_fatigue_history(
    user_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get fatigue history for a user
    
    Args:
        user_id: User identifier
        days: Number of days to retrieve (default: 7)
        
    Returns:
        List of fatigue records
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query database
        records = db.query(FatigueRecord).filter(
            FatigueRecord.user_id == user_id,
            FatigueRecord.timestamp >= start_date,
            FatigueRecord.timestamp <= end_date
        ).order_by(FatigueRecord.timestamp.desc()).all()
        
        return records
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving fatigue history: {str(e)}")


@router.post("/record", response_model=FatigueRecordResponse)
async def create_fatigue_record(
    record: FatigueRecordCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new fatigue record
    
    Args:
        record: Fatigue record data
        
    Returns:
        Created fatigue record
    """
    try:
        # Create new record
        db_record = FatigueRecord(
            user_id=record.user_id if hasattr(record, 'user_id') else 1,  # Default user
            overall_fatigue_score=record.overall_fatigue_score,
            facial_fatigue_score=record.facial_fatigue_score,
            typing_fatigue_score=record.typing_fatigue_score,
            blink_rate=record.blink_rate,
            typing_speed=record.typing_speed,
            activity_type=record.activity_type,
            notes=record.notes
        )
        
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        
        return db_record
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating fatigue record: {str(e)}")


@router.get("/stats/{user_id}", response_model=Dict)
async def get_fatigue_stats(
    user_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get fatigue statistics for a user
    
    Args:
        user_id: User identifier
        days: Number of days to analyze (default: 7)
        
    Returns:
        Dictionary with fatigue statistics
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query database
        records = db.query(FatigueRecord).filter(
            FatigueRecord.user_id == user_id,
            FatigueRecord.timestamp >= start_date,
            FatigueRecord.timestamp <= end_date
        ).all()
        
        if not records:
            return {
                "total_records": 0,
                "avg_fatigue_score": 0.0,
                "max_fatigue_score": 0.0,
                "min_fatigue_score": 0.0,
                "fatigue_trend": "stable",
                "high_fatigue_days": 0
            }
        
        # Calculate statistics
        fatigue_scores = [r.overall_fatigue_score for r in records if r.overall_fatigue_score is not None]
        
        if not fatigue_scores:
            return {
                "total_records": len(records),
                "avg_fatigue_score": 0.0,
                "max_fatigue_score": 0.0,
                "min_fatigue_score": 0.0,
                "fatigue_trend": "stable",
                "high_fatigue_days": 0
            }
        
        avg_fatigue = sum(fatigue_scores) / len(fatigue_scores)
        max_fatigue = max(fatigue_scores)
        min_fatigue = min(fatigue_scores)
        
        # Calculate trend (simplified)
        if len(fatigue_scores) >= 2:
            recent_avg = sum(fatigue_scores[-len(fatigue_scores)//2:]) / (len(fatigue_scores)//2)
            earlier_avg = sum(fatigue_scores[:len(fatigue_scores)//2]) / (len(fatigue_scores)//2)
            
            if recent_avg > earlier_avg + 0.1:
                trend = "increasing"
            elif recent_avg < earlier_avg - 0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Count high fatigue days
        high_fatigue_days = len([score for score in fatigue_scores if score > 0.7])
        
        return {
            "total_records": len(records),
            "avg_fatigue_score": round(avg_fatigue, 3),
            "max_fatigue_score": round(max_fatigue, 3),
            "min_fatigue_score": round(min_fatigue, 3),
            "fatigue_trend": trend,
            "high_fatigue_days": high_fatigue_days,
            "analysis_period_days": days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating fatigue stats: {str(e)}")


@router.websocket("/ws")
async def fatigue_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time fatigue monitoring
    """
    await websocket_manager.connect(websocket, "fatigue")
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Process different message types
            message_type = data.get("type")
            
            if message_type == "webcam_frame":
                # Analyze webcam frame
                fatigue_detector = FatigueDetector()
                result = await fatigue_detector.analyze_frame(data["frame"])
                
                # Send result back
                await websocket_manager.send_personal_message({
                    "type": "fatigue_analysis",
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
            elif message_type == "typing_data":
                # Analyze typing patterns
                fatigue_detector = FatigueDetector()
                result = await fatigue_detector.analyze_typing(data["typing_pattern"])
                
                # Send result back
                await websocket_manager.send_personal_message({
                    "type": "typing_analysis",
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
            elif message_type == "get_overall_score":
                # Get overall fatigue score
                user_id = data.get("user_id", 1)
                fatigue_detector = FatigueDetector()
                result = await fatigue_detector.get_overall_fatigue_score(user_id)
                
                # Send result back
                await websocket_manager.send_personal_message({
                    "type": "overall_fatigue_score",
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


@router.get("/alerts/{user_id}", response_model=List[Dict])
async def get_fatigue_alerts(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Get fatigue alerts and recommendations for a user
    
    Args:
        user_id: User identifier
        
    Returns:
        List of fatigue alerts and recommendations
    """
    try:
        # Get recent fatigue data
        recent_records = db.query(FatigueRecord).filter(
            FatigueRecord.user_id == user_id
        ).order_by(FatigueRecord.timestamp.desc()).limit(10).all()
        
        alerts = []
        
        if recent_records:
            # Calculate average fatigue score
            avg_fatigue = sum(r.overall_fatigue_score for r in recent_records if r.overall_fatigue_score) / len(recent_records)
            
            # Generate alerts based on fatigue level
            if avg_fatigue > 0.8:
                alerts.append({
                    "type": "critical",
                    "title": "High Fatigue Detected",
                    "message": "Your fatigue levels are critically high. Consider taking immediate rest.",
                    "recommendations": [
                        "Stop current activity immediately",
                        "Take a 30-minute break",
                        "Get some rest or sleep"
                    ]
                })
            elif avg_fatigue > 0.6:
                alerts.append({
                    "type": "warning",
                    "title": "Moderate Fatigue Detected",
                    "message": "Your fatigue levels are elevated. Consider taking a break.",
                    "recommendations": [
                        "Take a 15-minute break",
                        "Step away from the screen",
                        "Do some light stretching"
                    ]
                })
            elif avg_fatigue > 0.4:
                alerts.append({
                    "type": "info",
                    "title": "Mild Fatigue Detected",
                    "message": "Your fatigue levels are slightly elevated.",
                    "recommendations": [
                        "Take a 5-minute break",
                        "Hydrate and have a healthy snack",
                        "Maintain good posture"
                    ]
                })
        
        return alerts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving fatigue alerts: {str(e)}")