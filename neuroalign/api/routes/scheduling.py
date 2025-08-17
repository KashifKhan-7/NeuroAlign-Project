"""
Smart Scheduling API Routes
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

from neuroalign.utils.database import get_db
from neuroalign.models.database import Schedule, BioRhythmRecord, ScheduleCreate, ScheduleResponse
from neuroalign.services.bio_rhythm_analyzer import BioRhythmAnalyzer
from neuroalign.services.websocket_manager import WebSocketManager

router = APIRouter()
websocket_manager = WebSocketManager()


@router.post("/predict-energy", response_model=Dict)
async def predict_energy_levels(
    bio_signals: Dict,
    db: Session = Depends(get_db)
):
    """
    Predict energy levels based on bio-signals
    
    Args:
        bio_signals: Dictionary containing heart rate, sleep, activity data
        
    Returns:
        Dictionary with energy predictions and recommendations
    """
    try:
        # Initialize bio-rhythm analyzer
        bio_analyzer = BioRhythmAnalyzer()
        
        # Predict energy levels
        result = await bio_analyzer.predict_energy(bio_signals)
        
        # Store bio-rhythm record if user_id provided
        user_id = bio_signals.get("user_id")
        if user_id:
            bio_record = BioRhythmRecord(
                user_id=user_id,
                heart_rate=bio_signals.get("heart_rate"),
                sleep_duration=bio_signals.get("sleep_duration"),
                sleep_quality=bio_signals.get("sleep_quality"),
                steps_count=bio_signals.get("steps_count"),
                predicted_energy_level=result["current_energy_level"]
            )
            db.add(bio_record)
            db.commit()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting energy: {str(e)}")


@router.post("/optimize-schedule", response_model=Dict)
async def optimize_schedule(
    schedule_data: Dict,
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Optimize schedule based on predicted energy levels
    
    Args:
        schedule_data: Dictionary with schedule information
        user_id: User identifier
        
    Returns:
        Dictionary with optimized schedule recommendations
    """
    try:
        # Initialize bio-rhythm analyzer
        bio_analyzer = BioRhythmAnalyzer()
        
        # Optimize schedule
        result = await bio_analyzer.optimize_schedule(schedule_data, user_id)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing schedule: {str(e)}")


@router.post("/schedule", response_model=ScheduleResponse)
async def create_schedule(
    schedule: ScheduleCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new schedule item
    
    Args:
        schedule: Schedule data
        
    Returns:
        Created schedule item
    """
    try:
        # Create new schedule
        db_schedule = Schedule(
            user_id=schedule.user_id if hasattr(schedule, 'user_id') else 1,  # Default user
            title=schedule.title,
            description=schedule.description,
            start_time=schedule.start_time,
            end_time=schedule.end_time,
            date=schedule.start_time.date(),
            task_type=schedule.task_type,
            priority=schedule.priority,
            energy_requirement=schedule.energy_requirement,
            complexity=schedule.complexity
        )
        
        db.add(db_schedule)
        db.commit()
        db.refresh(db_schedule)
        
        return db_schedule
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating schedule: {str(e)}")


@router.get("/schedule/{user_id}", response_model=List[ScheduleResponse])
async def get_user_schedule(
    user_id: int,
    date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get schedule for a user
    
    Args:
        user_id: User identifier
        date: Optional date filter (YYYY-MM-DD format)
        
    Returns:
        List of schedule items
    """
    try:
        query = db.query(Schedule).filter(Schedule.user_id == user_id)
        
        if date:
            try:
                filter_date = datetime.strptime(date, "%Y-%m-%d").date()
                query = query.filter(Schedule.date == filter_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        schedules = query.order_by(Schedule.start_time).all()
        
        return schedules
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving schedule: {str(e)}")


@router.put("/schedule/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(
    schedule_id: int,
    schedule_update: Dict,
    db: Session = Depends(get_db)
):
    """
    Update a schedule item
    
    Args:
        schedule_id: Schedule identifier
        schedule_update: Updated schedule data
        
    Returns:
        Updated schedule item
    """
    try:
        # Get existing schedule
        schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
        
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        # Update fields
        for field, value in schedule_update.items():
            if hasattr(schedule, field):
                setattr(schedule, field, value)
        
        db.commit()
        db.refresh(schedule)
        
        return schedule
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating schedule: {str(e)}")


@router.delete("/schedule/{schedule_id}")
async def delete_schedule(
    schedule_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a schedule item
    
    Args:
        schedule_id: Schedule identifier
        
    Returns:
        Success message
    """
    try:
        # Get existing schedule
        schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
        
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        db.delete(schedule)
        db.commit()
        
        return {"message": "Schedule deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting schedule: {str(e)}")


@router.get("/optimal-windows/{user_id}", response_model=Dict)
async def get_optimal_windows(
    user_id: int,
    date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get optimal scheduling windows for a user
    
    Args:
        user_id: User identifier
        date: Optional date filter (YYYY-MM-DD format)
        
    Returns:
        Dictionary with optimal scheduling windows
    """
    try:
        # Initialize bio-rhythm analyzer
        bio_analyzer = BioRhythmAnalyzer()
        
        # Get energy prediction for the day
        energy_prediction = await bio_analyzer._get_daily_energy_prediction(user_id)
        
        # Calculate optimal windows
        optimal_windows = bio_analyzer._calculate_optimal_windows(energy_prediction)
        
        # Get existing schedule for the day
        filter_date = None
        if date:
            try:
                filter_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        else:
            filter_date = datetime.now().date()
        
        existing_schedule = db.query(Schedule).filter(
            Schedule.user_id == user_id,
            Schedule.date == filter_date
        ).all()
        
        # Calculate available time slots
        available_slots = []
        for window in optimal_windows:
            start_hour = window["start_hour"]
            end_hour = window["end_hour"]
            
            # Check for conflicts with existing schedule
            conflicts = []
            for schedule in existing_schedule:
                schedule_start = schedule.start_time.hour
                schedule_end = schedule.end_time.hour
                
                # Check for overlap
                if not (end_hour <= schedule_start or start_hour >= schedule_end):
                    conflicts.append({
                        "title": schedule.title,
                        "start": schedule_start,
                        "end": schedule_end
                    })
            
            available_slots.append({
                "start_hour": start_hour,
                "end_hour": end_hour,
                "duration": window["duration"],
                "avg_energy": window["avg_energy"],
                "conflicts": conflicts,
                "available": len(conflicts) == 0
            })
        
        return {
            "date": filter_date.isoformat(),
            "energy_prediction": energy_prediction,
            "optimal_windows": available_slots,
            "recommendations": bio_analyzer._generate_energy_recommendations(
                0.5, energy_prediction, optimal_windows
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating optimal windows: {str(e)}")


@router.get("/bio-rhythm-history/{user_id}", response_model=List[Dict])
async def get_bio_rhythm_history(
    user_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get bio-rhythm history for a user
    
    Args:
        user_id: User identifier
        days: Number of days to retrieve (default: 7)
        
    Returns:
        List of bio-rhythm records
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query database
        records = db.query(BioRhythmRecord).filter(
            BioRhythmRecord.user_id == user_id,
            BioRhythmRecord.timestamp >= start_date,
            BioRhythmRecord.timestamp <= end_date
        ).order_by(BioRhythmRecord.timestamp.desc()).all()
        
        # Convert to dictionary format
        history = []
        for record in records:
            history.append({
                "id": record.id,
                "timestamp": record.timestamp.isoformat(),
                "heart_rate": record.heart_rate,
                "sleep_duration": record.sleep_duration,
                "sleep_quality": record.sleep_quality,
                "steps_count": record.steps_count,
                "predicted_energy_level": record.predicted_energy_level,
                "energy_confidence": record.energy_confidence
            })
        
        return history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving bio-rhythm history: {str(e)}")


@router.post("/batch-optimize", response_model=Dict)
async def batch_optimize_schedules(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Batch optimize all schedules for a user
    
    Args:
        user_id: User identifier
        
    Returns:
        Dictionary with optimization results
    """
    try:
        # Get all future schedules for the user
        future_schedules = db.query(Schedule).filter(
            Schedule.user_id == user_id,
            Schedule.start_time >= datetime.now(),
            Schedule.is_optimized == False
        ).all()
        
        if not future_schedules:
            return {
                "message": "No future schedules to optimize",
                "optimized_count": 0,
                "schedules": []
            }
        
        # Initialize bio-rhythm analyzer
        bio_analyzer = BioRhythmAnalyzer()
        
        # Convert schedules to dictionary format
        schedule_data = {
            "items": [
                {
                    "id": schedule.id,
                    "title": schedule.title,
                    "start_time": schedule.start_time.isoformat(),
                    "end_time": schedule.end_time.isoformat(),
                    "task_type": schedule.task_type,
                    "priority": schedule.priority,
                    "energy_requirement": schedule.energy_requirement,
                    "complexity": schedule.complexity
                }
                for schedule in future_schedules
            ]
        }
        
        # Optimize schedules
        optimization_result = await bio_analyzer.optimize_schedule(schedule_data, user_id)
        
        # Update database with optimized times
        optimized_count = 0
        for optimized_item in optimization_result["optimized_schedule"]:
            schedule_id = optimized_item.get("id")
            if schedule_id:
                schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
                if schedule:
                    schedule.recommended_start_time = datetime.fromisoformat(optimized_item["recommended_start"])
                    schedule.recommended_end_time = datetime.fromisoformat(optimized_item["recommended_end"])
                    schedule.energy_prediction = optimized_item["energy_prediction"]
                    schedule.fatigue_risk = optimized_item["fatigue_risk"]
                    schedule.is_optimized = True
                    optimized_count += 1
        
        db.commit()
        
        return {
            "message": f"Successfully optimized {optimized_count} schedules",
            "optimized_count": optimized_count,
            "optimization_summary": optimization_result["optimization_summary"],
            "schedules": optimization_result["optimized_schedule"]
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error batch optimizing schedules: {str(e)}")


@router.websocket("/ws")
async def scheduling_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time scheduling updates
    """
    await websocket_manager.connect(websocket, "bio_rhythm")
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Process different message types
            message_type = data.get("type")
            
            if message_type == "bio_data":
                # Predict energy levels
                bio_analyzer = BioRhythmAnalyzer()
                result = await bio_analyzer.predict_energy(data["bio_signals"])
                
                # Send result back
                await websocket_manager.send_personal_message({
                    "type": "energy_prediction",
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
            elif message_type == "optimize_schedule":
                # Optimize schedule
                bio_analyzer = BioRhythmAnalyzer()
                result = await bio_analyzer.optimize_schedule(data["schedule_data"], data.get("user_id", 1))
                
                # Send result back
                await websocket_manager.send_personal_message({
                    "type": "schedule_optimization",
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)