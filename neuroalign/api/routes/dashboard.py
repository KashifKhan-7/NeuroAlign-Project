"""
Dashboard API Routes
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

from neuroalign.utils.database import get_db
from neuroalign.models.database import User, FatigueRecord, BioRhythmRecord, Schedule, Notification
from neuroalign.services.fatigue_detector import FatigueDetector
from neuroalign.services.bio_rhythm_analyzer import BioRhythmAnalyzer
from neuroalign.services.websocket_manager import WebSocketManager

router = APIRouter()
websocket_manager = WebSocketManager()


@router.get("/overview/{user_id}", response_model=Dict)
async def get_dashboard_overview(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Get dashboard overview for a user
    
    Args:
        user_id: User identifier
        
    Returns:
        Dictionary with dashboard overview data
    """
    try:
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Calculate date range (last 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Get fatigue statistics
        fatigue_stats = db.query(
            func.avg(FatigueRecord.overall_fatigue_score).label("avg_fatigue"),
            func.max(FatigueRecord.overall_fatigue_score).label("max_fatigue"),
            func.count(FatigueRecord.id).label("total_records")
        ).filter(
            FatigueRecord.user_id == user_id,
            FatigueRecord.timestamp >= start_date
        ).first()
        
        # Get bio-rhythm statistics
        bio_stats = db.query(
            func.avg(BioRhythmRecord.predicted_energy_level).label("avg_energy"),
            func.avg(BioRhythmRecord.heart_rate).label("avg_heart_rate"),
            func.avg(BioRhythmRecord.sleep_duration).label("avg_sleep")
        ).filter(
            BioRhythmRecord.user_id == user_id,
            BioRhythmRecord.timestamp >= start_date
        ).first()
        
        # Get schedule statistics
        schedule_stats = db.query(
            func.count(Schedule.id).label("total_schedules"),
            func.count(Schedule.id).filter(Schedule.is_completed == True).label("completed_schedules"),
            func.count(Schedule.id).filter(Schedule.is_optimized == True).label("optimized_schedules")
        ).filter(
            Schedule.user_id == user_id,
            Schedule.start_time >= start_date
        ).first()
        
        # Get recent notifications
        recent_notifications = db.query(Notification).filter(
            Notification.user_id == user_id
        ).order_by(Notification.created_at.desc()).limit(5).all()
        
        # Calculate fatigue trend
        fatigue_trend = "stable"
        if fatigue_stats.avg_fatigue:
            # Get fatigue data for trend calculation
            fatigue_data = db.query(FatigueRecord.overall_fatigue_score).filter(
                FatigueRecord.user_id == user_id,
                FatigueRecord.timestamp >= start_date
            ).order_by(FatigueRecord.timestamp).all()
            
            if len(fatigue_data) >= 2:
                recent_avg = sum([f[0] for f in fatigue_data[-len(fatigue_data)//2:]]) / (len(fatigue_data)//2)
                earlier_avg = sum([f[0] for f in fatigue_data[:len(fatigue_data)//2]]) / (len(fatigue_data)//2)
                
                if recent_avg > earlier_avg + 0.1:
                    fatigue_trend = "increasing"
                elif recent_avg < earlier_avg - 0.1:
                    fatigue_trend = "decreasing"
        
        # Get current fatigue level
        current_fatigue = 0.5  # Default
        if fatigue_stats.avg_fatigue:
            current_fatigue = fatigue_stats.avg_fatigue
        
        # Classify fatigue level
        if current_fatigue < 0.3:
            fatigue_level = "low"
        elif current_fatigue < 0.6:
            fatigue_level = "moderate"
        elif current_fatigue < 0.8:
            fatigue_level = "high"
        else:
            fatigue_level = "critical"
        
        return {
            "user": {
                "id": user.id,
                "username": user.username,
                "full_name": user.full_name
            },
            "fatigue_overview": {
                "current_level": fatigue_level,
                "current_score": round(current_fatigue, 3) if current_fatigue else 0.0,
                "average_score": round(fatigue_stats.avg_fatigue, 3) if fatigue_stats.avg_fatigue else 0.0,
                "max_score": round(fatigue_stats.max_fatigue, 3) if fatigue_stats.max_fatigue else 0.0,
                "trend": fatigue_trend,
                "total_records": fatigue_stats.total_records or 0
            },
            "bio_rhythm_overview": {
                "average_energy": round(bio_stats.avg_energy, 3) if bio_stats.avg_energy else 0.0,
                "average_heart_rate": round(bio_stats.avg_heart_rate, 1) if bio_stats.avg_heart_rate else 0.0,
                "average_sleep": round(bio_stats.avg_sleep, 1) if bio_stats.avg_sleep else 0.0
            },
            "schedule_overview": {
                "total_schedules": schedule_stats.total_schedules or 0,
                "completed_schedules": schedule_stats.completed_schedules or 0,
                "optimized_schedules": schedule_stats.optimized_schedules or 0,
                "completion_rate": round(
                    (schedule_stats.completed_schedules or 0) / max(schedule_stats.total_schedules or 1, 1) * 100, 1
                )
            },
            "recent_notifications": [
                {
                    "id": notif.id,
                    "title": notif.title,
                    "message": notif.message,
                    "type": notif.notification_type,
                    "severity": notif.severity,
                    "created_at": notif.created_at.isoformat(),
                    "read": notif.read_at is not None
                }
                for notif in recent_notifications
            ],
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dashboard overview: {str(e)}")


@router.get("/fatigue-chart/{user_id}", response_model=Dict)
async def get_fatigue_chart_data(
    user_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get fatigue chart data for visualization
    
    Args:
        user_id: User identifier
        days: Number of days to retrieve (default: 7)
        
    Returns:
        Dictionary with chart data
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get fatigue records grouped by day
        fatigue_data = db.query(
            func.date(FatigueRecord.timestamp).label("date"),
            func.avg(FatigueRecord.overall_fatigue_score).label("avg_fatigue"),
            func.avg(FatigueRecord.facial_fatigue_score).label("avg_facial_fatigue"),
            func.avg(FatigueRecord.typing_fatigue_score).label("avg_typing_fatigue"),
            func.count(FatigueRecord.id).label("record_count")
        ).filter(
            FatigueRecord.user_id == user_id,
            FatigueRecord.timestamp >= start_date
        ).group_by(func.date(FatigueRecord.timestamp)).order_by(func.date(FatigueRecord.timestamp)).all()
        
        # Format data for chart
        chart_data = {
            "labels": [],
            "datasets": [
                {
                    "label": "Overall Fatigue",
                    "data": [],
                    "borderColor": "rgb(255, 99, 132)",
                    "backgroundColor": "rgba(255, 99, 132, 0.1)"
                },
                {
                    "label": "Facial Fatigue",
                    "data": [],
                    "borderColor": "rgb(54, 162, 235)",
                    "backgroundColor": "rgba(54, 162, 235, 0.1)"
                },
                {
                    "label": "Typing Fatigue",
                    "data": [],
                    "borderColor": "rgb(255, 205, 86)",
                    "backgroundColor": "rgba(255, 205, 86, 0.1)"
                }
            ]
        }
        
        for record in fatigue_data:
            chart_data["labels"].append(record.date.strftime("%Y-%m-%d"))
            chart_data["datasets"][0]["data"].append(round(record.avg_fatigue, 3) if record.avg_fatigue else 0)
            chart_data["datasets"][1]["data"].append(round(record.avg_facial_fatigue, 3) if record.avg_facial_fatigue else 0)
            chart_data["datasets"][2]["data"].append(round(record.avg_typing_fatigue, 3) if record.avg_typing_fatigue else 0)
        
        return chart_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting fatigue chart data: {str(e)}")


@router.get("/energy-chart/{user_id}", response_model=Dict)
async def get_energy_chart_data(
    user_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get energy chart data for visualization
    
    Args:
        user_id: User identifier
        days: Number of days to retrieve (default: 7)
        
    Returns:
        Dictionary with chart data
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get bio-rhythm records grouped by day
        energy_data = db.query(
            func.date(BioRhythmRecord.timestamp).label("date"),
            func.avg(BioRhythmRecord.predicted_energy_level).label("avg_energy"),
            func.avg(BioRhythmRecord.heart_rate).label("avg_heart_rate"),
            func.avg(BioRhythmRecord.sleep_duration).label("avg_sleep"),
            func.count(BioRhythmRecord.id).label("record_count")
        ).filter(
            BioRhythmRecord.user_id == user_id,
            BioRhythmRecord.timestamp >= start_date
        ).group_by(func.date(BioRhythmRecord.timestamp)).order_by(func.date(BioRhythmRecord.timestamp)).all()
        
        # Format data for chart
        chart_data = {
            "labels": [],
            "datasets": [
                {
                    "label": "Energy Level",
                    "data": [],
                    "borderColor": "rgb(75, 192, 192)",
                    "backgroundColor": "rgba(75, 192, 192, 0.1)"
                },
                {
                    "label": "Heart Rate",
                    "data": [],
                    "borderColor": "rgb(255, 159, 64)",
                    "backgroundColor": "rgba(255, 159, 64, 0.1)",
                    "yAxisID": "y1"
                },
                {
                    "label": "Sleep Duration",
                    "data": [],
                    "borderColor": "rgb(153, 102, 255)",
                    "backgroundColor": "rgba(153, 102, 255, 0.1)",
                    "yAxisID": "y2"
                }
            ]
        }
        
        for record in energy_data:
            chart_data["labels"].append(record.date.strftime("%Y-%m-%d"))
            chart_data["datasets"][0]["data"].append(round(record.avg_energy, 3) if record.avg_energy else 0)
            chart_data["datasets"][1]["data"].append(round(record.avg_heart_rate, 1) if record.avg_heart_rate else 0)
            chart_data["datasets"][2]["data"].append(round(record.avg_sleep, 1) if record.avg_sleep else 0)
        
        return chart_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting energy chart data: {str(e)}")


@router.get("/schedule-chart/{user_id}", response_model=Dict)
async def get_schedule_chart_data(
    user_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get schedule chart data for visualization
    
    Args:
        user_id: User identifier
        days: Number of days to retrieve (default: 7)
        
    Returns:
        Dictionary with chart data
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get schedule data grouped by day
        schedule_data = db.query(
            func.date(Schedule.start_time).label("date"),
            func.count(Schedule.id).label("total_schedules"),
            func.count(Schedule.id).filter(Schedule.is_completed == True).label("completed_schedules"),
            func.count(Schedule.id).filter(Schedule.is_optimized == True).label("optimized_schedules"),
            func.avg(Schedule.energy_prediction).label("avg_energy_prediction"),
            func.avg(Schedule.fatigue_risk).label("avg_fatigue_risk")
        ).filter(
            Schedule.user_id == user_id,
            Schedule.start_time >= start_date
        ).group_by(func.date(Schedule.start_time)).order_by(func.date(Schedule.start_time)).all()
        
        # Format data for chart
        chart_data = {
            "labels": [],
            "datasets": [
                {
                    "label": "Total Schedules",
                    "data": [],
                    "borderColor": "rgb(255, 99, 132)",
                    "backgroundColor": "rgba(255, 99, 132, 0.1)"
                },
                {
                    "label": "Completed Schedules",
                    "data": [],
                    "borderColor": "rgb(54, 162, 235)",
                    "backgroundColor": "rgba(54, 162, 235, 0.1)"
                },
                {
                    "label": "Optimized Schedules",
                    "data": [],
                    "borderColor": "rgb(255, 205, 86)",
                    "backgroundColor": "rgba(255, 205, 86, 0.1)"
                }
            ]
        }
        
        for record in schedule_data:
            chart_data["labels"].append(record.date.strftime("%Y-%m-%d"))
            chart_data["datasets"][0]["data"].append(record.total_schedules or 0)
            chart_data["datasets"][1]["data"].append(record.completed_schedules or 0)
            chart_data["datasets"][2]["data"].append(record.optimized_schedules or 0)
        
        return chart_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting schedule chart data: {str(e)}")


@router.get("/recommendations/{user_id}", response_model=List[Dict])
async def get_user_recommendations(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Get personalized recommendations for a user
    
    Args:
        user_id: User identifier
        
    Returns:
        List of recommendations
    """
    try:
        recommendations = []
        
        # Get recent fatigue data
        recent_fatigue = db.query(FatigueRecord).filter(
            FatigueRecord.user_id == user_id
        ).order_by(FatigueRecord.timestamp.desc()).limit(10).all()
        
        # Get recent bio-rhythm data
        recent_bio = db.query(BioRhythmRecord).filter(
            BioRhythmRecord.user_id == user_id
        ).order_by(BioRhythmRecord.timestamp.desc()).limit(10).all()
        
        # Fatigue-based recommendations
        if recent_fatigue:
            avg_fatigue = sum(r.overall_fatigue_score for r in recent_fatigue if r.overall_fatigue_score) / len(recent_fatigue)
            
            if avg_fatigue > 0.7:
                recommendations.append({
                    "type": "fatigue",
                    "priority": "high",
                    "title": "High Fatigue Detected",
                    "message": "Your fatigue levels are consistently high. Consider implementing more breaks.",
                    "actions": [
                        "Schedule regular 15-minute breaks",
                        "Reduce screen time",
                        "Consider taking a day off"
                    ]
                })
            elif avg_fatigue > 0.5:
                recommendations.append({
                    "type": "fatigue",
                    "priority": "medium",
                    "title": "Moderate Fatigue",
                    "message": "Your fatigue levels are elevated. Focus on better work-life balance.",
                    "actions": [
                        "Take 5-minute breaks every hour",
                        "Practice the 20-20-20 rule for eyes",
                        "Ensure adequate sleep"
                    ]
                })
        
        # Bio-rhythm based recommendations
        if recent_bio:
            avg_sleep = sum(r.sleep_duration for r in recent_bio if r.sleep_duration) / len(recent_bio)
            avg_energy = sum(r.predicted_energy_level for r in recent_bio if r.predicted_energy_level) / len(recent_bio)
            
            if avg_sleep and avg_sleep < 7.0:
                recommendations.append({
                    "type": "sleep",
                    "priority": "high",
                    "title": "Insufficient Sleep",
                    "message": "You're not getting enough sleep, which affects your energy and productivity.",
                    "actions": [
                        "Aim for 7-9 hours of sleep per night",
                        "Establish a consistent sleep schedule",
                        "Create a relaxing bedtime routine"
                    ]
                })
            
            if avg_energy and avg_energy < 0.4:
                recommendations.append({
                    "type": "energy",
                    "priority": "medium",
                    "title": "Low Energy Levels",
                    "message": "Your energy levels are consistently low. Consider lifestyle adjustments.",
                    "actions": [
                        "Increase physical activity",
                        "Improve nutrition",
                        "Consider stress management techniques"
                    ]
                })
        
        # Schedule optimization recommendations
        future_schedules = db.query(Schedule).filter(
            Schedule.user_id == user_id,
            Schedule.start_time >= datetime.now(),
            Schedule.is_optimized == False
        ).count()
        
        if future_schedules > 0:
            recommendations.append({
                "type": "schedule",
                "priority": "medium",
                "title": "Schedule Optimization Available",
                "message": f"You have {future_schedules} unoptimized schedules. Consider optimizing them for better energy alignment.",
                "actions": [
                    "Review and optimize your schedule",
                    "Align tasks with your energy patterns",
                    "Consider rescheduling high-energy tasks"
                ]
            })
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")


@router.get("/system-status", response_model=Dict)
async def get_system_status():
    """
    Get system status and health information
    
    Returns:
        Dictionary with system status
    """
    try:
        # Get WebSocket connection stats
        ws_stats = websocket_manager.get_connection_stats()
        
        # Check if services are available
        services_status = {
            "fatigue_detector": True,  # Would check actual service status
            "bio_rhythm_analyzer": True,
            "websocket_manager": True,
            "database": True  # Would check database connection
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": services_status,
            "websocket_connections": ws_stats,
            "version": "1.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "version": "1.0.0"
        }