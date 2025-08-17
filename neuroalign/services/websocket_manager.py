"""
WebSocket Manager for real-time communication
"""

import asyncio
import json
from typing import Dict, List, Set
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time communication"""
    
    def __init__(self):
        """Initialize WebSocket manager"""
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "fatigue": set(),
            "bio_rhythm": set(),
            "dashboard": set()
        }
        self.connection_data: Dict[WebSocket, Dict] = {}
        
        print("🔌 WebSocket Manager initialized")
    
    async def connect(self, websocket: WebSocket, connection_type: str = "dashboard"):
        """Connect a new WebSocket client"""
        try:
            await websocket.accept()
            
            if connection_type in self.active_connections:
                self.active_connections[connection_type].add(websocket)
            
            # Store connection metadata
            self.connection_data[websocket] = {
                "type": connection_type,
                "connected_at": datetime.now(),
                "user_id": None,
                "last_activity": datetime.now()
            }
            
            logger.info(f"WebSocket connected: {connection_type}")
            
            # Send welcome message
            await self.send_personal_message({
                "type": "connection_established",
                "connection_type": connection_type,
                "timestamp": datetime.now().isoformat()
            }, websocket)
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        try:
            # Remove from all connection sets
            for connection_set in self.active_connections.values():
                connection_set.discard(websocket)
            
            # Remove connection data
            if websocket in self.connection_data:
                connection_type = self.connection_data[websocket]["type"]
                del self.connection_data[websocket]
                logger.info(f"WebSocket disconnected: {connection_type}")
                
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
    
    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        """Send a message to a specific WebSocket client"""
        try:
            if websocket in self.connection_data:
                self.connection_data[websocket]["last_activity"] = datetime.now()
            
            await websocket.send_text(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict, connection_type: str = "dashboard"):
        """Broadcast a message to all clients of a specific type"""
        if connection_type not in self.active_connections:
            return
        
        disconnected = set()
        
        for websocket in self.active_connections[connection_type]:
            try:
                await self.send_personal_message(message, websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def broadcast_fatigue_update(self, fatigue_data: Dict):
        """Broadcast fatigue detection updates"""
        message = {
            "type": "fatigue_update",
            "data": fatigue_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message, "fatigue")
    
    async def broadcast_energy_update(self, energy_data: Dict):
        """Broadcast energy prediction updates"""
        message = {
            "type": "energy_update",
            "data": energy_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message, "bio_rhythm")
    
    async def broadcast_schedule_update(self, schedule_data: Dict):
        """Broadcast schedule optimization updates"""
        message = {
            "type": "schedule_update",
            "data": schedule_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message, "dashboard")
    
    async def broadcast_alert(self, alert_data: Dict):
        """Broadcast system alerts"""
        message = {
            "type": "alert",
            "data": alert_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message, "dashboard")
    
    def get_connection_count(self, connection_type: str = None) -> int:
        """Get the number of active connections"""
        if connection_type:
            return len(self.active_connections.get(connection_type, set()))
        else:
            return sum(len(connections) for connections in self.active_connections.values())
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        stats = {
            "total_connections": self.get_connection_count(),
            "connections_by_type": {
                conn_type: len(connections) 
                for conn_type, connections in self.active_connections.items()
            },
            "connection_data": {
                conn_type: [
                    {
                        "connected_at": data["connected_at"].isoformat(),
                        "last_activity": data["last_activity"].isoformat(),
                        "user_id": data["user_id"]
                    }
                    for websocket, data in self.connection_data.items()
                    if data["type"] == conn_type
                ]
                for conn_type in self.active_connections.keys()
            }
        }
        return stats
    
    async def cleanup_inactive_connections(self, timeout_minutes: int = 30):
        """Clean up inactive connections"""
        current_time = datetime.now()
        inactive_websockets = []
        
        for websocket, data in self.connection_data.items():
            time_diff = (current_time - data["last_activity"]).total_seconds() / 60
            if time_diff > timeout_minutes:
                inactive_websockets.append(websocket)
        
        for websocket in inactive_websockets:
            logger.info(f"Cleaning up inactive WebSocket connection")
            self.disconnect(websocket)
    
    async def set_user_id(self, websocket: WebSocket, user_id: int):
        """Set user ID for a WebSocket connection"""
        if websocket in self.connection_data:
            self.connection_data[websocket]["user_id"] = user_id
            logger.info(f"Set user ID {user_id} for WebSocket connection")
    
    async def get_user_connections(self, user_id: int) -> List[WebSocket]:
        """Get all WebSocket connections for a specific user"""
        user_connections = []
        
        for websocket, data in self.connection_data.items():
            if data["user_id"] == user_id:
                user_connections.append(websocket)
        
        return user_connections
    
    async def send_to_user(self, user_id: int, message: Dict):
        """Send a message to all connections of a specific user"""
        user_connections = await self.get_user_connections(user_id)
        
        for websocket in user_connections:
            try:
                await self.send_personal_message(message, websocket)
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {e}")
                self.disconnect(websocket)