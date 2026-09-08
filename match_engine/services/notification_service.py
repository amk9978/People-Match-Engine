import json
import logging
from typing import Dict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for handling WebSocket notifications"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected via WebSocket")

    def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.info(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast_job_update(self, job_id: str, update: dict):
        """Broadcast job update to all connected clients"""
        message = {"type": "job_update", "job_id": job_id, **update}

        disconnected_clients = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.info(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)

        for client_id in disconnected_clients:
            self.disconnect(client_id)

    def get_connections_info(self) -> Dict:
        """Get information about active connections"""
        return {
            "active_connections": list(self.active_connections.keys()),
            "total_connections": len(self.active_connections),
        }
