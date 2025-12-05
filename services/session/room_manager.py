"""Session Room Manager - Manages WebSocket connections per defense session.

Enables real-time transcript broadcasting between all clients in the same session.
Uses Redis Pub/Sub for cross-process communication (important for multi-container deployments).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from typing import Any, Dict, Optional, Set
from weakref import WeakSet

from fastapi import WebSocket
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

# Stable container ID for multi-pod deployments
# Priority: CONTAINER_ID env > HOSTNAME env > socket.gethostname() > fallback
# This MUST be stable across the lifetime of the container instance
_CONTAINER_ID: Optional[str] = None


def get_container_id() -> str:
    """Get stable container ID for this instance.
    
    Uses environment variable or hostname - both are stable for container lifetime.
    Falls back to a generated ID only if both are unavailable.
    """
    global _CONTAINER_ID
    if _CONTAINER_ID is None:
        _CONTAINER_ID = (
            os.getenv("CONTAINER_ID") 
            or os.getenv("HOSTNAME") 
            or socket.gethostname()
            or f"container_{os.getpid()}"
        )
        logger.info(f"Container ID initialized: {_CONTAINER_ID}")
    return _CONTAINER_ID


class SessionRoomManager:
    """Manages WebSocket rooms for defense sessions.
    
    Each defense_session_id is a "room" where all connected clients
    receive broadcast messages (transcripts from any participant).
    """
    
    def __init__(self):
        # Map: defense_session_id -> set of WebSocket connections
        self._rooms: Dict[str, Set[WebSocket]] = {}
        # Lock for thread-safe room operations
        self._lock = asyncio.Lock()
        # Track subscriber tasks per room
        self._subscriber_tasks: Dict[str, asyncio.Task] = {}
        # Redis service (injected)
        self._redis_service = None
    
    def set_redis_service(self, redis_service) -> None:
        """Set Redis service for Pub/Sub."""
        self._redis_service = redis_service
    
    async def join_room(self, defense_session_id: str, ws: WebSocket) -> None:
        """Add a WebSocket connection to a room.
        
        Args:
            defense_session_id: The defense session ID (room identifier)
            ws: WebSocket connection to add
        """
        if not defense_session_id:
            return
        
        async with self._lock:
            if defense_session_id not in self._rooms:
                self._rooms[defense_session_id] = set()
                logger.info(f"🏠 Created room: {defense_session_id}")
                
                # Start Redis subscriber for this room (if Redis available)
                if self._redis_service:
                    self._start_subscriber(defense_session_id)
            
            self._rooms[defense_session_id].add(ws)
            room_size = len(self._rooms[defense_session_id])
            logger.info(f"👤 Client joined room {defense_session_id} | total={room_size}")
    
    async def leave_room(self, defense_session_id: str, ws: WebSocket) -> None:
        """Remove a WebSocket connection from a room.
        
        Args:
            defense_session_id: The defense session ID
            ws: WebSocket connection to remove
        """
        if not defense_session_id:
            return
        
        async with self._lock:
            if defense_session_id in self._rooms:
                self._rooms[defense_session_id].discard(ws)
                room_size = len(self._rooms[defense_session_id])
                logger.info(f"👋 Client left room {defense_session_id} | remaining={room_size}")
                
                # Cleanup empty rooms
                if room_size == 0:
                    del self._rooms[defense_session_id]
                    logger.info(f"🗑️ Removed empty room: {defense_session_id}")
                    
                    # Stop subscriber task
                    if defense_session_id in self._subscriber_tasks:
                        self._subscriber_tasks[defense_session_id].cancel()
                        del self._subscriber_tasks[defense_session_id]
    
    async def broadcast_to_room(
        self,
        defense_session_id: str,
        message: Dict[str, Any],
        exclude_ws: Optional[WebSocket] = None,
    ) -> int:
        """Broadcast message to all clients in a room.
        
        Args:
            defense_session_id: The room to broadcast to
            message: Message dict to send
            exclude_ws: Optional WebSocket to exclude (sender)
            
        Returns:
            Number of clients message was sent to
        """
        if not defense_session_id:
            return 0
        
        # ALWAYS broadcast local first (real-time, no delay)
        sent_count = await self._broadcast_local(defense_session_id, message, exclude_ws)
        
        # ALSO publish to Redis for cross-container broadcast (fire-and-forget)
        # Other containers' subscribers will handle their local connections
        if self._redis_service:
            channel = f"transcript:room:{defense_session_id}"
            try:
                # Add source marker using STABLE container ID to prevent echo
                # This ID must be consistent across the container's lifetime
                message_with_meta = {
                    **message,
                    "_source_container_id": get_container_id(),  # Stable container ID
                }
                # Fire-and-forget: don't await, don't block
                asyncio.create_task(self._publish_to_redis(channel, message_with_meta))
            except Exception as e:
                logger.debug(f"Redis publish scheduling failed: {e}")
        
        return sent_count
    
    async def _publish_to_redis(self, channel: str, message: Dict[str, Any]) -> None:
        """Publish to Redis (fire-and-forget helper)."""
        try:
            await asyncio.wait_for(
                self._redis_service.publish(channel, message),
                timeout=1.0  # Max 1s, don't block
            )
        except asyncio.TimeoutError:
            logger.debug(f"Redis publish timeout for {channel}")
        except Exception as e:
            logger.debug(f"Redis publish error: {e}")
    
    async def _broadcast_local(
        self,
        defense_session_id: str,
        message: Dict[str, Any],
        exclude_ws: Optional[WebSocket] = None,
    ) -> int:
        """Broadcast to local connections only."""
        sent_count = 0
        
        async with self._lock:
            if defense_session_id not in self._rooms:
                return 0
            
            # Copy set to avoid modification during iteration
            connections = list(self._rooms[defense_session_id])
        
        # Send outside lock to avoid blocking
        dead_connections = []
        for ws in connections:
            if ws == exclude_ws:
                continue
            
            try:
                if ws.application_state == WebSocketState.CONNECTED:
                    await ws.send_json(message)
                    sent_count += 1
                else:
                    dead_connections.append(ws)
            except Exception as e:
                logger.debug(f"Failed to send to client: {e}")
                dead_connections.append(ws)
        
        # Cleanup dead connections
        if dead_connections:
            async with self._lock:
                if defense_session_id in self._rooms:
                    for ws in dead_connections:
                        self._rooms[defense_session_id].discard(ws)
        
        return sent_count
    
    def _start_subscriber(self, defense_session_id: str) -> None:
        """Start Redis Pub/Sub subscriber for a room.
        
        This is for CROSS-CONTAINER communication only.
        Messages from same container are already broadcast locally.
        """
        if defense_session_id in self._subscriber_tasks:
            return
        
        async def _subscribe_loop():
            channel = f"transcript:room:{defense_session_id}"
            try:
                pubsub = await self._redis_service.subscribe(channel)
                if not pubsub:
                    return
                
                logger.info(f"📡 Started Redis subscriber for room {defense_session_id}")
                my_container_id = get_container_id()  # Stable container ID
                
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        try:
                            data = json.loads(message["data"])
                            
                            # Skip messages from same container (already broadcast locally)
                            source_container_id = data.pop("_source_container_id", None)
                            if source_container_id == my_container_id:
                                logger.debug(
                                    f"Skipping own message | room={defense_session_id} | "
                                    f"source={source_container_id}"
                                )
                                continue  # Skip - already handled locally
                            
                            # Remove other meta fields
                            data.pop("_exclude_ws_id", None)
                            
                            logger.debug(
                                f"Received cross-container message | room={defense_session_id} | "
                                f"from={source_container_id} | my_id={my_container_id}"
                            )
                            
                            # Broadcast to local connections (from OTHER containers)
                            await self._broadcast_local(defense_session_id, data, exclude_ws=None)
                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            logger.debug(f"Subscriber error: {e}")
            except asyncio.CancelledError:
                logger.info(f"📡 Stopped Redis subscriber for room {defense_session_id}")
            except Exception as e:
                logger.warning(f"Redis subscriber error for {defense_session_id}: {e}")
        
        task = asyncio.create_task(_subscribe_loop())
        self._subscriber_tasks[defense_session_id] = task
    
    def get_room_size(self, defense_session_id: str) -> int:
        """Get number of clients in a room."""
        if defense_session_id in self._rooms:
            return len(self._rooms[defense_session_id])
        return 0
    
    def get_all_rooms(self) -> Dict[str, int]:
        """Get all rooms with their sizes."""
        return {room_id: len(clients) for room_id, clients in self._rooms.items()}


# Global singleton instance
_room_manager: Optional[SessionRoomManager] = None


def get_session_room_manager() -> SessionRoomManager:
    """Get or create global room manager instance."""
    global _room_manager
    if _room_manager is None:
        _room_manager = SessionRoomManager()
    return _room_manager
