"""Speech router - WebSocket endpoint for real-time STT."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from api.dependencies import get_speech_service
from services.interfaces.i_speech_service import ISpeechService


router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/stt")
async def websocket_stt(
    ws: WebSocket,
    speech_service: ISpeechService = Depends(get_speech_service),
) -> None:
    """
    WebSocket endpoint for streaming speech-to-text.
    
    Query parameters (optional):
        - speaker: Initial speaker label (default: "Đang xác định")
        - phrases: Additional phrase hints separated by | or ,
        - defense_session_id: Defense session ID to filter speaker identification
        
    Backend automatically identifies speaker from audio - no need to send user_id.
    If defense_session_id is provided, speaker identification will only match against
    users enrolled in that defense session (fetched from /api/defense-sessions/{id}/users).
        
    Client can send:
        - Binary data: Audio chunks for recognition
        - Text "stop": End session and save transcript to external API
    """
    try:
        await ws.accept()
        logger.info("Client connected to /ws/stt")
    except Exception as e:
        logger.exception(f"Failed to accept WebSocket: {e}")
        return
    
    # Delegate all business logic to service
    try:
        await speech_service.handle_stt_session(ws)
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception(f"Error in STT WebSocket: {e}")

