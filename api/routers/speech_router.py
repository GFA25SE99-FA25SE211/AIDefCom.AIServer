"""Speech router - WebSocket endpoint for real-time STT."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from api.dependencies import get_speech_service
from services.speech_service import SpeechService


router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/stt")
async def websocket_stt(
    ws: WebSocket,
    speech_service: SpeechService = Depends(get_speech_service),
) -> None:
    """
    WebSocket endpoint for streaming speech-to-text.
    
    Query parameters:
        - speaker: Initial speaker label (default: "Đang xác định")
        - phrases: Additional phrase hints separated by | or ,
        - user_id: Optional user ID for transcript association
        
    Client can send:
        - Binary data: Audio chunks for recognition
        - Text "stop": End session and save transcript to database
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

