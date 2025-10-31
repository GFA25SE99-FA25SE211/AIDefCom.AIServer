"""Speech router - WebSocket endpoint for real-time STT."""

from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState

from api.dependencies import get_speech_service
from services.speech_service import SpeechService


router = APIRouter()


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
    """
    await ws.accept()
    print("✅ Client connected to /ws/stt")
    
    # Get query parameters
    speaker_label = ws.query_params.get("speaker") or "Đang xác định"
    phrase_param = ws.query_params.get("phrases")
    
    extra_phrases = []
    if phrase_param:
        extra_phrases = [
            token.strip()
            for token in re.split(r"[|,]", phrase_param)
            if token.strip()
        ]
    
    # Ping task to keep connection alive
    async def ping_task() -> None:
        try:
            while True:
                if ws.application_state != WebSocketState.CONNECTED:
                    break
                await ws.send_json({"type": "ping"})
                await asyncio.sleep(25)
        except Exception:
            pass
    
    pinger = asyncio.create_task(ping_task())
    
    try:
        # Stream recognition results
        async for event in speech_service.recognize_stream(
            ws,
            speaker_label=speaker_label,
            extra_phrases=extra_phrases,
        ):
            try:
                await ws.send_json(event)
            except Exception:
                break
    
    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"❌ Error in STT WebSocket: {e}")
    finally:
        pinger.cancel()
        try:
            await pinger
        except asyncio.CancelledError:
            pass
        print("🔚 STT WebSocket closed")
