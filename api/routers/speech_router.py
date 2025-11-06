"""Speech router - WebSocket endpoint for real-time STT."""

from __future__ import annotations

import asyncio
import re
import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState

from api.dependencies import get_speech_service
from services.speech_service import SpeechService
from services.redis_service import get_redis_service


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
        
    Client can send:
        - Binary data: Audio chunks for recognition
        - Text "stop": End session and save transcript to database
    """
    try:
        await ws.accept()
        print("✅ Client connected to /ws/stt")
    except Exception as e:
        import traceback
        print(f"❌ Failed to accept WebSocket: {e}")
        traceback.print_exc()
        return
    
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
    
    # Session data for transcript storage
    session_id = ws.headers.get("sec-websocket-key", "unknown")
    session_start = datetime.utcnow()
    transcript_lines = []
    
    # Redis service for caching
    redis_service = get_redis_service()
    
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
    
    # Audio queue to decouple WebSocket reading from speech processing
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=50)
    stop_event = asyncio.Event()
    
    async def websocket_reader() -> None:
        """Read messages from WebSocket and route binary/text appropriately."""
        try:
            while not stop_event.is_set():
                try:
                    # Check if WebSocket is still connected
                    if ws.application_state != WebSocketState.CONNECTED:
                        print("⚠️ WebSocket disconnected in reader")
                        break
                    
                    # Use receive() to handle both text and binary
                    data = await ws.receive()
                    
                    if "bytes" in data:
                        # Binary audio data -> push to queue
                        try:
                            audio_queue.put_nowait(data["bytes"])
                        except asyncio.QueueFull:
                            # Drop oldest frame if queue is full
                            try:
                                _ = audio_queue.get_nowait()
                                audio_queue.put_nowait(data["bytes"])
                            except asyncio.QueueEmpty:
                                # If queue is somehow empty now, just try putting again
                                try:
                                    audio_queue.put_nowait(data["bytes"])
                                except asyncio.QueueFull:
                                    # Skip this frame if still full
                                    print("⚠️ Audio queue full, dropping frame")
                    
                    elif "text" in data:
                        # Text command (e.g., "stop")
                        message = data["text"].strip().lower()
                        if message == "stop":
                            print("🛑 Received 'stop' command from client")
                            stop_event.set()
                            break
                    
                    else:
                        # Disconnect or other event
                        print("⚠️ Received disconnect event")
                        break
                        
                except WebSocketDisconnect:
                    print("🔌 WebSocket disconnected")
                    break
                except Exception as e:
                    print(f"⚠️ WebSocket read error: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        finally:
            # Signal end of stream
            try:
                await audio_queue.put(None)
            except Exception:
                pass
    
    reader_task = asyncio.create_task(websocket_reader())
    
    try:
        print(f"📡 Starting recognition stream...")
        # Stream recognition results
        async for event in speech_service.recognize_stream_from_queue(
            audio_queue,
            speaker_label=speaker_label,
            extra_phrases=extra_phrases,
        ):
            print(f"📨 Got event: {event.get('type')}")
            # Check if stop was requested
            if stop_event.is_set():
                print("🛑 Stop event set, breaking recognition loop")
                break
            
            # Check WebSocket state before sending
            if ws.application_state != WebSocketState.CONNECTED:
                print("⚠️ WebSocket not connected, stopping recognition")
                break
            
            try:
                await ws.send_json(event)
                
                # Collect transcript for final result events
                if event.get("type") == "result" and event.get("text"):
                    speaker = event.get("speaker", "Unknown")
                    text = event.get("text", "")
                    timestamp = datetime.utcnow().isoformat()
                    
                    transcript_lines.append({
                        "timestamp": timestamp,
                        "speaker": speaker,
                        "text": text,
                        "user_id": event.get("user_id"),
                    })
                    
                    # Cache the partial transcript in Redis
                    try:
                        cache_key = f"transcript:session:{session_id}"
                        await redis_service.set(cache_key, {
                            "session_id": session_id,
                            "start_time": session_start.isoformat(),
                            "lines": transcript_lines,
                        }, ttl=3600)
                    except Exception as cache_error:
                        print(f"⚠️ Failed to cache transcript: {cache_error}")
                    
            except WebSocketDisconnect:
                print("🔌 Client disconnected during send")
                break
            except Exception as send_error:
                print(f"⚠️ Failed to send event: {send_error}")
                break
    
    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        import traceback
        print(f"❌ Error in STT WebSocket: {e}")
        traceback.print_exc()
    finally:
        # Set stop event to signal all tasks to terminate
        stop_event.set()
        
        # Clean up tasks
        print("🧹 Cleaning up tasks...")
        
        # Cancel ping task
        if not pinger.done():
            pinger.cancel()
        try:
            await asyncio.wait_for(pinger, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        
        # Cancel reader task
        if not reader_task.done():
            reader_task.cancel()
        try:
            await asyncio.wait_for(reader_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        
        # If stop was requested or session ended, save transcript to database
        if transcript_lines:
            session_end = datetime.utcnow()
            duration_seconds = (session_end - session_start).total_seconds()
            
            transcript_data = {
                "session_id": session_id,
                "start_time": session_start.isoformat(),
                "end_time": session_end.isoformat(),
                "duration_seconds": duration_seconds,
                "initial_speaker": speaker_label,
                "lines": transcript_lines,
                "total_lines": len(transcript_lines),
            }
            
            # TODO: Gọi API bên ngoài để lưu transcript vào database
            # Ví dụ:
            # import httpx
            # async with httpx.AsyncClient() as client:
            #     response = await client.post(
            #         "https://your-api-endpoint.com/api/transcripts",
            #         json=transcript_data,
            #         headers={"Authorization": "Bearer YOUR_TOKEN"}
            #     )
            #     if response.status_code == 200:
            #         print(f"✅ Transcript saved to external API")
            #     else:
            #         print(f"❌ Failed to save transcript: {response.status_code}")
            
            print(f"💾 Ready to save transcript: {len(transcript_lines)} lines, {duration_seconds:.1f}s duration")
            print(f"📊 Transcript data: {json.dumps(transcript_data, indent=2, ensure_ascii=False)}")
            
            # Cache the final transcript in Redis
            try:
                cache_key = f"transcript:session:{session_id}:final"
                await redis_service.set(cache_key, transcript_data, ttl=86400)  # Keep for 24 hours
            except Exception as cache_error:
                print(f"⚠️ Failed to cache final transcript: {cache_error}")
            
            # Send confirmation to client if still connected
            try:
                if ws.application_state == WebSocketState.CONNECTED:
                    await ws.send_json({
                        "type": "session_saved",
                        "session_id": session_id,
                        "lines_saved": len(transcript_lines),
                        "duration": duration_seconds,
                    })
            except Exception as send_error:
                print(f"⚠️ Failed to send session_saved: {send_error}")
        
        # Clear question cache for this session
        try:
            from services.question_service import QuestionService
            question_service = QuestionService()
            deleted_count = await question_service.clear_questions(session_id)
            if deleted_count > 0:
                print(f"🗑️  Cleared {deleted_count} questions from cache (session: {session_id})")
        except Exception as clear_error:
            print(f"⚠️ Failed to clear question cache: {clear_error}")
        
        print("🔚 STT WebSocket closed")

