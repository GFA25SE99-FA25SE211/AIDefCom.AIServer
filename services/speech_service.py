"""Speech Service - Business logic for speech-to-text streaming."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import AsyncGenerator, Dict, Any, Iterable, Optional, TYPE_CHECKING, List

import numpy as np

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from repositories.interfaces import ISpeechRepository
from services.interfaces.i_speech_service import ISpeechService
from services.interfaces.i_voice_service import IVoiceService
from services.interfaces.i_question_service import IQuestionService
from services.audio.utils import (
    NoiseFilter,
    pcm_to_wav,
    detect_energy_spike,
    detect_acoustic_change,
    calculate_rms,
)
from services.speech.text_utils import (
    filter_filler_words,
    normalize_vietnamese_text,
    should_log_transcript,
    calculate_speech_confidence,
)
from services.audio.buffer_manager import AudioBufferManager, AudioBufferConfig
from services.voice.speaker_identifier import (
    SpeakerIdentificationManager,
    SpeakerIdentificationConfig,
    normalize_speaker_label,
)
from services.voice.speaker_tracker import MultiSpeakerTracker
from services.session.room_manager import get_session_room_manager
from services.session.state_manager import (
    SessionStateManager,
    get_session_state_manager,
)
from services.speech.recognition_handler import RecognitionStreamHandler
from repositories.interfaces import IRedisService

logger = logging.getLogger(__name__)


def _log_suppressed_error(context: str, error: Exception, level: str = "debug") -> None:
    """Log suppressed errors consistently for debugging.
    
    Args:
        context: Where the error occurred (e.g., "ping_task", "redis_cache")
        error: The exception that was caught
        level: Log level - "debug", "warning", or "error"
    """
    msg = f"[{context}] Suppressed error: {type(error).__name__}: {error}"
    if level == "error":
        logger.error(msg)
    elif level == "warning":
        logger.warning(msg)
    else:
        logger.debug(msg)


def _get_speech_config():
    """Get speech service config with fallback defaults."""
    try:
        from app.config import Config
        return {
            "identify_min_seconds": Config.SPEAKER_IDENTIFY_MIN_SECONDS,
            "identify_window_seconds": Config.SPEAKER_IDENTIFY_WINDOW_SECONDS,
            "history_seconds": Config.SPEAKER_HISTORY_SECONDS,
            "identify_interval_seconds": Config.SPEAKER_IDENTIFY_INTERVAL_SECONDS,
            "redis_timeout_seconds": Config.SPEAKER_REDIS_TIMEOUT_SECONDS,
            "fallback_cosine_threshold": Config.SPEAKER_FALLBACK_COSINE_THRESHOLD,
            "fallback_margin_threshold": Config.SPEAKER_FALLBACK_MARGIN_THRESHOLD,
            "weak_cosine_threshold": Config.SPEAKER_WEAK_COSINE_THRESHOLD,
        }
    except Exception:
        # Fallback defaults
        return {
            "identify_min_seconds": 2.0,
            "identify_window_seconds": 3.0,
            "history_seconds": 5.0,
            "identify_interval_seconds": 0.3,
            "redis_timeout_seconds": 0.5,
            "fallback_cosine_threshold": 0.30,
            "fallback_margin_threshold": 0.06,
            "weak_cosine_threshold": 0.22,
        }


# Load config at module level (can be refreshed by reloading module)
_SPEECH_CONFIG = _get_speech_config()

# Constants from config
IDENTIFY_MIN_SECONDS = _SPEECH_CONFIG["identify_min_seconds"]
IDENTIFY_WINDOW_SECONDS = _SPEECH_CONFIG["identify_window_seconds"]
HISTORY_SECONDS = _SPEECH_CONFIG["history_seconds"]
IDENTIFY_INTERVAL_SECONDS = _SPEECH_CONFIG["identify_interval_seconds"]
REDIS_TIMEOUT_SECONDS = _SPEECH_CONFIG["redis_timeout_seconds"]
FALLBACK_COSINE_THRESHOLD = _SPEECH_CONFIG["fallback_cosine_threshold"]
FALLBACK_MARGIN_THRESHOLD = _SPEECH_CONFIG["fallback_margin_threshold"]
WEAK_COSINE_THRESHOLD = _SPEECH_CONFIG["weak_cosine_threshold"]


class SpeechService(ISpeechService):
    """Speech-to-text service with speaker identification and question capture support."""
    
    def __init__(
        self,
        azure_speech_repo: ISpeechRepository,
        voice_service: Optional[IVoiceService] = None,
        redis_service: Optional[IRedisService] = None,
        question_service: Optional[IQuestionService] = None,
    ) -> None:
        """
        Initialize speech service.
        
        Args:
            azure_speech_repo: Repository for Azure Speech operations
            voice_service: Optional voice authentication service
            redis_service: Optional Redis service for caching
            question_service: Optional question duplicate detection service
        """
        self.azure_speech_repo = azure_speech_repo
        self.voice_service = voice_service
        self.redis_service = redis_service
        self.question_service = question_service
        self.noise_filter = NoiseFilter()
        self.sample_rate = azure_speech_repo.sample_rate
        
        # Initialize recognition stream handler (extracted from this class for SRP)
        self._recognition_handler = RecognitionStreamHandler(
            speech_repository=azure_speech_repo,
            voice_service=voice_service,
            redis_service=redis_service,
            sample_rate=self.sample_rate,
        )
        
        # Initialize session state manager (Redis-backed for horizontal scaling)
        self._session_state = get_session_state_manager(redis_service)
        
        logger.info("Speech Service initialized (Redis caching, QuestionService=%s)", bool(self.question_service))

    def get_defense_session_users(self, session_id: str) -> Optional[List[str]]:
        """Return list of user IDs enrolled in a defense session (delegates to voice service).
        If voice service not available or session invalid, returns None.
        """
        if not self.voice_service:
            return None
        try:
            return self.voice_service.get_defense_session_users(session_id)
        except Exception as e:
            _log_suppressed_error("get_defense_session_users", e, "warning")
            return None
    
    def _score_to_confidence(self, score: Optional[float]) -> Optional[str]:
        """Map cosine similarity scores to human-readable confidence levels."""
        if score is None or self.voice_service is None:
            return None
        threshold = getattr(self.voice_service, "cosine_threshold", 0.75)
        if score >= threshold + 0.12:
            return "High"
        if score >= threshold + 0.05:
            return "Medium"
        return "Low"
    
    @staticmethod
    def _colorize(event_type: str, text: str) -> str:
        """Colorize text based on event type."""
        color_map = {
            "partial": "#3498db",
            "result": "#2ecc71",
            "nomatch": "#e67e22",
            "error": "#e74c3c",
        }
        color = color_map.get(event_type, "#bdc3c7")
        return f"<span style=\"color:{color}\">{text}</span>"
    
    @staticmethod
    def _create_tracked_task(background_tasks: set, coro) -> None:
        """Create background task with automatic cleanup to prevent leak."""
        task = asyncio.create_task(coro)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
    
    @staticmethod
    def _format_vtt_timestamp(seconds: float) -> str:
        """Format seconds to VTT timestamp (HH:MM:SS.mmm).
        
        Args:
            seconds: Elapsed time in seconds
            
        Returns:
            VTT-formatted timestamp string
            
        Example:
            123.456 -> "00:02:03.456"
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    async def recognize_stream_from_queue(
        self,
        audio_queue: asyncio.Queue[bytes | None],
        speaker_label: str = "Đang xác định",
        extra_phrases: Iterable[str] | None = None,
        apply_noise_filter: bool = True,
        whitelist_user_ids: Optional[List[str]] = None,
        defense_session_id: Optional[str] = None,
        user_info_map: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream speech recognition from audio queue (instead of WebSocket).
        
        This method delegates to RecognitionStreamHandler which handles:
        - Audio buffering and framing (via AudioBufferManager)
        - Speaker identification (via SpeakerIdentificationManager)
        - Azure Speech integration
        - Result processing and caching
        
        Args:
            audio_queue: Queue receiving audio bytes (None signals end)
            speaker_label: Initial speaker label
            extra_phrases: Additional phrase hints
            apply_noise_filter: Whether to apply noise filtering
            whitelist_user_ids: Optional list of user IDs to filter identification (defense session)
            defense_session_id: Optional defense session ID for profile preloading
            user_info_map: Optional dict mapping user_id -> {name, role, display_name}
        
        Yields:
            Recognition events
        """
        # Delegate to handler (extracted for SRP)
        async for event in self._recognition_handler.handle_stream(
            audio_queue=audio_queue,
            speaker_label=speaker_label,
            extra_phrases=extra_phrases,
            apply_noise_filter=apply_noise_filter,
            whitelist_user_ids=whitelist_user_ids,
            defense_session_id=defense_session_id,
            user_info_map=user_info_map,
        ):
            yield event

    async def recognize_stream(
        self,
        websocket: WebSocket,
        speaker_label: str = "Đang xác định",
        extra_phrases: Iterable[str] | None = None,
        apply_noise_filter: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream speech recognition with speaker identification.
        
        Args:
            websocket: WebSocket connection
            speaker_label: Initial speaker label
            extra_phrases: Additional phrase hints
            apply_noise_filter: Whether to apply noise filtering
        
        Yields:
            Recognition events
        """
        # Wrapper implementation: translate websocket bytes into a queue and
        # delegate the real work to `recognize_stream_from_queue` to avoid
        # duplicating the whole recognition pipeline.
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def _reader() -> None:
            try:
                while True:
                    try:
                        chunk = await websocket.receive_bytes()
                    except WebSocketDisconnect:
                        break
                    # None or empty? keep consistent with queue API
                    await audio_queue.put(chunk if chunk else b"")
            finally:
                # Signal end of stream
                await audio_queue.put(None)

        reader_task = asyncio.create_task(_reader())

        try:
            async for event in self.recognize_stream_from_queue(
                audio_queue,
                speaker_label=speaker_label,
                extra_phrases=extra_phrases,
                apply_noise_filter=apply_noise_filter,
            ):
                yield event
        finally:
            if reader_task and not reader_task.done():
                reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reader_task

    async def handle_stt_session(self, ws: WebSocket) -> None:
        """Handle complete STT WebSocket session with transcript persistence.
        
        This method encapsulates all business logic for an STT session:
        - Parse query parameters
        - Manage WebSocket connection (ping, reader)
        - Stream recognition events
        - Collect and cache transcript
        - Broadcast transcripts to all clients in same defense session
        - Save to database on completion
        - Clear question cache
        
        Args:
            ws: WebSocket connection (already accepted)
        """
        import re
        import json
        from datetime import datetime
        from starlette.websockets import WebSocketState
        
        # Get room manager for broadcast
        room_manager = get_session_room_manager()
        if self.redis_service:
            room_manager.set_redis_service(self.redis_service)
        
        # Parse query parameters
        speaker_label = ws.query_params.get("speaker") or "Đang xác định"
        phrase_param = ws.query_params.get("phrases")
        # Support both defense_session_id and session_id (backward compatibility)
        defense_session_id = ws.query_params.get("defense_session_id") or ws.query_params.get("session_id")
        # Note: user_id is NOT required - backend will auto-identify speaker from audio
        
        extra_phrases = []
        if phrase_param:
            extra_phrases = [
                token.strip()
                for token in re.split(r"[|,]", phrase_param)
                if token.strip()
            ]
        
        # Session metadata - MUST define before using
        session_id = ws.headers.get("sec-websocket-key", f"session_{id(ws)}")
        
        # === USE defense_session_id FOR TRANSCRIPT CACHE (enables resume after reload) ===
        transcript_cache_key = f"transcript:defense:{defense_session_id}" if defense_session_id else f"transcript:session:{session_id}"
        
        # Join room for this defense session (enables broadcast)
        if defense_session_id:
            await room_manager.join_room(defense_session_id, ws)
            room_size = room_manager.get_room_size(defense_session_id)
            logger.info(f"🏠 Joined room {defense_session_id} | room_size={room_size}")
        
        # Send immediate connected confirmation to client (before any blocking operations)
        try:
            await ws.send_json({
                "type": "connected",
                "session_id": session_id,
                "defense_session_id": defense_session_id,
                "room_size": room_manager.get_room_size(defense_session_id) if defense_session_id else 0,
                "message": "WebSocket connected, starting recognition..."
            })
            logger.info(f"✅ Sent connected event | session_id={session_id}")
        except Exception as e:
            logger.warning(f"Failed to send connected event: {e}")
        
        # Fetch defense session users whitelist in background (non-blocking)
        whitelist_user_ids: Optional[List[str]] = None
        user_info_map: Optional[Dict[str, Dict[str, str]]] = None  # Map user_id -> {name, role, display_name}
        whitelist_fetch_task = None
        
        async def _fetch_whitelist_background():
            """Fetch whitelist with user info in background - fire and forget."""
            nonlocal whitelist_user_ids, user_info_map
            if not defense_session_id or not self.voice_service:
                return
            try:
                # Fetch users with full info (name, role, display_name)
                result = await asyncio.wait_for(
                    self.voice_service.get_defense_session_users_with_info(defense_session_id),
                    timeout=2.0
                )
                if result:
                    user_info_map = result
                    whitelist_user_ids = list(result.keys())
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Whitelist fetch timeout")
            except Exception as e:
                logger.debug(f"Whitelist fetch skipped: {e}")
        
        # Start background fetch immediately (don't await)
        if defense_session_id and self.voice_service:
            whitelist_fetch_task = asyncio.create_task(_fetch_whitelist_background())
        session_start = datetime.utcnow()
        transcript_lines = []
        
        # === LOAD EXISTING TRANSCRIPT FROM CACHE AND SEND TO CLIENT ===
        # This ensures client gets latest transcript when reconnecting/reloading
        if defense_session_id and self.redis_service:
            try:
                existing = await asyncio.wait_for(
                    self.redis_service.get(transcript_cache_key),
                    timeout=2.0
                )
                if existing and isinstance(existing, dict):
                    cached_lines = existing.get("lines", [])
                    if cached_lines:
                        # Gửi transcript đã cache cho client
                        await ws.send_json({
                            "type": "cached_transcript",
                            "defense_session_id": defense_session_id,
                            "lines": cached_lines,
                            "start_time": existing.get("start_time"),
                            "message": f"Loaded {len(cached_lines)} lines from cache"
                        })
                        logger.info(f"📦 Sent cached transcript to client | lines={len(cached_lines)}")
                        
                        # Load vào biến local để tiếp tục append
                        transcript_lines = cached_lines
                        original_start = existing.get("start_time")
                        if original_start:
                            try:
                                session_start = datetime.fromisoformat(original_start)
                            except Exception as e:
                                _log_suppressed_error("parse_session_start", e, "warning")
                        logger.info(f"📂 Resumed transcript | defense_session_id={defense_session_id} | existing_lines={len(transcript_lines)}")
            except asyncio.TimeoutError:
                logger.debug("Timeout loading cached transcript")
            except Exception as e:
                logger.debug(f"No cached transcript: {e}")
        
        # === FLAG: Only save to DB when explicitly requested ===
        # Set to True when: session:end command OR save:transcript command
        # This prevents saving incomplete transcripts when user reloads page
        should_save_to_db = False
        
        # Audio queue and control (300 items = ~6s buffer at 50fps)
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=300)
        stop_event = asyncio.Event()
        
        # Ping task to keep connection alive
        async def ping_task() -> None:
            try:
                while not stop_event.is_set():
                    if ws.application_state != WebSocketState.CONNECTED:
                        break
                    try:
                        await ws.send_json({"type": "ping"})
                    except Exception as e:
                        _log_suppressed_error("ping_task.send", e, "debug")
                        break
                    await asyncio.sleep(25)
            except Exception as e:
                _log_suppressed_error("ping_task.loop", e, "debug")
        
        pinger = asyncio.create_task(ping_task())
        
        # WebSocket reader task
        async def websocket_reader() -> None:
            """Read audio chunks and commands from WebSocket."""
            nonlocal whitelist_user_ids  # Access whitelist from outer scope
            audio_chunk_count = 0
            
            try:
                while not stop_event.is_set():
                    try:
                        if ws.application_state != WebSocketState.CONNECTED:
                            logger.debug("WebSocket disconnected in reader")
                            break
                        
                        data = await ws.receive()
                        
                        if "bytes" in data:
                            # Audio data -> push to queue
                            audio_chunk_count += 1
                            if audio_chunk_count == 1:
                                logger.info(f"🎤 First audio chunk received! size={len(data['bytes'])} bytes")
                            elif audio_chunk_count % 100 == 0:
                                logger.info(f"🎤 Audio chunks received: {audio_chunk_count}")
                            try:
                                audio_queue.put_nowait(data["bytes"])
                            except asyncio.QueueFull:
                                # Drop oldest frame
                                try:
                                    _ = audio_queue.get_nowait()
                                    audio_queue.put_nowait(data["bytes"])
                                except asyncio.QueueEmpty:
                                    try:
                                        audio_queue.put_nowait(data["bytes"])
                                    except asyncio.QueueFull:
                                        logger.warning("Audio queue full, dropping frame")
                        
                        elif "text" in data:
                            # Text command (e.g., control messages: stop, q:start, q:end)
                            message_raw = data["text"].strip()
                            message = message_raw.lower()
                            if message == "stop":
                                logger.info("Received 'stop' command from client")
                                stop_event.set()
                                break
                            elif message == "q:start":
                                # Begin question capture mode (stored in Redis for scaling)
                                await self._session_state.set_question_mode(
                                    session_id, 
                                    active=True,
                                    defense_session_id=defense_session_id
                                )
                                try:
                                    await ws.send_json({
                                        "type": "question_mode_started",
                                        "session_id": session_id
                                    })
                                except Exception as e:
                                    _log_suppressed_error("send_question_mode_started", e, "warning")
                            elif message == "q:end":
                                # End question mode and process captured text
                                is_question_mode = await self._session_state.is_question_mode(session_id)
                                if is_question_mode:
                                    # Get accumulated question text from Redis
                                    question_text = await self._session_state.get_question_text(session_id)
                                    
                                    # Clear question mode in Redis
                                    await self._session_state.clear_question_buffer(session_id)
                                    
                                    logger.info(f"📝 Question mode ended | text='{question_text[:80]}...' if question_text else ''")
                                    
                                    # Send ACK immediately to avoid blocking
                                    try:
                                        await ws.send_json({
                                            "type": "question_mode_ended",
                                            "session_id": session_id,
                                            "question_text": question_text[:100] if question_text else "",
                                            "message": "⏳ Đang kiểm tra câu hỏi..."
                                        })
                                        logger.info("✅ Sent question_mode_ended ACK")
                                    except Exception as ack_err:
                                        logger.warning(f"Failed to send ACK: {ack_err}")
                                    
                                    # Process in background (fire-and-forget with tracking)
                                    if self.question_service and question_text:
                                        import time
                                        start_time = time.time()
                                        
                                        # IMPORTANT: Use defense_session_id for duplicate check (not ws session_id)
                                        # All questions in same defense session should be checked together
                                        question_session_key = defense_session_id or session_id
                                        
                                        async def _check_and_register_bg():
                                            try:
                                                logger.info(f"🔄 Starting background question check | session={question_session_key} | text='{question_text[:50]}'")
                                                result = await asyncio.wait_for(
                                                    self.question_service.check_and_register(
                                                        session_id=question_session_key,  # Use defense_session_id!
                                                        question_text=question_text,
                                                        speaker=speaker_label or "Khách"
                                                    ),
                                                    timeout=15.0  # Increased: model load can take 10s+
                                                )
                                                elapsed = time.time() - start_time
                                                logger.info(f"✅ Question check completed in {elapsed:.2f}s | duplicate={result.get('is_duplicate')}")
                                                
                                                # Send result when done
                                                try:
                                                    if ws.application_state == WebSocketState.CONNECTED:
                                                        result["type"] = "question_mode_result"
                                                        result["session_id"] = session_id
                                                        await ws.send_json(result)
                                                        logger.info("✅ Sent question_mode_result")
                                                        
                                                        # === BROADCAST question_mode_result TO ALL CLIENTS ===
                                                        if defense_session_id:
                                                            broadcast_event = {
                                                                "type": "broadcast_question_result",
                                                                "question_text": question_text,
                                                                "is_duplicate": result.get("is_duplicate", False),
                                                                "similar": result.get("similar", []),
                                                                "speaker": speaker_label or "Member",
                                                                "source_session_id": session_id,
                                                            }
                                                            try:
                                                                await room_manager.broadcast_to_room(
                                                                    defense_session_id,
                                                                    broadcast_event,
                                                                    exclude_ws=ws,  # Không gửi lại cho người đặt
                                                                )
                                                                logger.debug(f"📢 Broadcast question_result to room {defense_session_id}")
                                                            except Exception as e:
                                                                _log_suppressed_error("broadcast_question_result", e, "warning")
                                                    else:
                                                        logger.warning("⚠️ WS not connected, cannot send result")
                                                except Exception as send_err:
                                                    logger.warning(f"Failed to send result: {send_err}")
                                            except asyncio.TimeoutError:
                                                logger.error(f"❌ Question check timeout (>15s) | text='{question_text[:50]}'")
                                                try:
                                                    if ws.application_state == WebSocketState.CONNECTED:
                                                        await ws.send_json({
                                                            "type": "question_mode_result",
                                                            "session_id": session_id,
                                                            "question_text": question_text,
                                                            "error": "Timeout: Kiểm tra câu hỏi quá lâu",
                                                            "is_duplicate": False,
                                                            "registered": False,
                                                            "question_id": None,
                                                            "total_questions": 0,
                                                            "similar": []
                                                        })
                                                except Exception as timeout_send_err:
                                                    _log_suppressed_error("send_timeout_response", timeout_send_err, "warning")
                                            except Exception as e:
                                                logger.error(f"❌ Background question check failed: {e}", exc_info=True)
                                                # Send error response
                                                try:
                                                    if ws.application_state == WebSocketState.CONNECTED:
                                                        await ws.send_json({
                                                            "type": "question_mode_result",
                                                            "session_id": session_id,
                                                            "question_text": question_text,
                                                            "error": str(e),
                                                            "is_duplicate": False,
                                                            "registered": False,
                                                            "question_id": None,
                                                            "total_questions": 0,
                                                            "similar": []
                                                        })
                                                except Exception as err_send_err:
                                                    _log_suppressed_error("send_error_response", err_send_err, "warning")
                                        
                                        # Create task and keep reference to prevent GC
                                        task = asyncio.create_task(_check_and_register_bg())
                                        # Store in session-level task tracker if needed
                                        logger.info("🚀 Background task created (fire-and-forget)")
                                    else:
                                        logger.warning("⚠️ QuestionService unavailable or empty text")
                                    
                                # Buffer already cleared above via clear_question_buffer()
                            elif message == "session:start":
                                # Thư ký bắt đầu ghi âm - broadcast cho member
                                logger.info(f"📢 Broadcasting session:start to room {defense_session_id}")
                                await room_manager.broadcast_to_room(
                                    defense_session_id,
                                    {"type": "session_started"},
                                    exclude_ws=None  # Gửi cho tất cả, kể cả thư ký (để confirm)
                                )
                            elif message == "session:end":
                                # Thư ký kết thúc phiên - broadcast cho member và save transcript
                                nonlocal should_save_to_db
                                should_save_to_db = True  # Mark to save transcript to DB
                                logger.info(f"📢 Broadcasting session:end to room {defense_session_id} | will_save_db=True")
                                await room_manager.broadcast_to_room(
                                    defense_session_id,
                                    {"type": "session_ended"},
                                    exclude_ws=None
                                )
                            elif message == "question:started":
                                # Member bắt đầu đặt câu hỏi - broadcast cho thư ký
                                if defense_session_id:
                                    logger.info(f"📢 Broadcasting question:started to room {defense_session_id}")
                                    await room_manager.broadcast_to_room(
                                        defense_session_id,
                                        {
                                            "type": "broadcast_question_started",
                                            "speaker": speaker_label or "Member",
                                            "source_session_id": session_id,
                                        },
                                        exclude_ws=ws,
                                    )
                            elif message == "question:processing":
                                # Member kết thúc đặt câu hỏi, đang xử lý - broadcast cho thư ký
                                if defense_session_id:
                                    logger.info(f"📢 Broadcasting question:processing to room {defense_session_id}")
                                    await room_manager.broadcast_to_room(
                                        defense_session_id,
                                        {
                                            "type": "broadcast_question_processing",
                                            "speaker": speaker_label or "Member",
                                            "source_session_id": session_id,
                                        },
                                        exclude_ws=ws,
                                    )
                            elif message_raw.startswith("transcript:edit:"):
                                # FE chỉnh sửa transcript line
                                # Format: transcript:edit:{json}
                                # JSON: {"id": "...", "edited_speaker": "...", "edited_text": "..."}
                                try:
                                    edit_json = message_raw[len("transcript:edit:"):]
                                    edit_data = json.loads(edit_json)
                                    line_id = edit_data.get("id")
                                    edited_speaker = edit_data.get("edited_speaker")
                                    edited_text = edit_data.get("edited_text")
                                    
                                    # Find and update the line in transcript_lines
                                    for line in transcript_lines:
                                        if line.get("id") == line_id:
                                            if edited_speaker:
                                                line["edited_speaker"] = edited_speaker
                                            if edited_text:
                                                line["edited_text"] = edited_text
                                            logger.info(f"📝 Transcript edit | id={line_id} | edited_speaker={edited_speaker} | edited_text={edited_text[:30] if edited_text else None}")
                                            break
                                    
                                    # Update Redis cache
                                    if self.redis_service:
                                        await self.redis_service.set(
                                            transcript_cache_key,
                                            {
                                                "defense_session_id": defense_session_id,
                                                "session_id": session_id,
                                                "start_time": session_start.isoformat(),
                                                "lines": transcript_lines,
                                            },
                                            ttl=7200,
                                        )
                                    
                                    # Broadcast edit to other clients
                                    if defense_session_id:
                                        await room_manager.broadcast_to_room(
                                            defense_session_id,
                                            {
                                                "type": "transcript_edited",
                                                "id": line_id,
                                                "edited_speaker": edited_speaker,
                                                "edited_text": edited_text,
                                                "source_session_id": session_id,
                                            },
                                            exclude_ws=ws,
                                        )
                                    
                                    await ws.send_json({
                                        "type": "transcript_edit_success",
                                        "id": line_id,
                                    })
                                except Exception as edit_err:
                                    logger.warning(f"Failed to edit transcript: {edit_err}")
                                    await ws.send_json({
                                        "type": "transcript_edit_error",
                                        "error": str(edit_err),
                                    })
                        else:
                            # Disconnect event
                            logger.debug("Received disconnect event")
                            break
                    
                    except WebSocketDisconnect:
                        logger.info("WebSocket disconnected")
                        break
                    except Exception as e:
                        logger.exception(f"WebSocket read error: {e}")
                        break
            finally:
                # Signal end of stream
                try:
                    await audio_queue.put(None)
                except Exception:
                    pass
        
        reader_task = asyncio.create_task(websocket_reader())
        
        try:
            logger.info(f"🚀 Starting STT session | session_id={session_id} | defense_session_id={defense_session_id}")
            logger.info(f"🔧 voice_service={self.voice_service is not None} | question_service={self.question_service is not None}")
            
            # Wait for whitelist fetch (with short timeout) - important for speaker identification!
            if whitelist_fetch_task and not whitelist_fetch_task.done():
                try:
                    await asyncio.wait_for(whitelist_fetch_task, timeout=3.0)
                    logger.info(f"✅ Whitelist ready: {len(whitelist_user_ids) if whitelist_user_ids else 0} users | IDs={whitelist_user_ids}")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Whitelist fetch timeout, proceeding without whitelist")
            else:
                logger.info(f"📋 Whitelist status: task={whitelist_fetch_task} | user_ids={whitelist_user_ids}")
            
            # Stream recognition results
            # Pass defense_session_id and user_info_map to enable voice profile preloading with display names
            async for event in self.recognize_stream_from_queue(
                audio_queue,
                speaker_label=speaker_label,
                extra_phrases=extra_phrases,
                whitelist_user_ids=whitelist_user_ids,
                defense_session_id=defense_session_id,
                user_info_map=user_info_map,
            ):
                # Check stop condition
                if stop_event.is_set():
                    logger.debug("Stop event set, breaking recognition loop")
                    break
                
                # Check WebSocket state
                if ws.application_state != WebSocketState.CONNECTED:
                    logger.debug("WebSocket not connected, stopping recognition")
                    break
                
                try:
                    # Check WS state before sending
                    if ws.application_state == WebSocketState.CONNECTED:
                        await ws.send_json(event)
                    else:
                        logger.debug("WS not connected, skipping event send")
                        break
                    
                    # === BROADCAST PARTIAL TEXT TO ALL CLIENTS (real-time typing) ===
                    if event.get("type") == "partial" and event.get("text") and defense_session_id:
                        broadcast_partial = {
                            "type": "broadcast_partial",
                            "speaker": event.get("speaker", "Unknown"),
                            "text": event.get("text"),
                            "source_session_id": session_id,
                        }
                        try:
                            await room_manager.broadcast_to_room(
                                defense_session_id,
                                broadcast_partial,
                                exclude_ws=ws,
                            )
                        except Exception:
                            pass  # Don't log partial broadcast failures
                    
                    # Collect final transcripts (skip noise events)
                    if event.get("type") == "result" and event.get("text"):
                        speaker = event.get("speaker", "Unknown")
                        text = event.get("text", "")
                        timestamp = datetime.utcnow().isoformat()
                        
                        # Calculate VTT-style timestamp (relative to session start)
                        elapsed = (datetime.utcnow() - session_start).total_seconds()
                        start_time_vtt = self._format_vtt_timestamp(elapsed)
                        end_time_vtt = self._format_vtt_timestamp(elapsed + 3.0)  # Assume 3s duration
                        
                        # Generate unique ID for this line
                        import uuid
                        line_id = f"stt_{int(elapsed * 1000)}_{uuid.uuid4().hex[:8]}"
                        
                        transcript_lines.append({
                            "id": line_id,
                            "timestamp": timestamp,
                            "start_time_vtt": start_time_vtt,
                            "end_time_vtt": end_time_vtt,
                            "speaker": speaker,
                            "text": text,
                            "user_id": event.get("user_id"),
                            # FE can populate these when editing:
                            "edited_speaker": None,
                            "edited_text": None,
                        })
                        
                        # === BROADCAST TO ALL CLIENTS IN SAME DEFENSE SESSION ===
                        if defense_session_id:
                            broadcast_event = {
                                "type": "broadcast_transcript",
                                "id": line_id,
                                "speaker": speaker,
                                "text": text,
                                "user_id": event.get("user_id"),
                                "timestamp": timestamp,
                                "start_time_vtt": start_time_vtt,
                                "end_time_vtt": end_time_vtt,
                                "confidence": event.get("confidence_adjusted"),
                                "source_session_id": session_id,  # Who sent this
                            }
                            try:
                                sent_count = await room_manager.broadcast_to_room(
                                    defense_session_id,
                                    broadcast_event,
                                    exclude_ws=ws,  # Don't send back to sender (they already got it)
                                )
                                if sent_count > 0:
                                    logger.debug(f"📢 Broadcast to {sent_count} clients in room {defense_session_id}")
                            except Exception as broadcast_err:
                                logger.debug(f"Broadcast failed: {broadcast_err}")
                        
                        # If question mode active, buffer this final text (stored in Redis)
                        is_question_mode = await self._session_state.is_question_mode(session_id)
                        if is_question_mode:
                            await self._session_state.append_question_buffer(session_id, text)
                        
                        # Cache partial transcript in Redis (using defense_session_id key for resume support)
                        try:
                            await self.redis_service.set(
                                transcript_cache_key,
                                {
                                    "defense_session_id": defense_session_id,
                                    "session_id": session_id,
                                    "start_time": session_start.isoformat(),
                                    "lines": transcript_lines,
                                },
                                ttl=7200,  # 2 hours for defense session
                            )
                        except Exception as cache_error:
                            logger.warning(f"Failed to cache transcript: {cache_error}")
                
                except WebSocketDisconnect:
                    logger.info("Client disconnected during send")
                    break
                except Exception as send_error:
                    logger.warning(f"Failed to send event: {send_error}")
                    break
        
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.exception(f"Error in STT session: {e}")
        finally:
            # === LEAVE ROOM ON DISCONNECT ===
            if defense_session_id:
                await room_manager.leave_room(defense_session_id, ws)
                logger.info(f"👋 Left room {defense_session_id}")
            
            # Cleanup
            stop_event.set()
            
            logger.debug("Cleaning up tasks...")
            
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
            
            # Save transcript if there are any lines
            if transcript_lines:
                session_end = datetime.utcnow()
                duration_seconds = (session_end - session_start).total_seconds()

                # Build full_text in VTT-style format:
                # HH:MM:SS.mmm --> HH:MM:SS.mmm
                # [Speaker] Text
                # [Edited Speaker] Edited Text  (only if edited)
                parts: List[str] = []
                for line in transcript_lines:
                    # Get original values
                    original_speaker = line.get("speaker", "Unknown")
                    original_text = (line.get("text") or "").strip()
                    if not original_text:
                        raw = (line.get("text_raw") or "").strip()
                        if raw:
                            original_text = raw
                    
                    # Get edited values (if any)
                    edited_speaker = line.get("edited_speaker")
                    edited_text = line.get("edited_text")
                    
                    # Get VTT timestamps
                    start_vtt = line.get("start_time_vtt", "00:00:00.000")
                    end_vtt = line.get("end_time_vtt", "00:00:03.000")
                    
                    if original_text or edited_text:
                        # VTT timestamp line
                        vtt_line = f"{start_vtt} --> {end_vtt}"
                        # Original speaker line
                        speaker_line = f"[{original_speaker}] {original_text}"
                        
                        if edited_speaker or edited_text:
                            # Has edit - add edited line
                            edit_speaker_display = edited_speaker or original_speaker
                            edit_text_display = edited_text or original_text
                            edit_line = f"[{edit_speaker_display}] {edit_text_display}"
                            parts.append(f"{vtt_line}\n{speaker_line}\n{edit_line}")
                        else:
                            # No edit - leave second line empty
                            parts.append(f"{vtt_line}\n{speaker_line}\n")
                
                full_text = "\n\n".join(parts).strip()

                MIN_TRANSCRIPT_CHARS = 12
                logger.info(
                    f"📝 Transcript summary | session_id={session_id} | lines={len(transcript_lines)} | length={len(full_text)} | preview='{full_text[:100]}'"
                )

                # Conditions to skip saving
                if not full_text:
                    logger.warning(f"⚠️ Skip save: empty transcript | session_id={session_id}")
                elif len(full_text) < MIN_TRANSCRIPT_CHARS:
                    logger.warning(
                        f"⚠️ Skip save: too short (<{MIN_TRANSCRIPT_CHARS}) | session_id={session_id} | text='{full_text}'"
                    )
                else:
                    transcript_data = {
                        "session_id": session_id,
                        "start_time": session_start.isoformat(),
                        "end_time": session_end.isoformat(),
                        "duration_seconds": duration_seconds,
                        "initial_speaker": speaker_label,
                        "lines": transcript_lines,
                        "total_lines": len(transcript_lines),
                    }

                    logger.info(
                        f"Transcript collected | session_id={session_id} | lines={len(transcript_lines)} | duration={duration_seconds:.1f}s"
                    )

                    # Save to external API endpoint: POST /api/transcripts with retry (diagnostic logging)
                    # IMPORTANT: Requires valid numeric defense_session_id (FK to DefenseSession table)
                    # IMPORTANT: Only save when should_save_to_db=True (session:end or save:transcript command)
                    if not should_save_to_db:
                        logger.info(
                            f"💡 Skip DB save: no save command received (user may have reloaded) | "
                            f"session_id={session_id} | lines={len(transcript_lines)}"
                        )
                        logger.info("💡 Transcript cached in Redis only. Send 'session:end' or 'save:transcript' to save to DB.")
                    else:
                        # === CHECK FOR GUEST SPEAKERS BEFORE SAVING ===
                        guest_lines = []
                        for line in transcript_lines:
                            # Prioritize edited_speaker, fall back to original speaker
                            effective_speaker = line.get("edited_speaker") or line.get("speaker", "")
                            effective_speaker_lower = effective_speaker.lower().strip()
                            if effective_speaker_lower in {"khách", "guest", "unknown", "đang xác định"}:
                                guest_lines.append({
                                    "id": line.get("id"),
                                    "speaker": effective_speaker,
                                    "text": (line.get("edited_text") or line.get("text", ""))[:50],
                                })
                        
                        if guest_lines:
                            # Block save - still have guest speakers
                            logger.warning(
                                f"⚠️ Cannot save transcript: {len(guest_lines)} lines with guest speaker | "
                                f"session_id={session_id}"
                            )
                            try:
                                if ws.application_state == WebSocketState.CONNECTED:
                                    await ws.send_json({
                                        "type": "save_blocked",
                                        "reason": "guest_speaker",
                                        "message": f"Không thể lưu: còn {len(guest_lines)} dòng chưa xác định người nói",
                                        "guest_lines": guest_lines[:10],  # First 10 for debugging
                                        "total_guest_lines": len(guest_lines),
                                    })
                            except Exception:
                                pass
                            # Don't save - skip to cache only
                            should_save_to_db = False
                        
                        if should_save_to_db and (not defense_session_id or not defense_session_id.isdigit()):
                            logger.warning(
                                f"⚠️ Skip API save: defense_session_id not provided or invalid | "
                                f"defense_session_id={defense_session_id} | session_id={session_id}"
                            )
                            logger.info("💡 Transcript cached in Redis only (no database save)")
                        elif should_save_to_db:
                            try:
                                from app.config import Config
                                import httpx

                                api_url = f"{Config.AUTH_SERVICE_BASE_URL}/api/transcripts"
                                session_id_int = int(defense_session_id)

                                payload = {
                                    "sessionId": session_id_int,
                                    "transcriptText": full_text,
                                    "isApproved": True,
                                }

                                logger.info(
                                    f"📤 Attempting transcript save | defense_session_id={session_id_int} | chars={len(full_text)}"
                                )

                                saved = False
                                last_error = None
                                for attempt in range(3):
                                    try:
                                        async with httpx.AsyncClient(
                                            verify=Config.AUTH_SERVICE_VERIFY_SSL,
                                            timeout=Config.AUTH_SERVICE_TIMEOUT,
                                        ) as client:
                                            response = await client.post(api_url, json=payload)
                                        if response.status_code in (200, 201):
                                            logger.info(
                                                f"✅ Saved transcript | attempt={attempt+1} status={response.status_code}"
                                            )
                                            saved = True
                                            break
                                        else:
                                            last_error = f"HTTP {response.status_code}: {response.text}"
                                            if attempt < 2:
                                                logger.warning(
                                                    f"⚠️ Retry {attempt+1}/3 saving transcript | status={response.status_code}"
                                                )
                                                await asyncio.sleep(2 ** attempt)
                                    except Exception as req_err:
                                        last_error = str(req_err)
                                        if attempt < 2:
                                            logger.warning(
                                                f"⚠️ Retry {attempt+1}/3 error: {req_err}"
                                            )
                                            await asyncio.sleep(2 ** attempt)

                                if not saved:
                                    logger.error(
                                        f"❌ Failed to save transcript after retries | defense_session_id={session_id_int} | error={last_error}"
                                    )
                            except Exception as api_err:
                                logger.exception(f"Critical error saving transcript: {api_err}")

                    # Cache final transcript (only if we had valid content)
                    try:
                        await self.redis_service.set(transcript_cache_key, transcript_data, ttl=86400)
                        # Also cache with :final suffix for explicit final version
                        await self.redis_service.set(f"{transcript_cache_key}:final", transcript_data, ttl=86400)
                    except Exception as cache_err:
                        logger.warning(f"Failed to cache final transcript: {cache_err}")

                    # Notify client
                    try:
                        if ws.application_state == WebSocketState.CONNECTED:
                            await ws.send_json({
                                "type": "session_saved",
                                "session_id": session_id,
                                "lines_saved": len(transcript_lines),
                                "duration": duration_seconds,
                            })
                    except Exception:
                        pass
            
            # Clear question cache for this defense session (use injected service if available)
            # NOTE: Only clear if this was the last client in the room, or use defense_session_id
            if self.question_service and defense_session_id:
                # Don't auto-clear - questions should persist until session officially ends
                # Questions are stored per defense_session_id, not per WS connection
                pass
            
            logger.info(f"STT session closed | session_id={session_id} | defense_session_id={defense_session_id}")
