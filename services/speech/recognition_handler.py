"""Recognition Stream Handler - Handles the core speech recognition stream logic.

This module extracts the recognize_stream_from_queue logic from SpeechService
to follow Single Responsibility Principle.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import functools
from typing import AsyncGenerator, Dict, Any, Iterable, Optional, List, TYPE_CHECKING

import numpy as np

from services.audio.buffer_manager import AudioBufferManager, AudioBufferConfig
from services.voice.speaker_identifier import (
    SpeakerIdentificationManager,
    SpeakerIdentificationConfig,
    normalize_speaker_label,
)
from services.speech.text_utils import (
    normalize_vietnamese_text,
    calculate_speech_confidence,
)
from services.voice.speaker_tracker import MultiSpeakerTracker

if TYPE_CHECKING:
    from services.interfaces.i_voice_service import IVoiceService
    from services.interfaces.i_speech_service import ISpeechService
    from repositories.interfaces import IRedisService
    from repositories.interfaces import ISpeechRepository

logger = logging.getLogger(__name__)


def _get_recognition_config():
    """Load recognition config from app config with fallbacks."""
    try:
        from app.config import Config
        return {
            "history_seconds": Config.SPEAKER_HISTORY_SECONDS,
            "identify_min_seconds": Config.SPEAKER_IDENTIFY_MIN_SECONDS,
            "identify_window_seconds": Config.SPEAKER_IDENTIFY_WINDOW_SECONDS,
        }
    except Exception:
        return {
            "history_seconds": 5.0,
            "identify_min_seconds": 2.0,
            "identify_window_seconds": 3.0,
        }


class RecognitionStreamHandler:
    """Handles speech recognition stream with speaker identification.
    
    This class encapsulates the complex logic of:
    - Audio buffering and framing
    - Speaker identification and switching
    - Azure Speech integration
    - Result processing and caching
    
    Extracted from SpeechService to follow SRP.
    """
    
    def __init__(
        self,
        speech_repository: "ISpeechRepository",
        voice_service: Optional["IVoiceService"] = None,
        redis_service: Optional["IRedisService"] = None,
        sample_rate: int = 16000,
    ):
        """
        Initialize recognition stream handler.
        
        Args:
            speech_repository: Repository for speech recognition
            voice_service: Optional voice authentication service
            redis_service: Optional Redis service for caching
            sample_rate: Audio sample rate
        """
        self.speech_repository = speech_repository
        self.voice_service = voice_service
        self.redis_service = redis_service
        self.sample_rate = sample_rate
        
        # Load config
        self._config = _get_recognition_config()
    
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
    
    async def _preload_voice_profiles(
        self,
        defense_session_id: str,
        whitelist_user_ids: List[str],
        user_info_map: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Preload voice profiles for defense session.
        
        Uses shared voice executor to avoid blocking event loop.
        """
        if not self.voice_service:
            return None
        
        try:
            from core.executors import run_voice_bound
            
            profiles = await run_voice_bound(
                self.voice_service.preload_session_profiles,
                defense_session_id,
                whitelist_user_ids,
                user_info_map=user_info_map,
            )
            logger.info(f"✅ Preloaded {len(profiles)} voice profiles for session {defense_session_id}")
            return profiles
        except Exception as e:
            logger.warning(f"⚠️ Failed to preload profiles: {e}")
            return None
    
    async def handle_stream(
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
        Handle speech recognition stream with speaker identification.
        
        Args:
            audio_queue: Queue receiving audio bytes (None signals end)
            speaker_label: Initial speaker label
            extra_phrases: Additional phrase hints
            apply_noise_filter: Whether to apply noise filtering
            whitelist_user_ids: Optional list of user IDs for identification
            defense_session_id: Optional defense session ID
            user_info_map: Optional user info mapping
        
        Yields:
            Recognition events
        """
        logger.info(f"🎙️ Starting recognition stream (speaker={speaker_label}, filter={apply_noise_filter})")
        loop = asyncio.get_running_loop()
        
        # Preload voice profiles if session provided
        preloaded_profiles: Optional[List[Dict[str, Any]]] = None
        if defense_session_id and whitelist_user_ids:
            preloaded_profiles = await self._preload_voice_profiles(
                defense_session_id,
                whitelist_user_ids,
                user_info_map,
            )
        
        # Initialize managers
        buffer_config = AudioBufferConfig(
            sample_rate=self.sample_rate,
            history_seconds=self._config["history_seconds"],
            min_identify_seconds=self._config["identify_min_seconds"],
            apply_noise_filter=apply_noise_filter,
        )
        audio_buffer = AudioBufferManager(config=buffer_config)
        
        speaker_id_manager: Optional[SpeakerIdentificationManager] = None
        if self.voice_service:
            speaker_id_manager = SpeakerIdentificationManager(
                voice_service=self.voice_service,
                redis_service=self.redis_service,
                config=SpeakerIdentificationConfig.from_app_config(),
                preloaded_profiles=preloaded_profiles,
                whitelist_user_ids=whitelist_user_ids,
            )
        
        # Initialize speaker tracker
        speaker_tracker = MultiSpeakerTracker(max_speakers=4, inactivity_timeout=30.0)
        current_speaker = normalize_speaker_label(speaker_label)
        speaker_tracker.switch_speaker(
            current_speaker,
            None,
            timestamp=loop.time(),
            reason="initial"
        )
        
        # Generate session ID
        session_id = hashlib.md5(f"{time.time()}_{id(audio_queue)}".encode()).hexdigest()[:16]
        
        # State
        result_queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue(maxsize=8)
        background_tasks: set[asyncio.Task] = set()
        identify_task: Optional[asyncio.Task] = None
        interruption_detected = False
        
        async def on_interruption_detected():
            """Callback when audio interruption is detected."""
            nonlocal interruption_detected
            if speaker_id_manager and not interruption_detected:
                logger.debug("⚡ Interruption detected - scheduling identification")
                interruption_detected = True
                self._create_tracked_task(
                    background_tasks,
                    schedule_identification(force=True)
                )
        
        async def on_sufficient_audio():
            """Callback when enough audio is buffered."""
            nonlocal interruption_detected
            if speaker_id_manager:
                # Only schedule if speaker unknown or interrupted
                current = speaker_id_manager.current_speaker
                if current == "Khách" or speaker_id_manager.current_user_id is None or interruption_detected:
                    self._create_tracked_task(
                        background_tasks,
                        schedule_identification(force=True)
                    )
        
        async def schedule_identification(force: bool = False, blocking: bool = False):
            """Schedule speaker identification task."""
            nonlocal identify_task
            
            if not speaker_id_manager:
                return
            
            if identify_task and not identify_task.done():
                if blocking:
                    with contextlib.suppress(Exception):
                        await identify_task
                else:
                    return
            
            if not await audio_buffer.has_sufficient_audio():
                return
            
            async def _run_identification():
                nonlocal identify_task, interruption_detected
                try:
                    audio_samples = await audio_buffer.get_audio_as_numpy(
                        window_seconds=self._config["identify_window_seconds"]
                    )
                    if len(audio_samples) == 0:
                        return
                    
                    result = await speaker_id_manager.identify_speaker(
                        audio_samples,
                        force_reidentify=force,
                    )
                    
                    if result:
                        # Update speaker tracker
                        if result.switched:
                            speaker_tracker.switch_speaker(
                                result.speaker,
                                result.user_id,
                                timestamp=loop.time(),
                                confidence=result.confidence_score,
                                reason=result.switch_reason,
                            )
                            interruption_detected = False  # Reset after switch
                        
                        # Emit speaker identification event
                        payload: Dict[str, Any] = {
                            "type": "speaker_identified",
                            "speaker": result.speaker,
                        }
                        if result.user_id:
                            payload["user_id"] = result.user_id
                        if result.confidence_level:
                            payload["confidence"] = result.confidence_level
                        if result.confidence_score:
                            payload["confidence_score"] = result.confidence_score
                        if result.forced:
                            payload["forced"] = True
                        if result.margin:
                            payload["margin"] = result.margin
                        
                        try:
                            result_queue.put_nowait(payload)
                        except asyncio.QueueFull:
                            try:
                                result_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            result_queue.put_nowait(payload)
                        
                        # Trim history after identification
                        await audio_buffer.trim_history()
                
                except Exception as exc:
                    logger.warning(f"Identification task failed: {exc}")
                finally:
                    identify_task = None
            
            identify_task = asyncio.create_task(_run_identification())
            if blocking:
                with contextlib.suppress(Exception):
                    await identify_task
        
        async def audio_chunk_stream() -> AsyncGenerator[bytes, None]:
            """Stream audio chunks from buffer manager."""
            async for chunk in audio_buffer.process_audio_stream(
                audio_queue,
                on_interruption_detected=on_interruption_detected,
                on_sufficient_audio=on_sufficient_audio,
            ):
                yield chunk
        
        async def process_azure_events(azure_events: AsyncGenerator) -> None:
            """Process Azure recognition events."""
            result_count = 0
            try:
                async for event in azure_events:
                    event_type = event.get("type")
                    
                    # Get current speaker info
                    if speaker_id_manager:
                        current_speaker = speaker_id_manager.current_speaker
                        current_user_id = speaker_id_manager.current_user_id
                    else:
                        current_speaker = normalize_speaker_label(speaker_label)
                        current_user_id = None
                    
                    if event_type == "result":
                        # Schedule identification on final results
                        await schedule_identification(force=True, blocking=True)
                        # Re-fetch speaker info AFTER identification completes
                        if speaker_id_manager:
                            current_speaker = speaker_id_manager.current_speaker
                            current_user_id = speaker_id_manager.current_user_id
                        result_count += 1
                        if result_count % 10 == 0:
                            logger.debug(f"[Performance] Active tasks: {len(background_tasks)}, Results: {result_count}")
                    elif event_type == "partial":
                        # Retry identification if speaker unknown
                        if current_speaker == "Khách" or current_user_id is None:
                            self._create_tracked_task(
                                background_tasks,
                                schedule_identification(force=True)
                            )
                    
                    event["speaker"] = current_speaker
                    if current_user_id:
                        event["user_id"] = current_user_id
                    
                    if event.get("text"):
                        raw_text = event["text"]
                        
                        if event_type == "result":
                            filtered_text = normalize_vietnamese_text(raw_text)
                            
                            if not filtered_text.strip():
                                event["type"] = "noise"
                                event["text"] = ""
                                event["text_raw"] = raw_text
                            else:
                                event["text"] = filtered_text
                                event["text_raw"] = raw_text
                                
                                azure_conf = event.get("confidence")
                                event["confidence_adjusted"] = calculate_speech_confidence(filtered_text, azure_conf)
                                
                                speaker_tracker.append_text(filtered_text)
                        else:
                            event["text"] = normalize_vietnamese_text(raw_text)
                        
                        display_plain = f"{current_speaker}: {event['text']}"
                        event["display"] = display_plain
                        event["display_colored"] = self._colorize(event_type or "", display_plain)
                        
                        # Cache results
                        if event_type == "result" and self.redis_service:
                            cache_key = f"recognition:session:{session_id}:result:{result_count}"
                            self._create_tracked_task(
                                background_tasks,
                                self.redis_service.set(cache_key, {
                                    "text": event.get("text"),
                                    "text_raw": raw_text,
                                    "speaker": current_speaker,
                                    "user_id": current_user_id,
                                    "timestamp": loop.time(),
                                    "confidence": event.get("confidence_adjusted"),
                                }, ttl=3600)
                            )
                    
                    try:
                        result_queue.put_nowait(event)
                    except asyncio.QueueFull:
                        try:
                            result_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        result_queue.put_nowait(event)
            
            except Exception as exc:
                logger.error(f"Azure event processing failed: {exc}")
            finally:
                result_queue.put_nowait(None)
        
        # Start processing
        audio_stream = audio_chunk_stream()
        azure_task = asyncio.create_task(
            process_azure_events(
                self.speech_repository.recognize_stream(
                    audio_stream,
                    extra_phrases=extra_phrases,
                )
            )
        )
        
        try:
            while True:
                event = await result_queue.get()
                if event is None:
                    break
                yield event
        
        finally:
            # Finalize tracking
            final_timestamp = loop.time()
            segments = speaker_tracker.finalize(final_timestamp)
            session_summary = speaker_tracker.get_session_summary(final_timestamp)
            
            logger.info(
                f"📊 Session Summary | speakers={session_summary['speaker_count']} | "
                f"segments={session_summary['total_segments']} | "
                f"active={session_summary['active_speaker']}"
            )
            
            for idx, speaker_info in enumerate(session_summary.get('speakers', [])[:3], 1):
                logger.info(
                    f"   #{idx} {speaker_info['speaker']}: "
                    f"{speaker_info['total_duration']:.1f}s "
                    f"({speaker_info['segment_count']} segments)"
                )
            
            # Cleanup
            if not azure_task.done():
                azure_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await azure_task
            
            with contextlib.suppress(Exception):
                await audio_stream.aclose()
            
            if identify_task and not identify_task.done():
                identify_task.cancel()
                with contextlib.suppress(Exception):
                    await identify_task
            
            for task in background_tasks:
                if not task.done():
                    task.cancel()
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
                background_tasks.clear()
            
            # Clear voice profile cache
            if defense_session_id and self.voice_service:
                try:
                    self.voice_service.clear_session_cache(defense_session_id)
                    logger.info(f"🧹 Cleared voice profile cache for session {defense_session_id}")
                except Exception as e:
                    logger.debug(f"Failed to clear session cache: {e}")
            
            logger.info("Recognition stream ended")
