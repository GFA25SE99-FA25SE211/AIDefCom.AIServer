"""Speech Service - Business logic for speech-to-text streaming."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import AsyncGenerator, Dict, Any, Iterable, Optional, TYPE_CHECKING, List

import numpy as np

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from repositories.interfaces.i_speech_repository import ISpeechRepository
from services.interfaces.i_speech_service import ISpeechService
from services.interfaces.i_voice_service import IVoiceService
from services.interfaces.i_question_service import IQuestionService
from services.audio_processing.audio_utils import (
    NoiseFilter,
    pcm_to_wav,
    detect_energy_spike,
    detect_acoustic_change,
    calculate_rms,
)
from services.audio_processing.speech_utils import (
    filter_filler_words,
    normalize_vietnamese_text,
    should_log_transcript,
    calculate_speech_confidence,
)
from services.multi_speaker_tracker import MultiSpeakerTracker
from services.session_room_manager import get_session_room_manager
from repositories.interfaces.i_redis_service import IRedisService

logger = logging.getLogger(__name__)

# Constants - Optimized for faster response on production
IDENTIFY_MIN_SECONDS = 0.8  # Reduced from 1.2s for faster initial identification
IDENTIFY_WINDOW_SECONDS = 2.0  # Reduced from 2.5s for quicker speaker changes
HISTORY_SECONDS = 3.0  # Reduced from 4s - less memory, faster processing
IDENTIFY_INTERVAL_SECONDS = 0.4  # Reduced from 0.5s for more responsive updates

# Redis operation timeout (don't let cache slow down recognition)
REDIS_TIMEOUT_SECONDS = 0.5  # Max 500ms for any Redis operation

# Dedicated thread pool for CPU-bound voice identification (avoid blocking event loop)
import concurrent.futures
_VOICE_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None

def _get_voice_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create dedicated executor for voice identification."""
    global _VOICE_EXECUTOR
    if _VOICE_EXECUTOR is None:
        # Use 2-4 workers depending on CPU cores (voice ID is CPU-intensive)
        import os
        max_workers = min(4, max(2, (os.cpu_count() or 2)))
        _VOICE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="voice_id_"
        )
        logger.info(f"Created voice identification executor with {max_workers} workers")
    return _VOICE_EXECUTOR


def _normalize_speaker_label(name: str | None) -> str:
    """Normalize speaker label."""
    if not name:
        return "Khách"
    normalized = name.strip()
    lowered = normalized.lower()
    if lowered in {"guest", "unknown", "khach", "khách", "dang xac dinh", "đang xác định"}:
        return "Khách"
    return normalized


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
        
        logger.info("Speech Service initialized (Redis caching, QuestionService=%s)", bool(self.question_service))
        # Question mode state (per session_id)
        self._question_mode: Dict[str, bool] = {}
        self._question_buffer: Dict[str, List[str]] = {}
        self._question_last_final: Dict[str, Optional[str]] = {}

    def get_defense_session_users(self, session_id: str) -> Optional[List[str]]:
        """Return list of user IDs enrolled in a defense session (delegates to voice service).
        If voice service not available or session invalid, returns None.
        """
        if not self.voice_service:
            return None
        try:
            return self.voice_service.get_defense_session_users(session_id)
        except Exception:
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

    async def recognize_stream_from_queue(
        self,
        audio_queue: asyncio.Queue[bytes | None],
        speaker_label: str = "Đang xác định",
        extra_phrases: Iterable[str] | None = None,
        apply_noise_filter: bool = True,
        whitelist_user_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream speech recognition from audio queue (instead of WebSocket).
        
        Args:
            audio_queue: Queue receiving audio bytes (None signals end)
            speaker_label: Initial speaker label
            extra_phrases: Additional phrase hints
            apply_noise_filter: Whether to apply noise filtering
            whitelist_user_ids: Optional list of user IDs to filter identification (defense session)
        
        Yields:
            Recognition events
        """
        logger.info(f"🎙️  Starting recognition stream (speaker={speaker_label}, filter={apply_noise_filter})")
        loop = asyncio.get_running_loop()

        current_speaker = _normalize_speaker_label(speaker_label)
        current_user_id: Optional[str] = None
        
        # NEW: Multi-speaker tracker
        speaker_tracker = MultiSpeakerTracker(max_speakers=4, inactivity_timeout=30.0)
        speaker_tracker.switch_speaker(
            current_speaker,
            current_user_id,
            timestamp=loop.time(),
            reason="initial"
        )
        
        # Generate session ID for caching
        import hashlib
        import time
        session_id = hashlib.md5(f"{time.time()}_{id(audio_queue)}".encode()).hexdigest()[:16]

        audio_history = bytearray()
        history_lock = asyncio.Lock()
        history_limit_bytes = int(self.sample_rate * HISTORY_SECONDS) * 2
        min_identify_bytes = int(self.sample_rate * IDENTIFY_MIN_SECONDS) * 2

        identify_task: Optional[asyncio.Task] = None
        last_identify_ts = 0.0
        identify_interval = max(IDENTIFY_INTERVAL_SECONDS, 0.4)
        
        # NEW: Sliding window verification
        verify_task: Optional[asyncio.Task] = None
        last_verify_ts = 0.0
        verify_interval = 0.5  # Re-verify every 500ms

        FRAME_SAMPLES = 320  # 20ms @ 16kHz
        FRAME_BYTES = FRAME_SAMPLES * 2
        frame_buffer = bytearray()

        result_queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue(maxsize=8)
        pending_candidate: Optional[str] = None
        pending_user_id: Optional[str] = None
        pending_hits = 0
        pending_top_score = 0.0
        # Track stability/decay of current speaker lock
        current_confidence_score: float = 0.0
        last_reinforce_ts: float = 0.0
        last_switch_ts: float = 0.0
        
        # NEW: Interruption detection state
        previous_audio_chunk: Optional[bytes] = None
        interruption_detected: bool = False
        
        # Track background tasks to prevent memory leak
        background_tasks: set[asyncio.Task] = set()
        
        # NEW: Lock for speaker state updates (prevent race conditions)
        speaker_state_lock = asyncio.Lock()

        async def resolve_speaker_from_history(force_reidentify: bool = False) -> None:
            """Identify speaker using the buffered audio history."""
            nonlocal current_speaker, current_user_id
            nonlocal pending_candidate, pending_user_id, pending_hits, pending_top_score
            nonlocal current_confidence_score, last_reinforce_ts, last_switch_ts
            nonlocal interruption_detected

            if self.voice_service is None:
                return

            async with history_lock:
                pcm_snapshot = bytes(audio_history)

            if len(pcm_snapshot) < min_identify_bytes:
                return

            # Chỉ lấy đoạn cuối để tăng tốc nhận diện
            samples = np.frombuffer(pcm_snapshot, dtype=np.int16)
            window_samples = int(self.sample_rate * IDENTIFY_WINDOW_SECONDS)
            if samples.size > window_samples:
                samples = samples[-window_samples:]
            wav_bytes = pcm_to_wav(samples.tobytes(), sample_rate=self.sample_rate)
            
            # Check cache first - hash full audio to avoid collisions
            try:
                import xxhash
                # Hash full audio or stratified samples for better uniqueness
                if len(wav_bytes) <= 8000:
                    audio_hash = xxhash.xxh64(wav_bytes).hexdigest()[:16]
                else:
                    # Stratified sampling: start, middle, end
                    samples = [
                        wav_bytes[:2000],
                        wav_bytes[len(wav_bytes)//2:len(wav_bytes)//2+2000],
                        wav_bytes[-2000:]
                    ]
                    audio_hash = xxhash.xxh64(b''.join(samples)).hexdigest()[:16]
            except ImportError:
                import hashlib
                audio_hash = hashlib.md5(wav_bytes).hexdigest()[:16]
            
            cache_key = f"speaker:id:{audio_hash}"
            
            # Redis get with timeout (don't block recognition)
            cached_result = None
            try:
                cached_result = await asyncio.wait_for(
                    self.redis_service.get(cache_key),
                    timeout=REDIS_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.debug(f"Redis get timeout for {cache_key}")
            except Exception as e:
                logger.debug(f"Redis get error: {e}")
            
            if cached_result:
                logger.debug(f"✅ Cache hit: {audio_hash}")
                result = cached_result
            else:
                try:
                    # Use dedicated executor to avoid blocking event loop
                    executor = _get_voice_executor()
                    result = await loop.run_in_executor(
                        executor,
                        self.voice_service.identify_speaker,
                        wav_bytes,
                        whitelist_user_ids,
                    )
                    # Cache result (fire-and-forget, don't block)
                    asyncio.create_task(
                        asyncio.wait_for(
                            self.redis_service.set(cache_key, result, ttl=90),
                            timeout=REDIS_TIMEOUT_SECONDS
                        )
                    )
                except Exception as exc:
                    logger.warning(f"Voice identification failed: {exc}")
                    return

            forced = False
            confidence_score: Optional[float] = None
            confidence_level: Optional[str] = None
            new_user = current_user_id
            new_name = current_speaker
            margin: Optional[float] = None

            candidates = result.get("candidates") or []
            
            # Debug logging for identification
            logger.info(f"🔍 Identify result: identified={result.get('identified')} | candidates={len(candidates)}")
            if candidates:
                top = candidates[0]
                logger.info(f"   Top: {top.get('name')} | cosine={top.get('cosine'):.3f}")

            if result.get("identified"):
                new_name = _normalize_speaker_label(result.get("speaker")) or current_speaker
                new_user = result.get("user_id") or new_user
                score_val = result.get("score")
                if isinstance(score_val, (int, float)):
                    confidence_score = float(score_val)
                confidence_level = result.get("confidence") or self._score_to_confidence(confidence_score)
                if len(candidates) > 1 and confidence_score is not None:
                    second_score_raw = candidates[1].get("cosine")
                    if isinstance(second_score_raw, (int, float)):
                        margin = confidence_score - float(second_score_raw)
            else:
                if candidates:
                    top_candidate = candidates[0]
                    new_name = _normalize_speaker_label(top_candidate.get("name"))
                    new_user = top_candidate.get("user_id") or new_user
                    score_val = top_candidate.get("cosine")
                    if isinstance(score_val, (int, float)):
                        confidence_score = float(score_val)
                    confidence_level = self._score_to_confidence(confidence_score)
                    if len(candidates) > 1:
                        second_score_raw = candidates[1].get("cosine")
                        if isinstance(second_score_raw, (int, float)):
                            margin = (confidence_score or 0.0) - float(second_score_raw)
                    forced = True
                    logger.info(f"   ⚠️  Using top candidate (forced): {new_name} | score={confidence_score:.3f}")
                else:
                    new_name = "Khách"
                    new_user = None
                    logger.info(f"   ❌ No candidates, defaulting to Khách")

            if not new_name:
                new_name = "Khách"

            if new_name == "Khách":
                new_user = None

            score_for_decision = confidence_score or 0.0
            cosine_threshold = getattr(self.voice_service, "cosine_threshold", 0.75)
            # NEW: Adaptive margin and hits for small groups
            base_margin = getattr(self.voice_service, "speaker_switch_margin", 0.06)
            base_hits = getattr(self.voice_service, "speaker_switch_hits_required", 3)
            
            # Lower requirements for small defense sessions (easier to distinguish)
            if whitelist_user_ids and len(whitelist_user_ids) <= 6:
                required_margin = 0.04  # 4% for small groups
                required_hits = 2
            else:
                required_margin = base_margin
                required_hits = base_hits

            # Speaker lock decay - dynamic based on confidence (3-5s)
            now_ts = loop.time()
            # Higher confidence = longer lock, lower confidence = faster decay
            base_decay = 3.0
            confidence_bonus = min(current_confidence_score * 2.0, 2.0) if current_confidence_score else 0.0
            DECAY_SECONDS = base_decay + confidence_bonus  # 3.0-5.0s
            
            async with speaker_state_lock:
                if last_reinforce_ts and (now_ts - last_reinforce_ts) > DECAY_SECONDS:
                    if current_speaker != "Khách":
                        logger.info(f"⌛ Speaker lock decayed after {now_ts - last_reinforce_ts:.2f}s; relaxing to Khách")
                    current_speaker = "Khách"
                    current_user_id = None
                    current_confidence_score = 0.0
                    pending_candidate = None
                    pending_user_id = None
                    pending_hits = 0
                    pending_top_score = 0.0

                # If same speaker, only short-circuit when confidence is strong
                if new_name == current_speaker and (new_user or None) == current_user_id:
                    # Reinforcement margin increased from 0.03 to 0.05
                    if (margin is not None and margin >= 0.05) or (score_for_decision >= cosine_threshold + 0.03):
                        # Reinforce current speaker
                        last_reinforce_ts = now_ts
                        current_confidence_score = score_for_decision
                        pending_candidate = None
                        pending_user_id = None
                        pending_hits = 0
                        pending_top_score = 0.0
                        return
                    # Otherwise fall through to allow potential switch if evidence builds

            explicit_identified = bool(result.get("identified"))
            
            margin_str = f"{margin:.3f}" if margin is not None else "N/A"
            logger.info(f"🎯 Decision params: name={new_name} | score={score_for_decision:.3f} | threshold={cosine_threshold:.3f} | identified={explicit_identified} | forced={forced} | margin={margin_str}")
            # NEW: Stricter thresholds for better accuracy
            high_score_required = cosine_threshold + 0.08  # Increased from +0.06
            medium_score_required = cosine_threshold + 0.04  # Increased from +0.02
            baseline_required = cosine_threshold  # Changed from -0.02 to exact threshold
            
            # NEW: Margin requirements (stronger than before)
            margin_strong = margin is not None and margin >= required_margin  # 0.06
            margin_ok = margin is not None and margin >= (required_margin * 0.67)  # 0.04
            margin_weak = margin is not None and margin >= (required_margin * 0.5)  # 0.03
            
            high_score = score_for_decision >= high_score_required
            medium_score = score_for_decision >= medium_score_required
            baseline_score = score_for_decision >= baseline_required

            should_switch = False
            switch_reason = ""

            # NEW: More conservative switching rules to prevent false positives
            if new_name != current_speaker:
                # Rule 1: Immediate switch ONLY with very strong evidence
                if score_for_decision >= (cosine_threshold + 0.05) and margin_strong:
                    should_switch = True
                    switch_reason = "immediate_strong_evidence"
                elif pending_candidate == new_name:
                    # Rule 2: Require MORE hits with strong margin
                    if pending_hits + 1 >= required_hits and baseline_score and margin_ok:
                        should_switch = True
                        switch_reason = f"{required_hits}_hits_with_baseline_margin"
            
            # Enhanced logic: prioritize high confidence + margin
            if not should_switch:
                if explicit_identified and high_score and margin_strong:
                    should_switch = True
                    switch_reason = switch_reason or "explicit_high_strong_margin"
                elif explicit_identified and medium_score and margin_strong:
                    should_switch = True
                    switch_reason = switch_reason or "explicit_medium_strong_margin"
                elif score_for_decision >= (cosine_threshold + 0.02) and margin_strong:
                    should_switch = True
                    switch_reason = switch_reason or "threshold_strong_margin"
                elif forced and score_for_decision >= baseline_required and margin_strong:
                    should_switch = True
                    switch_reason = switch_reason or "forced_baseline_strong_margin"
                else:
                    # Accumulate evidence (with lock to prevent race)
                    async with speaker_state_lock:
                        if pending_candidate == new_name and pending_user_id == new_user:
                            pending_hits += 1
                            pending_top_score = max(pending_top_score, score_for_decision)
                        else:
                            pending_candidate = new_name
                            pending_user_id = new_user
                            pending_hits = 1
                            pending_top_score = score_for_decision

                        margin_str = f"{margin:.3f}" if margin is not None else "N/A"
                        logger.debug(f"📊 Pending: {new_name} hits={pending_hits}/{required_hits} score={pending_top_score:.3f} margin={margin_str}")

                        # Rule 3: Require full hit count with good margin
                        if pending_hits >= required_hits and baseline_score and margin_ok:
                            should_switch = True
                            switch_reason = switch_reason or f"{required_hits}_hits_baseline_margin"
                        elif pending_hits >= (required_hits + 1) and baseline_score:
                            # Extra hit if margin is weak
                            should_switch = True
                            switch_reason = switch_reason or f"{required_hits + 1}_hits_baseline"
                        else:
                            logger.debug(f"⏳ Waiting: {new_name} needs {required_hits - pending_hits} more hits or better score/margin")
                            return

            # NEW: Update multi-speaker tracker with lock
            now_ts = loop.time()
            
            async with speaker_state_lock:
                switched = speaker_tracker.switch_speaker(
                    new_speaker=new_name,
                    new_user_id=new_user,
                    timestamp=now_ts,
                    confidence=score_for_decision,
                    reason=switch_reason
                )
                
                if switched or force_reidentify:
                    current_speaker = new_name
                    current_user_id = new_user
                    pending_candidate = None
                    pending_user_id = None
                    pending_hits = 0
                    pending_top_score = 0.0
                    interruption_detected = False  # Reset after switch
                    # Update reinforcement timers
                    last_reinforce_ts = now_ts
                    last_switch_ts = now_ts
                    current_confidence_score = score_for_decision
                    
                    logger.info(f"✅ Switch speaker → {current_speaker} (reason={switch_reason}, score={score_for_decision:.3f}{', margin='+format(margin, '.3f') if margin is not None else ''})")
                else:
                    # Same speaker, just update state
                    last_reinforce_ts = now_ts
                    current_confidence_score = score_for_decision

            async with history_lock:
                if len(audio_history) > history_limit_bytes * 1.5:
                    del audio_history[: history_limit_bytes // 2]

            payload: Dict[str, Any] = {"type": "speaker_identified", "speaker": current_speaker}
            if current_user_id:
                payload["user_id"] = current_user_id
            if confidence_level is not None:
                payload["confidence"] = confidence_level
            if confidence_score is not None:
                payload["confidence_score"] = confidence_score
            if forced:
                payload["forced"] = True
            if margin is not None:
                payload["margin"] = margin

            try:
                result_queue.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    _ = result_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                result_queue.put_nowait(payload)

        async def schedule_identification(force: bool = False, *, blocking: bool = False) -> None:
            """Schedule speaker identification tasks with throttling."""
            nonlocal identify_task, last_identify_ts

            if self.voice_service is None:
                return

            if identify_task and not identify_task.done():
                if force and blocking:
                    with contextlib.suppress(Exception):
                        await identify_task
                    identify_task = None
                else:
                    return

            async with history_lock:
                history_len = len(audio_history)

            if history_len < min_identify_bytes:
                return

            now = loop.time()
            if not force and (now - last_identify_ts) < identify_interval:
                return

            last_identify_ts = now

            async def _run_identification() -> None:
                nonlocal identify_task
                try:
                    await resolve_speaker_from_history()
                except Exception as exc:
                    logger.warning(f"Identification task failed: {exc}")
                finally:
                    identify_task = None

            identify_task = asyncio.create_task(_run_identification())
            if blocking:
                with contextlib.suppress(Exception):
                    await identify_task
        
        # Use class-level tracked task helper to avoid duplicating logic
        # background_tasks is captured from outer scope

        async def audio_chunk_stream() -> AsyncGenerator[bytes, None]:
            """Read audio from queue, buffer for identification, and yield for Azure."""
            nonlocal frame_buffer, previous_audio_chunk, interruption_detected

            try:
                while True:
                    audio_data = await audio_queue.get()
                    
                    # None signals end of stream
                    if audio_data is None:
                        break

                    if not audio_data:
                        continue

                    if len(audio_data) > 8192:
                        audio_data = audio_data[:8192]

                    frame_buffer.extend(audio_data)

                    while len(frame_buffer) >= FRAME_BYTES:
                        raw_frame = bytes(frame_buffer[:FRAME_BYTES])
                        del frame_buffer[:FRAME_BYTES]

                        chunk = self.noise_filter.reduce_noise(raw_frame) if apply_noise_filter else raw_frame

                        # NEW: Interruption detection
                        if previous_audio_chunk and self.voice_service and not interruption_detected:
                            # Check for energy spike (possible interruption)
                            if detect_energy_spike(chunk, previous_audio_chunk, spike_threshold=1.5):
                                logger.info("⚡ Energy spike detected - possible speaker interruption")
                                # Force immediate re-identification
                                self._create_tracked_task(
                                    background_tasks,
                                    schedule_identification(force=True, blocking=False)
                                )
                                interruption_detected = True
                                # Flag will be reset after speaker switch (in resolve_speaker_from_history)
                            
                            # Check for acoustic change (pitch/timbre shift)
                            elif detect_acoustic_change(chunk, previous_audio_chunk, change_threshold=0.3):
                                logger.debug("🎵 Acoustic change detected")
                                self._create_tracked_task(
                                    background_tasks,
                                    schedule_identification(force=True, blocking=False)
                                )
                                interruption_detected = True
                                # Flag will be reset after speaker switch (in resolve_speaker_from_history)
                        
                        previous_audio_chunk = chunk

                        async with history_lock:
                            audio_history.extend(chunk)
                            if len(audio_history) > history_limit_bytes:
                                overflow = len(audio_history) - history_limit_bytes
                                del audio_history[:overflow]
                            history_len = len(audio_history)

                            # Always schedule identification if we have enough audio
                            # This ensures we detect speaker changes quickly
                            if history_len >= min_identify_bytes:
                                # Force more often if:
                                # 1. No speaker identified yet
                                # 2. Current speaker is "Khách" (uncertain)
                                # 3. Interruption detected
                                force_identify = (
                                    current_user_id is None or
                                    current_speaker == "Khách" or
                                    interruption_detected
                                )
                                self._create_tracked_task(background_tasks, schedule_identification(force=force_identify, blocking=False))
                            else:
                                # Not enough audio yet, just schedule with throttle
                                self._create_tracked_task(background_tasks, schedule_identification(force=False, blocking=False))

                        yield chunk

                if frame_buffer:
                    remainder = bytes(frame_buffer)
                    frame_buffer.clear()
                    if remainder:
                        chunk = self.noise_filter.reduce_noise(remainder) if apply_noise_filter else remainder
                        async with history_lock:
                            audio_history.extend(chunk)
                            if len(audio_history) > history_limit_bytes:
                                overflow = len(audio_history) - history_limit_bytes
                                del audio_history[:overflow]
                        # Don't block - run in background
                        self._create_tracked_task(background_tasks, schedule_identification(force=True, blocking=False))
                        yield chunk

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"Audio stream interrupted: {exc}")

        # Use class-level _colorize helper

        async def process_azure_events(azure_events: AsyncGenerator) -> None:
            """Merge Azure results with speaker information and queue them for clients."""
            result_count = 0
            try:
                async for event in azure_events:
                    event_type = event.get("type")

                    if event_type == "result":
                        await schedule_identification(force=True, blocking=True)
                        result_count += 1
                        # Log background tasks every 10 results
                        if result_count % 10 == 0:
                            logger.info(f"[Performance] Active tasks: {len(background_tasks)}, Results: {result_count}")
                    elif event_type == "partial":
                        self._create_tracked_task(background_tasks, schedule_identification(force=False, blocking=False))
                    else:
                        self._create_tracked_task(background_tasks, schedule_identification(force=False, blocking=False))

                    event["speaker"] = current_speaker
                    if current_user_id:
                        event["user_id"] = current_user_id

                    if event.get("text"):
                        raw_text = event["text"]
                        
                        # NEW: Apply filler word filtering for final results
                        if event_type == "result":
                            # Keep original text without filtering filler words
                            filtered_text = raw_text
                            # Normalize Vietnamese text only (keep all words)
                            filtered_text = normalize_vietnamese_text(filtered_text)
                            
                            # Check if should log (skip if completely empty after normalization)
                            if not filtered_text.strip():
                                logger.debug(f"🚫 Filtered empty transcript: '{raw_text}'")
                                # Send as noise event instead of skipping
                                event["type"] = "noise"
                                event["text"] = ""
                                event["text_raw"] = raw_text
                            else:
                                # Update event with normalized text (keep all words including "là", "thì", etc.)
                                event["text"] = filtered_text
                                event["text_raw"] = raw_text  # Keep original for debugging
                                
                                # Calculate confidence
                                azure_conf = event.get("confidence")
                                event["confidence_adjusted"] = calculate_speech_confidence(filtered_text, azure_conf)
                                
                                # NEW: Append to speaker tracker
                                speaker_tracker.append_text(filtered_text)
                        else:
                            # For partials, apply light normalization (no filtering)
                            event["text"] = normalize_vietnamese_text(raw_text)
                        
                        display_plain = f"{current_speaker}: {event['text']}"
                        event["display"] = display_plain
                        event["display_colored"] = self._colorize(event_type or "", display_plain)
                        
                        # Cache recognition results in Redis
                        if event_type == "result":
                            cache_key = f"recognition:session:{session_id}:result:{result_count}"
                            self._create_tracked_task(background_tasks, self.redis_service.set(cache_key, {
                                "text": event.get("text"),
                                "text_raw": raw_text,
                                "speaker": current_speaker,
                                "user_id": current_user_id,
                                "timestamp": loop.time(),
                                "confidence": event.get("confidence_adjusted"),
                            }, ttl=3600))  # Cache for 1 hour

                    try:
                        result_queue.put_nowait(event)
                    except asyncio.QueueFull:
                        try:
                            _ = result_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        result_queue.put_nowait(event)

            except Exception as exc:
                logger.error(f"Azure event processing failed: {exc}")
            finally:
                result_queue.put_nowait(None)

        audio_stream = audio_chunk_stream()
        azure_task = asyncio.create_task(
            process_azure_events(
                self.azure_speech_repo.recognize_stream(
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
            # NEW: Finalize multi-speaker tracking and log summary
            final_timestamp = loop.time()
            segments = speaker_tracker.finalize(final_timestamp)
            session_summary = speaker_tracker.get_session_summary(final_timestamp)
            
            logger.info(
                f"📊 Session Summary | speakers={session_summary['speaker_count']} | "
                f"segments={session_summary['total_segments']} | "
                f"active={session_summary['active_speaker']}"
            )
            
            # Log top speakers by duration
            for idx, speaker_info in enumerate(session_summary.get('speakers', [])[:3], 1):
                logger.info(
                    f"   #{idx} {speaker_info['speaker']}: "
                    f"{speaker_info['total_duration']:.1f}s "
                    f"({speaker_info['segment_count']} segments)"
                )
            
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
            
            # Cancel all background tasks
            for task in background_tasks:
                if not task.done():
                    task.cancel()
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
                background_tasks.clear()

            logger.info("Recognition stream ended")

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
        
        # Debug: Log all query params to verify what frontend is sending
        logger.info(f"📥 WS Query Params: {dict(ws.query_params)} | defense_session_id={defense_session_id}")
        
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
        whitelist_user_ids = None
        whitelist_fetch_task = None
        
        async def _fetch_whitelist_background():
            """Fetch whitelist in background - fire and forget."""
            nonlocal whitelist_user_ids
            if not defense_session_id or not self.voice_service:
                return
            try:
                # Single attempt with short timeout
                result = await asyncio.wait_for(
                    self.voice_service.get_defense_session_users(defense_session_id),
                    timeout=2.0  # Reduced from 3s
                )
                if result:
                    whitelist_user_ids = result
                    logger.info(f"🎯 Whitelist loaded: {len(whitelist_user_ids)} users")
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Whitelist fetch timeout")
            except Exception as e:
                logger.debug(f"Whitelist fetch skipped: {e}")
        
        # Start background fetch immediately (don't await)
        if defense_session_id and self.voice_service:
            whitelist_fetch_task = asyncio.create_task(_fetch_whitelist_background())
        session_start = datetime.utcnow()
        transcript_lines = []
        
        # === LOAD EXISTING TRANSCRIPT FROM CACHE (resume after reload) ===
        if defense_session_id and self.redis_service:
            try:
                existing = await asyncio.wait_for(
                    self.redis_service.get(transcript_cache_key),
                    timeout=2.0
                )
                if existing and isinstance(existing, dict):
                    transcript_lines = existing.get("lines", [])
                    original_start = existing.get("start_time")
                    if original_start:
                        try:
                            session_start = datetime.fromisoformat(original_start)
                        except Exception:
                            pass
                    logger.info(f"📂 Resumed transcript | defense_session_id={defense_session_id} | existing_lines={len(transcript_lines)}")
            except asyncio.TimeoutError:
                logger.debug("Timeout loading existing transcript")
            except Exception as e:
                logger.debug(f"No existing transcript: {e}")
        
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
                    except Exception:
                        break
                    await asyncio.sleep(25)
            except Exception:
                pass
        
        pinger = asyncio.create_task(ping_task())
        
        # WebSocket reader task
        async def websocket_reader() -> None:
            """Read audio chunks and commands from WebSocket."""
            try:
                while not stop_event.is_set():
                    try:
                        if ws.application_state != WebSocketState.CONNECTED:
                            logger.debug("WebSocket disconnected in reader")
                            break
                        
                        data = await ws.receive()
                        
                        if "bytes" in data:
                            # Audio data -> push to queue
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
                                # Begin question capture mode
                                self._question_mode[session_id] = True
                                self._question_buffer[session_id] = []
                                self._question_last_final[session_id] = None
                                try:
                                    await ws.send_json({
                                        "type": "question_mode_started",
                                        "session_id": session_id
                                    })
                                except Exception:
                                    pass
                            elif message == "q:end":
                                # End question mode and process captured text
                                if self._question_mode.get(session_id):
                                    self._question_mode[session_id] = False
                                    buffered_parts = self._question_buffer.get(session_id, [])
                                    question_text = " ".join(buffered_parts).strip()
                                    # Fallback if empty -> use last final
                                    if not question_text:
                                        last_final = self._question_last_final.get(session_id)
                                        if last_final:
                                            question_text = last_final
                                    
                                    logger.info(f"📝 Question mode ended | text='{question_text[:80]}...'")
                                    
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
                                                            except Exception:
                                                                pass
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
                                                except Exception:
                                                    pass
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
                                                except Exception:
                                                    pass
                                        
                                        # Create task and keep reference to prevent GC
                                        task = asyncio.create_task(_check_and_register_bg())
                                        # Store in session-level task tracker if needed
                                        logger.info("🚀 Background task created (fire-and-forget)")
                                    else:
                                        logger.warning("⚠️ QuestionService unavailable or empty text")
                                    
                                # Clear buffer regardless
                                self._question_buffer.pop(session_id, None)
                                self._question_last_final.pop(session_id, None)
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
            logger.info(f"Starting STT session | session_id={session_id}")
            
            # DON'T wait for whitelist - start recognition immediately
            # Whitelist will be used when it becomes available
            
            # Stream recognition results (no blocking waits)
            async for event in self.recognize_stream_from_queue(
                audio_queue,
                speaker_label=speaker_label,
                extra_phrases=extra_phrases,
                whitelist_user_ids=whitelist_user_ids,
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
                        
                        transcript_lines.append({
                            "timestamp": timestamp,
                            "speaker": speaker,
                            "text": text,
                            "user_id": event.get("user_id"),
                        })
                        
                        # === BROADCAST TO ALL CLIENTS IN SAME DEFENSE SESSION ===
                        if defense_session_id:
                            broadcast_event = {
                                "type": "broadcast_transcript",
                                "speaker": speaker,
                                "text": text,
                                "user_id": event.get("user_id"),
                                "timestamp": timestamp,
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
                        
                        # If question mode active, buffer this final text
                        if self._question_mode.get(session_id):
                            self._question_buffer.setdefault(session_id, []).append(text)
                            self._question_last_final[session_id] = text
                        
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

                # Build full_text with speaker names in format: [Speaker]: [Text]
                parts: List[str] = []
                for line in transcript_lines:
                    speaker = line.get("speaker", "Unknown")
                    txt = (line.get("text") or "").strip()
                    if not txt:
                        raw = (line.get("text_raw") or "").strip()
                        if raw:
                            txt = raw
                    if txt:
                        parts.append(f"{speaker}: {txt}")
                full_text = "\n".join(parts).strip()

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
                    elif not defense_session_id or not defense_session_id.isdigit():
                        logger.warning(
                            f"⚠️ Skip API save: defense_session_id not provided or invalid | "
                            f"defense_session_id={defense_session_id} | session_id={session_id}"
                        )
                        logger.info("💡 Transcript cached in Redis only (no database save)")
                    else:
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
