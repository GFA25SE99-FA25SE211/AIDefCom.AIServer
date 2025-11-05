"""Speech Service - Business logic for speech-to-text streaming."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import AsyncGenerator, Dict, Any, Iterable, Optional, TYPE_CHECKING

import numpy as np

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from repositories.azure_speech_repository import AzureSpeechRepository
from services.audio_processing.audio_utils import NoiseFilter, pcm_to_wav
from services.redis_service import get_redis_service

if TYPE_CHECKING:
    from services.voice_service import VoiceService

logger = logging.getLogger(__name__)

# Constants
IDENTIFY_MIN_SECONDS = 1.2  # Min audio length for identification
IDENTIFY_WINDOW_SECONDS = 2.5  # Window of audio to analyze
HISTORY_SECONDS = 4.0  # Keep 4s of audio history
IDENTIFY_INTERVAL_SECONDS = 0.6  # Check every 0.6s for speaker changes


def _normalize_speaker_label(name: str | None) -> str:
    """Normalize speaker label."""
    if not name:
        return "Khách"
    normalized = name.strip()
    lowered = normalized.lower()
    if lowered in {"guest", "unknown", "khach", "khách", "dang xac dinh", "đang xác định"}:
        return "Khách"
    return normalized


class SpeechService:
    """Speech-to-text service with speaker identification."""
    
    def __init__(
        self,
        azure_speech_repo: AzureSpeechRepository,
        voice_service: Optional["VoiceService"] = None,
    ) -> None:
        """
        Initialize speech service.
        
        Args:
            azure_speech_repo: Repository for Azure Speech operations
            voice_service: Optional voice authentication service
        """
        self.azure_speech_repo = azure_speech_repo
        self.voice_service = voice_service
        self.noise_filter = NoiseFilter()
        self.sample_rate = azure_speech_repo.sample_rate
        self.redis_service = get_redis_service()
        
        logger.info("Speech Service initialized with Redis caching")
    
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
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream speech recognition from audio queue (instead of WebSocket).
        
        Args:
            audio_queue: Queue receiving audio bytes (None signals end)
            speaker_label: Initial speaker label
            extra_phrases: Additional phrase hints
            apply_noise_filter: Whether to apply noise filtering
        
        Yields:
            Recognition events
        """
        logger.info(f"🎙️  Starting recognition stream (speaker={speaker_label}, filter={apply_noise_filter})")
        loop = asyncio.get_running_loop()

        current_speaker = _normalize_speaker_label(speaker_label)
        current_user_id: Optional[str] = None
        
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
        
        # Track background tasks to prevent memory leak
        background_tasks: set[asyncio.Task] = set()

        async def resolve_speaker_from_history() -> None:
            """Identify speaker using the buffered audio history."""
            nonlocal current_speaker, current_user_id
            nonlocal pending_candidate, pending_user_id, pending_hits, pending_top_score
            nonlocal current_confidence_score, last_reinforce_ts, last_switch_ts

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
            
            # Check cache first - use xxhash for speed
            try:
                import xxhash
                audio_hash = xxhash.xxh64(wav_bytes[:2000]).hexdigest()[:12]
            except ImportError:
                import hashlib
                audio_hash = hashlib.md5(wav_bytes[:2000]).hexdigest()[:12]
            
            cache_key = f"speaker:id:{audio_hash}"
            
            cached_result = await self.redis_service.get(cache_key)
            if cached_result:
                logger.debug(f"✅ Cache hit: {audio_hash}")
                result = cached_result
            else:
                try:
                    result = await loop.run_in_executor(
                        None,
                        self.voice_service.identify_speaker,
                        wav_bytes,
                    )
                    # Cache 2 min (giảm từ 5 min)
                    await self.redis_service.set(cache_key, result, ttl=120)
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

            # Speaker lock decay to avoid sticking forever
            now_ts = loop.time()
            DECAY_SECONDS = 2.8
            if last_reinforce_ts and (now_ts - last_reinforce_ts) > DECAY_SECONDS:
                if current_speaker != "Khách":
                    logger.info(f"⌛ Speaker lock decayed after {now_ts - last_reinforce_ts:.2f}s; relaxing to Đang xác định")
                current_speaker = "Khách"
                current_user_id = None
                current_confidence_score = 0.0
                pending_candidate = None
                pending_user_id = None
                pending_hits = 0
                pending_top_score = 0.0

            # If same speaker, only short-circuit when confidence is strong
            if new_name == current_speaker and (new_user or None) == current_user_id:
                if (margin is not None and margin >= 0.03) or (score_for_decision >= cosine_threshold + 0.02):
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
            high_score_required = cosine_threshold + 0.06
            medium_score_required = cosine_threshold + 0.02
            baseline_required = cosine_threshold - 0.02  # Balanced for accuracy
            margin_strong = margin is not None and margin >= 0.05
            margin_ok = margin is not None and margin >= 0.03
            margin_weak = margin is not None and margin >= 0.015  # Lowered from 0.025
            high_score = score_for_decision >= high_score_required
            medium_score = score_for_decision >= medium_score_required
            baseline_score = score_for_decision >= baseline_required

            should_switch = False
            switch_reason = ""

            # Immediate switch rule for strong evidence of a different speaker
            if new_name != current_speaker:
                if score_for_decision >= (cosine_threshold + 0.03) and (margin is not None and margin >= 0.04):
                    should_switch = True
                    switch_reason = "immediate_strong_evidence"
                elif pending_candidate == new_name:
                    # Evidence-based switching after repeated hits
                    if pending_hits + 1 >= 2 and (baseline_score and (margin is None or margin >= 0.02)):
                        should_switch = True
                        switch_reason = "two_hits_with_baseline"
            
            # Improved logic: balance accuracy and recognition rate
            if explicit_identified and high_score:
                should_switch = True
                switch_reason = switch_reason or "explicit_high"
            elif explicit_identified and medium_score and margin_strong:
                should_switch = True
                switch_reason = switch_reason or "explicit_medium_strong_margin"
            elif explicit_identified and baseline_score and margin_strong:
                should_switch = True
                switch_reason = switch_reason or "explicit_baseline_strong_margin"
            elif score_for_decision >= cosine_threshold and margin_ok:
                should_switch = True
                switch_reason = switch_reason or "threshold_margin"
            elif forced and score_for_decision >= baseline_required and margin_strong:
                should_switch = True
                switch_reason = switch_reason or "forced_baseline_strong_margin"
            else:
                if pending_candidate == new_name and pending_user_id == new_user:
                    pending_hits += 1
                    pending_top_score = max(pending_top_score, score_for_decision)
                else:
                    pending_candidate = new_name
                    pending_user_id = new_user
                    pending_hits = 1
                    pending_top_score = score_for_decision

                # margin_str = f"{margin:.3f}" if margin is not None else "N/A"
                # logger.debug(f"📊 Pending: {new_name} hits={pending_hits} score={pending_top_score:.3f} margin={margin_str}")

                if pending_hits >= 2 and baseline_score and margin_weak:
                    should_switch = True
                    switch_reason = switch_reason or "two_hits_baseline_margin"
                elif pending_hits >= 3 and baseline_score:
                    should_switch = True
                    switch_reason = switch_reason or "three_hits_baseline"
                else:
                    logger.debug(f"⏳ Waiting for more evidence: {new_name} (need {3-pending_hits} more hits or better score/margin)")
                    return

            current_speaker = new_name
            current_user_id = new_user
            pending_candidate = None
            pending_user_id = None
            pending_hits = 0
            pending_top_score = 0.0
            # Update reinforcement timers
            last_reinforce_ts = now_ts
            last_switch_ts = now_ts
            current_confidence_score = score_for_decision
            logger.info(f"✅ Switch speaker → {current_speaker} (reason={switch_reason}, score={score_for_decision:.3f}{', margin='+format(margin, '.3f') if margin is not None else ''})")

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
            nonlocal frame_buffer

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
                                force_identify = (current_user_id is None or current_speaker == "Khách")
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
                        display_plain = f"{current_speaker}: {event['text']}"
                        event["display"] = display_plain
                        event["display_colored"] = self._colorize(event_type or "", display_plain)
                        
                        # Cache recognition results in Redis
                        if event_type == "result":
                            cache_key = f"recognition:session:{session_id}:result:{result_count}"
                            self._create_tracked_task(background_tasks, self.redis_service.set(cache_key, {
                                "text": event.get("text"),
                                "speaker": current_speaker,
                                "user_id": current_user_id,
                                "timestamp": loop.time(),
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
