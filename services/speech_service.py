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

if TYPE_CHECKING:
    from services.voice_service import VoiceService

logger = logging.getLogger(__name__)

# Constants
IDENTIFY_MIN_SECONDS = 1.0  # Đủ nhanh để nhận diện nhưng vẫn đảm bảo chất lượng
IDENTIFY_WINDOW_SECONDS = 3.0
HISTORY_SECONDS = 4.0
IDENTIFY_INTERVAL_SECONDS = 0.45  # Kiểm tra thường xuyên để phát hiện chuyển loa


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
        
        logger.info("Speech Service initialized")
    
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
        loop = asyncio.get_running_loop()

        current_speaker = _normalize_speaker_label(speaker_label)
        current_user_id: Optional[str] = None

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

        async def resolve_speaker_from_history() -> None:
            """Identify speaker using the buffered audio history."""
            nonlocal current_speaker, current_user_id
            nonlocal pending_candidate, pending_user_id, pending_hits, pending_top_score

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

            try:
                result = await loop.run_in_executor(
                    None,
                    self.voice_service.identify_speaker,
                    wav_bytes,
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
                else:
                    new_name = "Khách"
                    new_user = None

            if not new_name:
                new_name = "Khách"

            if new_name == "Khách":
                new_user = None

            if new_name == current_speaker and (new_user or None) == current_user_id:
                pending_candidate = None
                pending_user_id = None
                pending_hits = 0
                pending_top_score = 0.0
                return

            score_for_decision = confidence_score or 0.0
            cosine_threshold = getattr(self.voice_service, "cosine_threshold", 0.75)
            high_score_required = cosine_threshold + 0.06
            medium_score_required = cosine_threshold + 0.02
            baseline_required = cosine_threshold + 0.015
            margin_strong = margin is not None and margin >= 0.04
            margin_ok = margin is not None and margin >= 0.025
            explicit_identified = bool(result.get("identified"))
            high_score = score_for_decision >= high_score_required
            medium_score = score_for_decision >= medium_score_required
            baseline_score = score_for_decision >= baseline_required

            should_switch = False

            if forced and (score_for_decision >= cosine_threshold or margin is not None and margin > 0.0):
                should_switch = True
            elif explicit_identified and (high_score or (medium_score and (margin_strong or forced)) or (baseline_score and margin_ok)):
                should_switch = True
            elif high_score and (margin_strong or forced):
                should_switch = True
            else:
                if pending_candidate == new_name and pending_user_id == new_user:
                    pending_hits += 1
                    pending_top_score = max(pending_top_score, score_for_decision)
                else:
                    pending_candidate = new_name
                    pending_user_id = new_user
                    pending_hits = 1
                    pending_top_score = score_for_decision

                if pending_hits >= 2 and (medium_score or margin_ok):
                    should_switch = True
                elif pending_hits >= 3 and baseline_score:
                    should_switch = True
                else:
                    return

            current_speaker = new_name
            current_user_id = new_user
            pending_candidate = None
            pending_user_id = None
            pending_hits = 0
            pending_top_score = 0.0

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

        async def audio_chunk_stream() -> AsyncGenerator[bytes, None]:
            """Read audio from websocket, buffer for identification, and yield for Azure."""
            nonlocal frame_buffer

            try:
                while True:
                    try:
                        audio_data = await websocket.receive_bytes()
                    except WebSocketDisconnect:
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

                        needs_force = (
                            history_len >= min_identify_bytes
                            and (current_user_id is None or current_speaker == "Khách")
                        )

                        if needs_force:
                            blocking_now = current_user_id is None
                            asyncio.create_task(
                                schedule_identification(force=True, blocking=blocking_now)
                            )
                        else:
                            asyncio.create_task(
                                schedule_identification(force=False, blocking=False)
                            )

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
                        asyncio.create_task(
                            schedule_identification(force=True, blocking=current_user_id is None)
                        )
                        yield chunk

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"Audio stream interrupted: {exc}")

        def _colorize(event_type: str, text: str) -> str:
            color_map = {
                "partial": "#3498db",
                "result": "#2ecc71",
                "nomatch": "#e67e22",
                "error": "#e74c3c",
            }
            color = color_map.get(event_type, "#bdc3c7")
            return f"<span style=\"color:{color}\">{text}</span>"

        async def process_azure_events(azure_events: AsyncGenerator) -> None:
            """Merge Azure results with speaker information and queue them for clients."""
            try:
                async for event in azure_events:
                    event_type = event.get("type")

                    if event_type == "result":
                        await schedule_identification(force=True, blocking=True)
                    elif event_type == "partial":
                        asyncio.create_task(
                            schedule_identification(force=False, blocking=False)
                        )
                    else:
                        asyncio.create_task(
                            schedule_identification(force=False, blocking=False)
                        )

                    event["speaker"] = current_speaker
                    if current_user_id:
                        event["user_id"] = current_user_id

                    if event.get("text"):
                        display_plain = f"{current_speaker}: {event['text']}"
                        event["display"] = display_plain
                        event["display_colored"] = _colorize(event_type or "", display_plain)

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

            logger.info("Recognition stream ended")
