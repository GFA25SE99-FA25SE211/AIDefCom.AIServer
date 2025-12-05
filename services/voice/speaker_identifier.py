"""Speaker Identification Manager - Handles speaker identification and switching logic.

This module extracts speaker identification logic from SpeechService to follow SRP.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Awaitable

import numpy as np

from services.audio.utils import pcm_to_wav
from core.executors import get_voice_executor, run_voice_bound

logger = logging.getLogger(__name__)


def _get_identification_config():
    """Load identification config from app config with fallbacks."""
    try:
        from app.config import Config
        return {
            "identify_window_seconds": Config.SPEAKER_IDENTIFY_WINDOW_SECONDS,
            "identify_interval_seconds": Config.SPEAKER_IDENTIFY_INTERVAL_SECONDS,
            "redis_timeout_seconds": Config.SPEAKER_REDIS_TIMEOUT_SECONDS,
            "fallback_cosine_threshold": Config.SPEAKER_FALLBACK_COSINE_THRESHOLD,
            "fallback_margin_threshold": Config.SPEAKER_FALLBACK_MARGIN_THRESHOLD,
            "weak_cosine_threshold": Config.SPEAKER_WEAK_COSINE_THRESHOLD,
            "cosine_threshold": Config.VOICE_COSINE_THRESHOLD,
            "speaker_switch_margin": Config.VOICE_SPEAKER_SWITCH_MARGIN,
            "speaker_switch_hits_required": Config.VOICE_SPEAKER_SWITCH_HITS_REQUIRED,
            "speaker_lock_decay_seconds": Config.VOICE_SPEAKER_LOCK_DECAY_SECONDS,
        }
    except Exception:
        return {
            "identify_window_seconds": 3.0,
            "identify_interval_seconds": 0.3,
            "redis_timeout_seconds": 0.5,
            "fallback_cosine_threshold": 0.30,
            "fallback_margin_threshold": 0.06,
            "weak_cosine_threshold": 0.22,
            "cosine_threshold": 0.45,
            "speaker_switch_margin": 0.06,
            "speaker_switch_hits_required": 3,
            "speaker_lock_decay_seconds": 5.0,
        }


@dataclass
class SpeakerIdentificationConfig:
    """Configuration for speaker identification."""
    
    sample_rate: int = 16000
    identify_window_seconds: float = 3.0
    identify_interval_seconds: float = 0.3
    redis_timeout_seconds: float = 0.5
    
    # Thresholds
    fallback_cosine_threshold: float = 0.30
    fallback_margin_threshold: float = 0.06
    weak_cosine_threshold: float = 0.22
    cosine_threshold: float = 0.45
    
    # Speaker switching
    speaker_switch_margin: float = 0.06
    speaker_switch_hits_required: int = 3
    speaker_lock_decay_seconds: float = 5.0
    
    # Reinforcement
    reinforcement_margin: float = 0.05
    
    @classmethod
    def from_app_config(cls) -> "SpeakerIdentificationConfig":
        """Create config from app configuration."""
        cfg = _get_identification_config()
        return cls(
            identify_window_seconds=cfg["identify_window_seconds"],
            identify_interval_seconds=cfg["identify_interval_seconds"],
            redis_timeout_seconds=cfg["redis_timeout_seconds"],
            fallback_cosine_threshold=cfg["fallback_cosine_threshold"],
            fallback_margin_threshold=cfg["fallback_margin_threshold"],
            weak_cosine_threshold=cfg["weak_cosine_threshold"],
            cosine_threshold=cfg["cosine_threshold"],
            speaker_switch_margin=cfg["speaker_switch_margin"],
            speaker_switch_hits_required=cfg["speaker_switch_hits_required"],
            speaker_lock_decay_seconds=cfg["speaker_lock_decay_seconds"],
        )


@dataclass
class SpeakerState:
    """Current state of speaker identification."""
    
    current_speaker: str = "Khách"
    current_user_id: Optional[str] = None
    confidence_score: float = 0.0
    last_reinforce_ts: float = 0.0
    last_switch_ts: float = 0.0
    
    # Pending candidate for speaker switch (requires multiple confirmations)
    pending_candidate: Optional[str] = None
    pending_user_id: Optional[str] = None
    pending_hits: int = 0
    pending_top_score: float = 0.0


@dataclass
class IdentificationResult:
    """Result of speaker identification."""
    
    speaker: str
    user_id: Optional[str]
    confidence_score: Optional[float]
    confidence_level: Optional[str]  # "High", "Medium", "Low", "uncertain"
    margin: Optional[float]
    switched: bool
    switch_reason: str = ""
    forced: bool = False


def normalize_speaker_label(name: str | None) -> str:
    """Normalize speaker label to consistent format."""
    if not name:
        return "Khách"
    normalized = name.strip()
    lowered = normalized.lower()
    if lowered in {"guest", "unknown", "khach", "khách", "dang xac dinh", "đang xác định"}:
        return "Khách"
    return normalized


class SpeakerIdentificationManager:
    """Manages speaker identification, switching, and state.
    
    Responsibilities:
    - Identify speaker from audio samples
    - Manage speaker switching with confirmation hits
    - Handle speaker lock/decay logic
    - Cache identification results
    
    This class is extracted from SpeechService to follow Single Responsibility Principle.
    """
    
    def __init__(
        self,
        voice_service: Any,  # IVoiceService
        redis_service: Optional[Any] = None,  # IRedisService
        config: Optional[SpeakerIdentificationConfig] = None,
        preloaded_profiles: Optional[List[Dict[str, Any]]] = None,
        whitelist_user_ids: Optional[List[str]] = None,
    ):
        """
        Initialize speaker identification manager.
        
        Args:
            voice_service: Voice authentication service
            redis_service: Optional Redis service for caching
            config: Identification configuration
            preloaded_profiles: Pre-loaded voice profiles for the session
            whitelist_user_ids: Whitelist of user IDs for this session
        """
        self.voice_service = voice_service
        self.redis_service = redis_service
        self.config = config or SpeakerIdentificationConfig.from_app_config()
        self.preloaded_profiles = preloaded_profiles
        self.whitelist_user_ids = whitelist_user_ids
        
        # State
        self.state = SpeakerState()
        self._state_lock = asyncio.Lock()
        self._identify_task: Optional[asyncio.Task] = None
        self._last_identify_ts: float = 0.0
    
    def _score_to_confidence(self, score: Optional[float]) -> Optional[str]:
        """Map cosine similarity score to confidence level."""
        if score is None:
            return None
        threshold = self.config.cosine_threshold
        if score >= threshold + 0.12:
            return "High"
        if score >= threshold + 0.05:
            return "Medium"
        return "Low"
    
    async def identify_speaker(
        self,
        audio_samples: np.ndarray,
        force_reidentify: bool = False,
    ) -> Optional[IdentificationResult]:
        """
        Identify speaker from audio samples.
        
        Args:
            audio_samples: Audio samples as int16 numpy array
            force_reidentify: Force re-identification even if recently identified
            
        Returns:
            Identification result or None if identification failed/skipped
        """
        if self.voice_service is None:
            logger.warning("⚠️ voice_service is None, skipping identification")
            return None
        
        loop = asyncio.get_running_loop()
        now = loop.time()
        
        # Throttle identification
        if not force_reidentify:
            if (now - self._last_identify_ts) < self.config.identify_interval_seconds:
                return None
        
        # Check minimum duration
        duration_sec = len(audio_samples) / self.config.sample_rate
        if duration_sec < 1.5:
            return None
        
        self._last_identify_ts = now
        
        # Window the audio
        window_samples = int(self.config.sample_rate * self.config.identify_window_seconds)
        if audio_samples.size > window_samples:
            audio_samples = audio_samples[-window_samples:]
        
        # Convert to WAV
        wav_bytes = pcm_to_wav(audio_samples.tobytes(), sample_rate=self.config.sample_rate)
        
        # Check cache
        cached_result = await self._check_cache(wav_bytes)
        if cached_result is not None:
            result = cached_result
        else:
            # Run identification in thread pool
            result = await self._run_identification(wav_bytes, loop)
            if result:
                # Cache result asynchronously
                asyncio.create_task(self._cache_result(wav_bytes, result))
        
        if not result:
            return None
        
        # Process result and update state
        return await self._process_identification_result(result, now, force_reidentify)
    
    async def _check_cache(self, wav_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Check Redis cache for identification result."""
        if not self.redis_service:
            return None
        
        try:
            import xxhash
            if len(wav_bytes) <= 8000:
                audio_hash = xxhash.xxh64(wav_bytes).hexdigest()[:16]
            else:
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
        
        try:
            cached = await asyncio.wait_for(
                self.redis_service.get(cache_key),
                timeout=self.config.redis_timeout_seconds
            )
            if cached:
                return cached
        except (asyncio.TimeoutError, Exception):
            pass
        
        return None
    
    async def _cache_result(self, wav_bytes: bytes, result: Dict[str, Any]) -> None:
        """Cache identification result in Redis."""
        if not self.redis_service:
            return
        
        try:
            import xxhash
            if len(wav_bytes) <= 8000:
                audio_hash = xxhash.xxh64(wav_bytes).hexdigest()[:16]
            else:
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
        
        try:
            await asyncio.wait_for(
                self.redis_service.set(cache_key, result, ttl=90),
                timeout=self.config.redis_timeout_seconds
            )
        except Exception:
            pass
    
    async def _run_identification(
        self,
        wav_bytes: bytes,
        loop: asyncio.AbstractEventLoop,
    ) -> Optional[Dict[str, Any]]:
        """Run voice identification in thread pool.
        
        Uses shared voice executor from core.executors to avoid blocking event loop.
        """
        try:
            if self.preloaded_profiles is not None:
                # Fast path: use preloaded profiles
                result = await run_voice_bound(
                    self.voice_service.identify_speaker_with_cache,
                    wav_bytes,
                    preloaded_profiles=self.preloaded_profiles,
                    pre_filtered=True,
                )
            else:
                # Slow path: load profiles on-demand
                result = await run_voice_bound(
                    self.voice_service.identify_speaker,
                    wav_bytes,
                    whitelist_user_ids=self.whitelist_user_ids,
                    pre_filtered=True,
                )
            
            return result
        
        except Exception as exc:
            logger.warning(f"Voice identification failed: {exc}")
            return None
    
    async def _process_identification_result(
        self,
        result: Dict[str, Any],
        now: float,
        force_reidentify: bool,
    ) -> IdentificationResult:
        """Process identification result and update speaker state."""
        
        candidates = result.get("candidates") or []
        
        # Extract identification info
        new_name = "Khách"
        new_user = None
        confidence_score: Optional[float] = None
        confidence_level: Optional[str] = None
        margin: Optional[float] = None
        forced = False
        
        if result.get("identified"):
            new_name = normalize_speaker_label(result.get("speaker"))
            new_user = result.get("user_id")
            score_val = result.get("score")
            if isinstance(score_val, (int, float)):
                confidence_score = float(score_val)
            confidence_level = result.get("confidence") or self._score_to_confidence(confidence_score)
            if len(candidates) > 1 and confidence_score is not None:
                second_score = candidates[1].get("cosine")
                if isinstance(second_score, (int, float)):
                    margin = confidence_score - float(second_score)
        else:
            # Fallback: use top candidate if reasonable
            if candidates:
                top = candidates[0]
                top_cosine = top.get("cosine", 0.0)
                second_cosine = candidates[1].get("cosine", 0.0) if len(candidates) > 1 else 0.0
                top_margin = top_cosine - second_cosine
                
                if top_cosine >= self.config.fallback_cosine_threshold and top_margin >= self.config.fallback_margin_threshold:
                    new_name = normalize_speaker_label(top.get("name"))
                    new_user = top.get("user_id") or top.get("id")
                    confidence_score = top_cosine
                    confidence_level = self._score_to_confidence(confidence_score)
                    margin = top_margin
                    forced = True
                elif top_cosine >= self.config.weak_cosine_threshold and top_margin >= (self.config.fallback_margin_threshold * 0.5):
                    new_name = normalize_speaker_label(top.get("name"))
                    new_user = top.get("user_id") or top.get("id")
                    confidence_score = top_cosine
                    confidence_level = "uncertain"
                    margin = top_margin
                    forced = True
        
        if not new_name:
            new_name = "Khách"
        if new_name == "Khách":
            new_user = None
        
        # Update state and determine if switch should happen
        switched, switch_reason = await self._update_speaker_state(
            new_name, new_user, confidence_score, margin, now, forced, force_reidentify
        )
        
        return IdentificationResult(
            speaker=self.state.current_speaker,
            user_id=self.state.current_user_id,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            margin=margin,
            switched=switched,
            switch_reason=switch_reason,
            forced=forced,
        )
    
    async def _update_speaker_state(
        self,
        new_name: str,
        new_user: Optional[str],
        score: Optional[float],
        margin: Optional[float],
        now: float,
        forced: bool,
        force_reidentify: bool,
    ) -> tuple[bool, str]:
        """
        Update speaker state based on identification result.
        
        Returns:
            Tuple of (switched, reason)
        """
        score_val = score or 0.0
        
        async with self._state_lock:
            # Check speaker lock decay
            base_decay = 3.0
            confidence_bonus = min(self.state.confidence_score * 2.0, 2.0)
            decay_seconds = base_decay + confidence_bonus
            
            if self.state.last_reinforce_ts and (now - self.state.last_reinforce_ts) > decay_seconds:
                if self.state.current_speaker != "Khách":
                    logger.info(f"⌛ Speaker lock decayed after {now - self.state.last_reinforce_ts:.2f}s")
                self.state = SpeakerState()  # Reset to default
            
            # Same speaker - reinforce if confident
            if new_name == self.state.current_speaker and new_user == self.state.current_user_id:
                if (margin is not None and margin >= self.config.reinforcement_margin) or \
                   (score_val >= self.config.cosine_threshold + 0.03):
                    self.state.last_reinforce_ts = now
                    self.state.confidence_score = score_val
                    self._reset_pending()
                    return False, "reinforced"
            
            # Determine if we should switch
            should_switch = False
            switch_reason = ""
            
            # Adaptive thresholds for small groups
            if self.whitelist_user_ids and len(self.whitelist_user_ids) <= 6:
                required_margin = 0.04
                required_hits = 2
            else:
                required_margin = self.config.speaker_switch_margin
                required_hits = self.config.speaker_switch_hits_required
            
            margin_strong = margin is not None and margin >= required_margin
            margin_ok = margin is not None and margin >= (required_margin * 0.67)
            
            # Fallback fast-path
            if forced and new_name != "Khách":
                if score_val >= self.config.weak_cosine_threshold and margin_ok:
                    should_switch = True
                    switch_reason = f"fallback_accepted_score_{score_val:.2f}"
            
            # Strong evidence - immediate switch
            if not should_switch and score_val >= (self.config.cosine_threshold + 0.05) and margin_strong:
                should_switch = True
                switch_reason = "immediate_strong_evidence"
            
            # Accumulate hits for pending candidate
            if not should_switch:
                if self.state.pending_candidate == new_name and self.state.pending_user_id == new_user:
                    self.state.pending_hits += 1
                    self.state.pending_top_score = max(self.state.pending_top_score, score_val)
                else:
                    self.state.pending_candidate = new_name
                    self.state.pending_user_id = new_user
                    self.state.pending_hits = 1
                    self.state.pending_top_score = score_val
                
                # Check if enough hits accumulated
                if self.state.pending_hits >= required_hits and score_val >= self.config.cosine_threshold and margin_ok:
                    should_switch = True
                    switch_reason = f"{required_hits}_hits_confirmed"
            
            if should_switch or force_reidentify:
                old_speaker = self.state.current_speaker
                self.state.current_speaker = new_name
                self.state.current_user_id = new_user
                self.state.confidence_score = score_val
                self.state.last_reinforce_ts = now
                self.state.last_switch_ts = now
                self._reset_pending()
                
                if old_speaker != new_name:
                    logger.info(f"✅ Speaker switched: {old_speaker} → {new_name} (reason={switch_reason})")
                    return True, switch_reason
            
            return False, ""
    
    def _reset_pending(self) -> None:
        """Reset pending candidate state."""
        self.state.pending_candidate = None
        self.state.pending_user_id = None
        self.state.pending_hits = 0
        self.state.pending_top_score = 0.0
    
    @property
    def current_speaker(self) -> str:
        """Get current speaker name."""
        return self.state.current_speaker
    
    @property
    def current_user_id(self) -> Optional[str]:
        """Get current speaker user ID."""
        return self.state.current_user_id
    
    def reset(self) -> None:
        """Reset identification state."""
        self.state = SpeakerState()
        self._last_identify_ts = 0.0
