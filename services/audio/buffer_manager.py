"""Audio Buffer Manager - Handles audio buffering, framing, and history management.

This module extracts audio processing logic from SpeechService to follow SRP.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional, Callable, Awaitable

import numpy as np

from services.audio.utils import NoiseFilter, detect_energy_spike, detect_acoustic_change

logger = logging.getLogger(__name__)


@dataclass
class AudioBufferConfig:
    """Configuration for audio buffer management."""
    
    sample_rate: int = 16000
    frame_duration_ms: int = 20  # 20ms frames
    history_seconds: float = 5.0  # Keep 5s of audio history
    min_identify_seconds: float = 2.0  # Minimum audio for speaker ID
    apply_noise_filter: bool = True
    
    # Interruption detection
    energy_spike_threshold: float = 1.5
    acoustic_change_threshold: float = 0.3
    
    @property
    def frame_samples(self) -> int:
        """Number of samples per frame."""
        return int(self.sample_rate * self.frame_duration_ms / 1000)
    
    @property
    def frame_bytes(self) -> int:
        """Bytes per frame (16-bit audio = 2 bytes per sample)."""
        return self.frame_samples * 2
    
    @property
    def history_limit_bytes(self) -> int:
        """Maximum bytes to keep in history buffer."""
        return int(self.sample_rate * self.history_seconds) * 2
    
    @property
    def min_identify_bytes(self) -> int:
        """Minimum bytes needed for speaker identification."""
        return int(self.sample_rate * self.min_identify_seconds) * 2


class AudioBufferManager:
    """Manages audio buffering, framing, and history for speech recognition.
    
    Responsibilities:
    - Buffer incoming audio chunks into fixed-size frames
    - Maintain rolling audio history for speaker identification
    - Apply noise filtering
    - Detect speaker interruptions (energy spikes, acoustic changes)
    
    This class is extracted from SpeechService to follow Single Responsibility Principle.
    """
    
    def __init__(self, config: Optional[AudioBufferConfig] = None):
        """
        Initialize audio buffer manager.
        
        Args:
            config: Buffer configuration (uses defaults if not provided)
        """
        self.config = config or AudioBufferConfig()
        self.noise_filter = NoiseFilter() if self.config.apply_noise_filter else None
        
        # Internal state
        self._frame_buffer = bytearray()
        self._audio_history = bytearray()
        self._history_lock = asyncio.Lock()
        self._previous_chunk: Optional[bytes] = None
        self._chunks_processed = 0
    
    async def process_audio_stream(
        self,
        audio_queue: asyncio.Queue[bytes | None],
        on_interruption_detected: Optional[Callable[[], Awaitable[None]]] = None,
        on_sufficient_audio: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Process incoming audio from queue, yielding fixed-size frames.
        
        Args:
            audio_queue: Queue receiving raw audio bytes (None signals end)
            on_interruption_detected: Callback when speaker interruption is detected
            on_sufficient_audio: Callback when enough audio is buffered for identification
            
        Yields:
            Processed audio frames (noise-filtered if enabled)
        """
        try:
            while True:
                audio_data = await audio_queue.get()
                
                # None signals end of stream
                if audio_data is None:
                    break
                
                if not audio_data:
                    continue
                
                self._chunks_processed += 1
                
                # Limit chunk size to prevent buffer overflow
                if len(audio_data) > 8192:
                    audio_data = audio_data[:8192]
                
                self._frame_buffer.extend(audio_data)
                
                # Process complete frames
                while len(self._frame_buffer) >= self.config.frame_bytes:
                    raw_frame = bytes(self._frame_buffer[:self.config.frame_bytes])
                    del self._frame_buffer[:self.config.frame_bytes]
                    
                    # Apply noise filtering
                    if self.noise_filter:
                        chunk = self.noise_filter.reduce_noise(raw_frame)
                    else:
                        chunk = raw_frame
                    
                    # Check for interruption
                    if self._previous_chunk and on_interruption_detected:
                        if self._detect_interruption(chunk):
                            await on_interruption_detected()
                    
                    self._previous_chunk = chunk
                    
                    # Update history
                    async with self._history_lock:
                        self._audio_history.extend(chunk)
                        if len(self._audio_history) > self.config.history_limit_bytes:
                            overflow = len(self._audio_history) - self.config.history_limit_bytes
                            del self._audio_history[:overflow]
                        
                        # Check if we have enough audio for identification
                        if on_sufficient_audio and len(self._audio_history) >= self.config.min_identify_bytes:
                            await on_sufficient_audio()
                    
                    # Yield control to event loop
                    await asyncio.sleep(0)
                    
                    yield chunk
            
            # Process remaining buffer
            if self._frame_buffer:
                remainder = bytes(self._frame_buffer)
                self._frame_buffer.clear()
                if remainder:
                    chunk = self.noise_filter.reduce_noise(remainder) if self.noise_filter else remainder
                    async with self._history_lock:
                        self._audio_history.extend(chunk)
                        if len(self._audio_history) > self.config.history_limit_bytes:
                            overflow = len(self._audio_history) - self.config.history_limit_bytes
                            del self._audio_history[:overflow]
                    yield chunk
        
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Audio stream interrupted: {exc}")
    
    def _detect_interruption(self, current_chunk: bytes) -> bool:
        """
        Detect if there's a speaker interruption based on audio characteristics.
        
        Args:
            current_chunk: Current audio frame
            
        Returns:
            True if interruption detected
        """
        if not self._previous_chunk:
            return False
        
        # Check energy spike
        if detect_energy_spike(
            current_chunk, 
            self._previous_chunk, 
            spike_threshold=self.config.energy_spike_threshold
        ):
            return True
        
        # Check acoustic change
        if detect_acoustic_change(
            current_chunk,
            self._previous_chunk,
            change_threshold=self.config.acoustic_change_threshold
        ):
            return True
        
        return False
    
    async def get_audio_snapshot(self, window_seconds: Optional[float] = None) -> bytes:
        """
        Get a snapshot of the audio history buffer.
        
        Args:
            window_seconds: If provided, return only the last N seconds
            
        Returns:
            PCM audio bytes
        """
        async with self._history_lock:
            pcm_data = bytes(self._audio_history)
        
        if window_seconds and pcm_data:
            window_bytes = int(self.config.sample_rate * window_seconds) * 2
            if len(pcm_data) > window_bytes:
                pcm_data = pcm_data[-window_bytes:]
        
        return pcm_data
    
    async def get_audio_as_numpy(self, window_seconds: Optional[float] = None) -> np.ndarray:
        """
        Get audio history as numpy array.
        
        Args:
            window_seconds: If provided, return only the last N seconds
            
        Returns:
            Audio samples as int16 numpy array
        """
        pcm_data = await self.get_audio_snapshot(window_seconds)
        if not pcm_data:
            return np.array([], dtype=np.int16)
        return np.frombuffer(pcm_data, dtype=np.int16)
    
    async def has_sufficient_audio(self) -> bool:
        """Check if enough audio is buffered for speaker identification."""
        async with self._history_lock:
            return len(self._audio_history) >= self.config.min_identify_bytes
    
    async def get_history_duration(self) -> float:
        """Get current history duration in seconds."""
        async with self._history_lock:
            return len(self._audio_history) / 2 / self.config.sample_rate
    
    async def clear_history(self) -> None:
        """Clear audio history buffer."""
        async with self._history_lock:
            self._audio_history.clear()
        self._previous_chunk = None
    
    async def trim_history(self) -> None:
        """Trim history to half of limit (for memory management after identification)."""
        async with self._history_lock:
            if len(self._audio_history) > self.config.history_limit_bytes * 1.5:
                del self._audio_history[:self.config.history_limit_bytes // 2]
    
    @property
    def chunks_processed(self) -> int:
        """Number of audio chunks processed."""
        return self._chunks_processed
    
    def reset(self) -> None:
        """Reset buffer state for reuse."""
        self._frame_buffer.clear()
        self._audio_history.clear()
        self._previous_chunk = None
        self._chunks_processed = 0
