"""Embedding-based Speaker Change Detection.

Detects speaker changes (interruptions) by comparing audio embeddings
rather than just energy levels. This is more accurate because:

1. Energy spike detection fails when:
   - Current speaker raises their voice
   - Background noise increases
   - Someone coughs/laughs

2. Embedding-based detection checks if the audio "sounds like" a different person,
   which is the actual goal of interruption detection.

Algorithm:
1. Maintain embedding of "active speaker" (last N seconds)
2. Compute embedding of new audio chunk
3. If similarity to active speaker drops below threshold, flag as interruption
4. Optional: Use VAD to only compare when speech is detected
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple, Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingExtractor(Protocol):
    """Protocol for embedding extraction models."""
    
    def extract_embedding(self, audio_signal: np.ndarray) -> np.ndarray:
        """Extract embedding from audio signal.
        
        Args:
            audio_signal: Audio samples as float32, normalized to [-1, 1]
            
        Returns:
            Embedding vector
        """
        ...


@dataclass
class InterruptionEvent:
    """Represents a detected interruption event."""
    timestamp: float
    confidence: float
    active_speaker_similarity: float
    reason: str
    old_speaker_id: Optional[str] = None
    new_speaker_embedding: Optional[np.ndarray] = None


@dataclass
class SpeakerEmbeddingState:
    """State tracking for a speaker's embeddings."""
    speaker_id: str
    speaker_name: str
    embeddings: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    last_active: float = 0.0
    
    def get_mean_embedding(self) -> Optional[np.ndarray]:
        """Get mean embedding from recent samples."""
        if not self.embeddings:
            return None
        matrix = np.stack(list(self.embeddings), axis=0)
        mean = np.mean(matrix, axis=0)
        return mean / (np.linalg.norm(mean) + 1e-8)
    
    def add_embedding(self, embedding: np.ndarray, timestamp: float) -> None:
        """Add new embedding sample."""
        normalized = embedding / (np.linalg.norm(embedding) + 1e-8)
        self.embeddings.append(normalized)
        self.last_active = timestamp


class EmbeddingInterruptionDetector:
    """Detects speaker interruptions using embedding comparison.
    
    Instead of just detecting energy spikes, this compares the acoustic
    characteristics of the current audio to the known active speaker.
    
    Algorithm:
    1. Track active speaker's recent embeddings (rolling window)
    2. Extract embedding from new audio chunk
    3. Compare to active speaker's mean embedding
    4. If similarity drops significantly, flag as potential interruption
    5. Use hysteresis to avoid false positives from temporary variations
    
    Benefits:
    - Distinguishes "speaker raises voice" from "different speaker"
    - Works in noisy environments
    - Adapts to speaker variations over time
    """
    
    def __init__(
        self,
        embedding_extractor: Optional[EmbeddingExtractor] = None,
        similarity_threshold: float = 0.65,  # Below this = different speaker
        hysteresis_count: int = 2,  # Require N consecutive low-similarity chunks
        embedding_window_size: int = 10,  # Keep last N embeddings per speaker
        min_audio_duration: float = 0.5,  # Minimum audio length for embedding
        sample_rate: int = 16000,
    ):
        """Initialize interruption detector.
        
        Args:
            embedding_extractor: Model for extracting audio embeddings
            similarity_threshold: Cosine similarity threshold for same-speaker
            hysteresis_count: Number of consecutive low-sim chunks to trigger
            embedding_window_size: Rolling window of embeddings per speaker
            min_audio_duration: Minimum audio duration for reliable embedding
            sample_rate: Audio sample rate
        """
        self.embedding_extractor = embedding_extractor
        self.similarity_threshold = similarity_threshold
        self.hysteresis_count = hysteresis_count
        self.embedding_window_size = embedding_window_size
        self.min_audio_duration = min_audio_duration
        self.sample_rate = sample_rate
        self.min_samples = int(min_audio_duration * sample_rate)
        
        # State
        self._active_speaker: Optional[SpeakerEmbeddingState] = None
        self._speaker_states: Dict[str, SpeakerEmbeddingState] = {}
        self._low_similarity_streak = 0
        self._last_probe_embedding: Optional[np.ndarray] = None
        self._audio_buffer = bytearray()
        
        # Energy-based fallback
        self._previous_rms: Optional[float] = None
    
    def set_embedding_extractor(self, extractor: EmbeddingExtractor) -> None:
        """Set the embedding extractor model."""
        self.embedding_extractor = extractor
    
    def set_active_speaker(
        self,
        speaker_id: str,
        speaker_name: str,
        initial_embedding: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> None:
        """Set the currently active speaker.
        
        Call this after successful speaker identification.
        
        Args:
            speaker_id: Unique speaker ID
            speaker_name: Display name
            initial_embedding: Optional initial embedding (from identification)
            timestamp: Current timestamp
        """
        if speaker_id not in self._speaker_states:
            self._speaker_states[speaker_id] = SpeakerEmbeddingState(
                speaker_id=speaker_id,
                speaker_name=speaker_name,
            )
        
        state = self._speaker_states[speaker_id]
        state.last_active = timestamp
        
        if initial_embedding is not None:
            state.add_embedding(initial_embedding, timestamp)
        
        self._active_speaker = state
        self._low_similarity_streak = 0
        
        logger.debug(f"Active speaker set: {speaker_name} ({speaker_id})")
    
    def update_active_speaker_embedding(
        self,
        embedding: np.ndarray,
        timestamp: float,
    ) -> None:
        """Update active speaker's embedding (during normal speech).
        
        Call this periodically during recognized speech to adapt
        to the speaker's current acoustic characteristics.
        """
        if self._active_speaker is None:
            return
        
        self._active_speaker.add_embedding(embedding, timestamp)
    
    def detect_interruption(
        self,
        audio_chunk: bytes,
        timestamp: float,
        use_vad: bool = True,
    ) -> Optional[InterruptionEvent]:
        """Check if the audio chunk represents an interruption.
        
        Args:
            audio_chunk: PCM audio bytes (int16)
            timestamp: Current timestamp
            use_vad: If True, skip quiet audio (no speech)
            
        Returns:
            InterruptionEvent if interruption detected, None otherwise
        """
        if self.embedding_extractor is None:
            # Fallback to energy-based detection
            return self._detect_by_energy(audio_chunk, timestamp)
        
        if self._active_speaker is None:
            # No active speaker to compare against
            return None
        
        # Convert to float audio
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
        
        # Simple VAD: check if audio has enough energy
        if use_vad:
            rms = float(np.sqrt(np.mean(samples * samples)))
            if rms < 0.01:  # Very quiet
                return None
        
        # Buffer audio until we have enough for embedding
        self._audio_buffer.extend(audio_chunk)
        
        buffered_samples = len(self._audio_buffer) // 2  # 2 bytes per sample
        if buffered_samples < self.min_samples:
            return None
        
        # Extract embedding from buffered audio
        try:
            buffer_array = np.frombuffer(bytes(self._audio_buffer), dtype=np.int16).astype(np.float32)
            buffer_array = buffer_array / 32768.0
            
            # Remove DC offset
            buffer_array = buffer_array - np.mean(buffer_array)
            
            probe_embedding = self.embedding_extractor.extract_embedding(buffer_array)
            self._last_probe_embedding = probe_embedding
            
            # Clear buffer after extraction
            self._audio_buffer.clear()
            
        except Exception as e:
            logger.warning(f"Failed to extract embedding for interruption detection: {e}")
            return None
        
        # Get active speaker's reference embedding
        active_embedding = self._active_speaker.get_mean_embedding()
        if active_embedding is None:
            return None
        
        # Compute similarity
        probe_norm = probe_embedding / (np.linalg.norm(probe_embedding) + 1e-8)
        similarity = float(np.dot(probe_norm, active_embedding))
        
        logger.debug(
            f"Interruption check | active={self._active_speaker.speaker_name} | "
            f"similarity={similarity:.3f} | threshold={self.similarity_threshold}"
        )
        
        # Check if below threshold
        if similarity < self.similarity_threshold:
            self._low_similarity_streak += 1
            
            if self._low_similarity_streak >= self.hysteresis_count:
                # Confirmed interruption
                event = InterruptionEvent(
                    timestamp=timestamp,
                    confidence=1.0 - similarity,
                    active_speaker_similarity=similarity,
                    reason=f"embedding_mismatch ({similarity:.3f} < {self.similarity_threshold})",
                    old_speaker_id=self._active_speaker.speaker_id,
                    new_speaker_embedding=probe_embedding,
                )
                
                logger.info(
                    f"⚡ Interruption detected | "
                    f"active={self._active_speaker.speaker_name} | "
                    f"similarity={similarity:.3f} | "
                    f"streak={self._low_similarity_streak}"
                )
                
                # Reset streak
                self._low_similarity_streak = 0
                
                return event
        else:
            # Reset streak on high similarity
            self._low_similarity_streak = 0
            
            # Update active speaker's embedding (adapting to current audio)
            self._active_speaker.add_embedding(probe_embedding, timestamp)
        
        return None
    
    def _detect_by_energy(
        self,
        audio_chunk: bytes,
        timestamp: float,
    ) -> Optional[InterruptionEvent]:
        """Fallback energy-based interruption detection.
        
        Used when embedding extractor is not available.
        """
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
        
        rms = float(np.sqrt(np.mean(samples * samples)))
        
        if self._previous_rms is not None:
            ratio = rms / (self._previous_rms + 1e-8)
            
            # Energy spike = 2x increase
            if ratio > 2.0 and rms > 0.05:
                event = InterruptionEvent(
                    timestamp=timestamp,
                    confidence=min(1.0, (ratio - 2.0) / 2.0),
                    active_speaker_similarity=-1.0,  # Unknown
                    reason=f"energy_spike (ratio={ratio:.2f})",
                )
                logger.debug(f"⚡ Energy spike detected | ratio={ratio:.2f}")
                self._previous_rms = rms
                return event
        
        self._previous_rms = rms
        return None
    
    def get_last_probe_embedding(self) -> Optional[np.ndarray]:
        """Get the last computed probe embedding.
        
        Useful for speaker identification after interruption.
        """
        return self._last_probe_embedding
    
    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        self._audio_buffer.clear()
    
    def reset(self) -> None:
        """Reset detector state."""
        self._active_speaker = None
        self._speaker_states.clear()
        self._low_similarity_streak = 0
        self._last_probe_embedding = None
        self._audio_buffer.clear()
        self._previous_rms = None


class CombinedInterruptionDetector:
    """Combines embedding-based and energy-based detection.
    
    Uses:
    1. Energy spike as quick filter (low cost)
    2. Embedding comparison for confirmation (high accuracy)
    
    This provides both speed and accuracy.
    """
    
    def __init__(
        self,
        embedding_detector: Optional[EmbeddingInterruptionDetector] = None,
        energy_spike_threshold: float = 1.8,
        require_embedding_confirm: bool = True,
    ):
        """Initialize combined detector.
        
        Args:
            embedding_detector: Embedding-based detector
            energy_spike_threshold: Threshold for energy-based pre-filter
            require_embedding_confirm: If True, energy spikes need embedding confirmation
        """
        self.embedding_detector = embedding_detector or EmbeddingInterruptionDetector()
        self.energy_spike_threshold = energy_spike_threshold
        self.require_embedding_confirm = require_embedding_confirm
        
        self._previous_rms: Optional[float] = None
        self._pending_spike_timestamp: Optional[float] = None
    
    def set_embedding_extractor(self, extractor: EmbeddingExtractor) -> None:
        """Set embedding extractor for the detector."""
        self.embedding_detector.set_embedding_extractor(extractor)
    
    def set_active_speaker(
        self,
        speaker_id: str,
        speaker_name: str,
        initial_embedding: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> None:
        """Set active speaker for comparison."""
        self.embedding_detector.set_active_speaker(
            speaker_id, speaker_name, initial_embedding, timestamp
        )
    
    def detect(
        self,
        audio_chunk: bytes,
        timestamp: float,
    ) -> Optional[InterruptionEvent]:
        """Detect interruption using combined approach.
        
        Args:
            audio_chunk: PCM audio bytes
            timestamp: Current timestamp
            
        Returns:
            InterruptionEvent if interruption confirmed
        """
        # Step 1: Energy-based pre-filter
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples)))
        
        energy_spike = False
        if self._previous_rms is not None:
            ratio = rms / (self._previous_rms + 1e-8)
            if ratio > self.energy_spike_threshold and rms > 0.03:
                energy_spike = True
                self._pending_spike_timestamp = timestamp
                logger.debug(f"📈 Energy spike pre-filter triggered | ratio={ratio:.2f}")
        
        self._previous_rms = rms
        
        # Step 2: Embedding-based confirmation
        if self.embedding_detector.embedding_extractor is not None:
            event = self.embedding_detector.detect_interruption(audio_chunk, timestamp)
            
            if event is not None:
                # Embedding confirmed interruption
                return event
            
            # Energy spike but no embedding confirmation
            if energy_spike and not self.require_embedding_confirm:
                return InterruptionEvent(
                    timestamp=timestamp,
                    confidence=0.5,  # Lower confidence without embedding confirm
                    active_speaker_similarity=-1.0,
                    reason="energy_spike_only",
                )
        else:
            # No embedding extractor - fall back to energy only
            if energy_spike:
                return InterruptionEvent(
                    timestamp=timestamp,
                    confidence=0.5,
                    active_speaker_similarity=-1.0,
                    reason="energy_spike_fallback",
                )
        
        return None
    
    def update_active_speaker_embedding(
        self,
        embedding: np.ndarray,
        timestamp: float,
    ) -> None:
        """Update active speaker's embedding."""
        self.embedding_detector.update_active_speaker_embedding(embedding, timestamp)
    
    def reset(self) -> None:
        """Reset detector state."""
        self.embedding_detector.reset()
        self._previous_rms = None
        self._pending_spike_timestamp = None
