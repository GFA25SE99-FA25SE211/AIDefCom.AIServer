"""Voice Activity Detection (VAD) using Silero VAD.

Provides AI-based voice detection that is much more accurate than RMS-based
approaches. Silero VAD can distinguish human speech from:
- Keyboard typing
- Fan noise
- Background music
- Other non-speech sounds

This module provides both synchronous and asynchronous APIs optimized for
real-time streaming applications.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Silero VAD configuration
SILERO_SAMPLING_RATE = 16000  # Silero VAD requires 16kHz
SILERO_WINDOW_SIZE_SAMPLES = 512  # ~32ms at 16kHz (must be 256, 512, or 768)


@dataclass
class VADSegment:
    """Represents a detected speech segment."""
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    confidence: float
    
    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass
class VADResult:
    """Result from VAD processing."""
    is_speech: bool
    confidence: float
    speech_ratio: float  # Ratio of speech frames in the audio
    segments: List[VADSegment]


class SileroVAD:
    """Silero VAD wrapper for voice activity detection.
    
    Uses the Silero VAD model which is:
    - Lightweight (~2MB)
    - Fast (real-time capable)
    - Accurate (trained on diverse speech data)
    - Language-agnostic
    
    Usage:
        vad = SileroVAD()
        
        # Check single audio chunk
        is_speech, confidence = vad.is_speech(audio_bytes)
        
        # Get detailed segments
        result = vad.detect(audio_bytes)
        for segment in result.segments:
            print(f"Speech from {segment.start_seconds:.2f}s to {segment.end_seconds:.2f}s")
    """
    
    # Class-level model cache (singleton pattern for efficiency)
    _model = None
    _model_lock = threading.Lock()
    _utils = None
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        sample_rate: int = SILERO_SAMPLING_RATE,
    ):
        """Initialize Silero VAD.
        
        Args:
            threshold: Speech probability threshold (0.0 to 1.0).
                      Higher = stricter (fewer false positives).
                      Default 0.5 is balanced.
            min_speech_duration_ms: Minimum speech segment duration in ms.
                                   Shorter segments are discarded as noise.
            min_silence_duration_ms: Minimum silence duration to split segments.
            sample_rate: Audio sample rate (must be 16000 for Silero).
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate
        
        # Validate sample rate
        if sample_rate != SILERO_SAMPLING_RATE:
            logger.warning(
                f"Silero VAD requires {SILERO_SAMPLING_RATE}Hz, got {sample_rate}Hz. "
                "Audio will be resampled automatically."
            )
        
        # Load model (lazy, cached)
        self._ensure_model_loaded()
    
    @classmethod
    def _ensure_model_loaded(cls) -> None:
        """Load Silero VAD model (thread-safe, cached)."""
        if cls._model is not None:
            return
        
        with cls._model_lock:
            if cls._model is not None:
                return
            
            try:
                logger.info("Loading Silero VAD model...")
                
                # Load model from torch hub (auto-downloads if needed)
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,  # Use PyTorch for better compatibility
                    trust_repo=True,
                )
                
                # Move to appropriate device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                model.eval()
                
                cls._model = model
                cls._utils = utils
                
                logger.info(f"Silero VAD loaded successfully | device={device}")
                
            except Exception as e:
                logger.error(f"Failed to load Silero VAD: {e}")
                raise RuntimeError(f"Failed to load Silero VAD model: {e}") from e
    
    def _prepare_audio(self, audio: Union[bytes, np.ndarray]) -> torch.Tensor:
        """Convert audio to tensor format required by Silero.
        
        Args:
            audio: Audio data as bytes (int16 PCM) or numpy array
            
        Returns:
            Tensor of shape (samples,) with values in [-1, 1]
        """
        if isinstance(audio, bytes):
            # Convert PCM bytes to numpy array
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        else:
            samples = audio.astype(np.float32)
        
        # Normalize to [-1, 1]
        if samples.max() > 1.0 or samples.min() < -1.0:
            samples = samples / 32768.0
        
        # Resample if needed
        if self.sample_rate != SILERO_SAMPLING_RATE:
            samples = self._resample(samples, self.sample_rate, SILERO_SAMPLING_RATE)
        
        # Convert to tensor
        tensor = torch.from_numpy(samples)
        
        # Move to same device as model
        if self._model is not None:
            tensor = tensor.to(next(self._model.parameters()).device)
        
        return tensor
    
    @staticmethod
    def _resample(signal: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if src_sr == target_sr or signal.size == 0:
            return signal
        
        ratio = target_sr / float(src_sr)
        new_length = int(len(signal) * ratio)
        indices = np.linspace(0, len(signal) - 1, new_length)
        return np.interp(indices, np.arange(len(signal)), signal).astype(np.float32)
    
    def is_speech(
        self, 
        audio: Union[bytes, np.ndarray],
    ) -> Tuple[bool, float]:
        """Check if audio contains speech.
        
        Fast method for real-time streaming - checks entire chunk at once.
        
        Args:
            audio: Audio data (PCM bytes or numpy array)
            
        Returns:
            Tuple of (is_speech: bool, confidence: float)
        """
        if not audio or (isinstance(audio, bytes) and len(audio) < 64):
            return False, 0.0
        
        try:
            tensor = self._prepare_audio(audio)
            
            if tensor.numel() < SILERO_WINDOW_SIZE_SAMPLES:
                # Too short for VAD
                return False, 0.0
            
            # Get speech probability
            with torch.no_grad():
                # Reset model state for new audio
                self._model.reset_states()
                
                # Process in windows
                probabilities = []
                for i in range(0, len(tensor) - SILERO_WINDOW_SIZE_SAMPLES + 1, SILERO_WINDOW_SIZE_SAMPLES):
                    window = tensor[i:i + SILERO_WINDOW_SIZE_SAMPLES]
                    prob = self._model(window, SILERO_SAMPLING_RATE).item()
                    probabilities.append(prob)
                
                if not probabilities:
                    return False, 0.0
                
                # Use max probability (any speech detected)
                max_prob = max(probabilities)
                avg_prob = sum(probabilities) / len(probabilities)
                
                is_speech = max_prob >= self.threshold
                confidence = avg_prob
                
                return is_speech, confidence
                
        except Exception as e:
            logger.warning(f"VAD error: {e}")
            return False, 0.0
    
    def detect(
        self,
        audio: Union[bytes, np.ndarray],
    ) -> VADResult:
        """Detect speech segments in audio.
        
        More detailed than is_speech() - returns all speech segments.
        
        Args:
            audio: Audio data (PCM bytes or numpy array)
            
        Returns:
            VADResult with speech segments
        """
        if not audio or (isinstance(audio, bytes) and len(audio) < 64):
            return VADResult(
                is_speech=False,
                confidence=0.0,
                speech_ratio=0.0,
                segments=[]
            )
        
        try:
            tensor = self._prepare_audio(audio)
            total_samples = len(tensor)
            
            if total_samples < SILERO_WINDOW_SIZE_SAMPLES:
                return VADResult(
                    is_speech=False,
                    confidence=0.0,
                    speech_ratio=0.0,
                    segments=[]
                )
            
            # Get frame-by-frame probabilities
            with torch.no_grad():
                self._model.reset_states()
                
                frame_probs = []
                for i in range(0, total_samples - SILERO_WINDOW_SIZE_SAMPLES + 1, SILERO_WINDOW_SIZE_SAMPLES):
                    window = tensor[i:i + SILERO_WINDOW_SIZE_SAMPLES]
                    prob = self._model(window, SILERO_SAMPLING_RATE).item()
                    frame_probs.append((i, prob))
            
            if not frame_probs:
                return VADResult(
                    is_speech=False,
                    confidence=0.0,
                    speech_ratio=0.0,
                    segments=[]
                )
            
            # Find speech segments
            segments = self._extract_segments(frame_probs, total_samples)
            
            # Calculate statistics
            speech_frames = sum(1 for _, p in frame_probs if p >= self.threshold)
            speech_ratio = speech_frames / len(frame_probs)
            avg_confidence = sum(p for _, p in frame_probs) / len(frame_probs)
            is_speech = len(segments) > 0
            
            return VADResult(
                is_speech=is_speech,
                confidence=avg_confidence,
                speech_ratio=speech_ratio,
                segments=segments
            )
            
        except Exception as e:
            logger.warning(f"VAD detection error: {e}")
            return VADResult(
                is_speech=False,
                confidence=0.0,
                speech_ratio=0.0,
                segments=[]
            )
    
    def _extract_segments(
        self,
        frame_probs: List[Tuple[int, float]],
        total_samples: int,
    ) -> List[VADSegment]:
        """Extract speech segments from frame probabilities."""
        segments = []
        in_speech = False
        segment_start = 0
        segment_probs = []
        
        min_speech_samples = int(self.min_speech_duration_ms * SILERO_SAMPLING_RATE / 1000)
        min_silence_samples = int(self.min_silence_duration_ms * SILERO_SAMPLING_RATE / 1000)
        
        silence_counter = 0
        
        for sample_idx, prob in frame_probs:
            is_frame_speech = prob >= self.threshold
            
            if is_frame_speech:
                if not in_speech:
                    # Start new segment
                    in_speech = True
                    segment_start = sample_idx
                    segment_probs = [prob]
                else:
                    segment_probs.append(prob)
                silence_counter = 0
            else:
                if in_speech:
                    silence_counter += SILERO_WINDOW_SIZE_SAMPLES
                    
                    if silence_counter >= min_silence_samples:
                        # End segment
                        segment_end = sample_idx
                        duration_samples = segment_end - segment_start
                        
                        if duration_samples >= min_speech_samples:
                            avg_conf = sum(segment_probs) / len(segment_probs) if segment_probs else 0.0
                            segments.append(VADSegment(
                                start_sample=segment_start,
                                end_sample=segment_end,
                                start_seconds=segment_start / SILERO_SAMPLING_RATE,
                                end_seconds=segment_end / SILERO_SAMPLING_RATE,
                                confidence=avg_conf,
                            ))
                        
                        in_speech = False
                        segment_probs = []
                        silence_counter = 0
        
        # Handle segment at end
        if in_speech and segment_probs:
            segment_end = total_samples
            duration_samples = segment_end - segment_start
            
            if duration_samples >= min_speech_samples:
                avg_conf = sum(segment_probs) / len(segment_probs)
                segments.append(VADSegment(
                    start_sample=segment_start,
                    end_sample=segment_end,
                    start_seconds=segment_start / SILERO_SAMPLING_RATE,
                    end_seconds=segment_end / SILERO_SAMPLING_RATE,
                    confidence=avg_conf,
                ))
        
        return segments
    
    def filter_audio(
        self,
        audio: Union[bytes, np.ndarray],
        return_bytes: bool = True,
    ) -> Union[bytes, np.ndarray]:
        """Filter audio to keep only speech segments.
        
        Useful for removing silence/noise before sending to STT.
        
        Args:
            audio: Input audio
            return_bytes: If True, return bytes; otherwise numpy array
            
        Returns:
            Filtered audio containing only speech
        """
        result = self.detect(audio)
        
        if not result.segments:
            # No speech detected - return empty or very short audio
            if return_bytes:
                return b''
            return np.array([], dtype=np.int16)
        
        # Convert to numpy if needed
        if isinstance(audio, bytes):
            samples = np.frombuffer(audio, dtype=np.int16)
        else:
            samples = audio
        
        # Extract speech segments
        speech_parts = []
        for segment in result.segments:
            # Adjust for original sample rate
            if self.sample_rate != SILERO_SAMPLING_RATE:
                ratio = self.sample_rate / SILERO_SAMPLING_RATE
                start = int(segment.start_sample * ratio)
                end = int(segment.end_sample * ratio)
            else:
                start = segment.start_sample
                end = segment.end_sample
            
            start = max(0, start)
            end = min(len(samples), end)
            speech_parts.append(samples[start:end])
        
        if not speech_parts:
            if return_bytes:
                return b''
            return np.array([], dtype=np.int16)
        
        # Concatenate segments
        filtered = np.concatenate(speech_parts)
        
        if return_bytes:
            return filtered.astype(np.int16).tobytes()
        return filtered


class WebRTCVAD:
    """WebRTC VAD wrapper as a lightweight alternative to Silero.
    
    Uses the py-webrtcvad library which is:
    - Very lightweight (no ML model)
    - Extremely fast
    - Good for simple voice detection
    
    Pros:
    - No model loading time
    - Lower CPU usage
    - Works well for clean audio
    
    Cons:
    - Less accurate than Silero for noisy environments
    - Can mistake some noises for speech
    """
    
    def __init__(
        self,
        aggressiveness: int = 2,
        sample_rate: int = 16000,
    ):
        """Initialize WebRTC VAD.
        
        Args:
            aggressiveness: VAD aggressiveness mode (0-3).
                           0 = least aggressive (more false positives)
                           3 = most aggressive (more false negatives)
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
        """
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(aggressiveness)
            self.sample_rate = sample_rate
            self.frame_duration_ms = 30  # WebRTC VAD supports 10, 20, or 30ms frames
            self._available = True
            logger.info(f"WebRTC VAD initialized | aggressiveness={aggressiveness}")
        except ImportError:
            logger.warning("webrtcvad not installed. Install with: pip install webrtcvad")
            self._vad = None
            self._available = False
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def is_speech(self, audio: bytes) -> Tuple[bool, float]:
        """Check if audio contains speech.
        
        Args:
            audio: PCM audio bytes (int16)
            
        Returns:
            Tuple of (is_speech, confidence)
            Note: WebRTC VAD doesn't provide confidence, so we estimate it.
        """
        if not self._available or not audio:
            return False, 0.0
        
        try:
            frame_size = int(self.sample_rate * self.frame_duration_ms / 1000) * 2  # bytes
            
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio) - frame_size + 1, frame_size):
                frame = audio[i:i + frame_size]
                if len(frame) == frame_size:
                    if self._vad.is_speech(frame, self.sample_rate):
                        speech_frames += 1
                    total_frames += 1
            
            if total_frames == 0:
                return False, 0.0
            
            speech_ratio = speech_frames / total_frames
            is_speech = speech_ratio >= 0.3  # At least 30% speech frames
            
            return is_speech, speech_ratio
            
        except Exception as e:
            logger.warning(f"WebRTC VAD error: {e}")
            return False, 0.0


# Global VAD instance (lazy initialization)
_silero_vad: Optional[SileroVAD] = None
_webrtc_vad: Optional[WebRTCVAD] = None


def get_silero_vad(
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
) -> SileroVAD:
    """Get or create global Silero VAD instance.
    
    Args:
        threshold: Speech probability threshold
        min_speech_duration_ms: Minimum speech duration
        
    Returns:
        SileroVAD instance
    """
    global _silero_vad
    if _silero_vad is None:
        _silero_vad = SileroVAD(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
        )
    return _silero_vad


def get_webrtc_vad(aggressiveness: int = 2) -> WebRTCVAD:
    """Get or create global WebRTC VAD instance.
    
    Args:
        aggressiveness: VAD aggressiveness (0-3)
        
    Returns:
        WebRTCVAD instance
    """
    global _webrtc_vad
    if _webrtc_vad is None:
        _webrtc_vad = WebRTCVAD(aggressiveness=aggressiveness)
    return _webrtc_vad


def is_speech(
    audio: Union[bytes, np.ndarray],
    use_silero: bool = True,
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """Convenience function to check if audio contains speech.
    
    Uses Silero VAD by default, falls back to WebRTC if unavailable.
    
    Args:
        audio: Audio data (PCM bytes or numpy array)
        use_silero: Whether to prefer Silero VAD (default True)
        threshold: Speech probability threshold for Silero
        
    Returns:
        Tuple of (is_speech, confidence)
    """
    if use_silero:
        try:
            vad = get_silero_vad(threshold=threshold)
            return vad.is_speech(audio)
        except Exception as e:
            logger.warning(f"Silero VAD failed, falling back to WebRTC: {e}")
    
    # Fallback to WebRTC
    webrtc = get_webrtc_vad()
    if webrtc.is_available:
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.int16).tobytes()
        return webrtc.is_speech(audio)
    
    # No VAD available - return True to not block audio
    logger.warning("No VAD available, assuming speech")
    return True, 0.5
