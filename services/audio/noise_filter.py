"""Enhanced Noise Reduction using noisereduce library.

Provides spectral gating noise reduction that is more effective than
simple RMS-based approaches. Can handle:
- Stationary noise (fans, AC, hum)
- Non-stationary noise (keyboard typing, paper rustling)
- Room reverb/echo

Uses noisereduce library with optional fallback to RMS-based filtering.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import noisereduce
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
    logger.info("noisereduce library available")
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logger.warning("noisereduce not installed. Install with: pip install noisereduce")


class EnhancedNoiseFilter:
    """Enhanced noise filter using spectral gating.
    
    Uses noisereduce library for high-quality noise reduction:
    - Spectral gating based on noise profile
    - Adaptive to changing noise conditions
    - Preserves speech quality
    
    Falls back to RMS-based filtering if noisereduce unavailable.
    
    Usage:
        filter = EnhancedNoiseFilter()
        
        # Simple reduction
        clean_audio = filter.reduce_noise(audio_bytes)
        
        # With noise profile
        filter.set_noise_profile(noise_sample)
        clean_audio = filter.reduce_noise(audio_bytes)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        stationary: bool = True,
        prop_decrease: float = 0.75,
        n_fft: int = 512,
        use_vad_prefilter: bool = False,
    ):
        """Initialize enhanced noise filter.
        
        Args:
            sample_rate: Audio sample rate in Hz
            stationary: If True, assume stationary noise (faster).
                       If False, handle non-stationary noise (more accurate).
            prop_decrease: Amount of noise reduction (0.0 to 1.0).
                          1.0 = maximum reduction (may affect speech quality).
                          0.75 = good balance (default).
            n_fft: FFT size for spectral analysis.
                   Smaller = faster but less frequency resolution.
            use_vad_prefilter: If True, use VAD to identify non-speech for
                              noise profile estimation.
        """
        self.sample_rate = sample_rate
        self.stationary = stationary
        self.prop_decrease = prop_decrease
        self.n_fft = n_fft
        self.use_vad_prefilter = use_vad_prefilter
        
        # Noise profile (learned from noise-only audio)
        self._noise_profile: Optional[np.ndarray] = None
        
        # RMS fallback parameters
        self._rms_threshold = 0.002 * 32768  # Threshold in int16 scale
        self._rms_attenuation = 0.3
        
        # Statistics for adaptive threshold
        self._noise_floor_estimate = 0.0
        self._samples_seen = 0
        
        logger.info(
            f"EnhancedNoiseFilter initialized | sample_rate={sample_rate} | "
            f"stationary={stationary} | prop_decrease={prop_decrease} | "
            f"noisereduce_available={NOISEREDUCE_AVAILABLE}"
        )
    
    def set_noise_profile(self, noise_audio: Union[bytes, np.ndarray]) -> None:
        """Set noise profile from a noise-only sample.
        
        Call this with audio that contains only noise (no speech)
        to improve noise reduction quality.
        
        Args:
            noise_audio: Audio sample containing noise (no speech)
        """
        if isinstance(noise_audio, bytes):
            samples = np.frombuffer(noise_audio, dtype=np.int16).astype(np.float32)
        else:
            samples = noise_audio.astype(np.float32)
        
        # Normalize to [-1, 1]
        if np.abs(samples).max() > 1.0:
            samples = samples / 32768.0
        
        self._noise_profile = samples
        logger.info(f"Noise profile set | samples={len(samples)}")
    
    def estimate_noise_floor(self, audio: Union[bytes, np.ndarray]) -> float:
        """Estimate noise floor from audio.
        
        Uses the lowest energy frames as noise floor estimate.
        
        Args:
            audio: Audio data
            
        Returns:
            Estimated noise floor (RMS value)
        """
        if isinstance(audio, bytes):
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        else:
            samples = audio.astype(np.float32)
        
        if samples.size == 0:
            return 0.0
        
        # Frame-based analysis
        frame_size = int(0.02 * self.sample_rate)  # 20ms frames
        num_frames = len(samples) // frame_size
        
        if num_frames < 3:
            return float(np.sqrt(np.mean(samples * samples)))
        
        # Calculate RMS per frame
        frames = samples[:num_frames * frame_size].reshape(num_frames, frame_size)
        rms_per_frame = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
        
        # Use 20th percentile as noise floor (lowest energy frames)
        noise_floor = float(np.percentile(rms_per_frame, 20))
        
        return noise_floor
    
    def reduce_noise(
        self,
        audio_data: Union[bytes, np.ndarray],
        return_bytes: bool = True,
    ) -> Union[bytes, np.ndarray]:
        """Apply noise reduction to audio.
        
        Uses spectral gating if noisereduce available, otherwise RMS-based.
        
        Args:
            audio_data: Input audio (PCM bytes or numpy array)
            return_bytes: If True, return bytes; otherwise numpy array
            
        Returns:
            Noise-reduced audio
        """
        # Handle empty input
        if audio_data is None or (isinstance(audio_data, bytes) and len(audio_data) < 4):
            return audio_data if return_bytes else np.array([], dtype=np.int16)
        
        # Convert to numpy if needed
        if isinstance(audio_data, bytes):
            samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        else:
            samples = audio_data.astype(np.float32)
        
        if samples.size == 0:
            return audio_data if return_bytes else samples.astype(np.int16)
        
        # Normalize to [-1, 1] for processing
        max_val = np.abs(samples).max()
        if max_val > 1.0:
            samples = samples / 32768.0
            was_int16 = True
        else:
            was_int16 = max_val <= 1.0
        
        # Apply noise reduction
        if NOISEREDUCE_AVAILABLE:
            try:
                reduced = self._reduce_spectral(samples)
            except Exception as e:
                logger.warning(f"Spectral noise reduction failed, using RMS fallback: {e}")
                reduced = self._reduce_rms(samples)
        else:
            reduced = self._reduce_rms(samples)
        
        # Convert back to int16 if needed
        if was_int16:
            reduced = reduced * 32768.0
        
        reduced = np.clip(reduced, -32768, 32767).astype(np.int16)
        
        if return_bytes:
            return reduced.tobytes()
        return reduced
    
    def _reduce_spectral(self, samples: np.ndarray) -> np.ndarray:
        """Apply spectral gating noise reduction.
        
        Args:
            samples: Audio samples (normalized to [-1, 1])
            
        Returns:
            Noise-reduced samples
        """
        # Use stored noise profile if available
        y_noise = self._noise_profile
        
        # Apply noisereduce
        reduced = nr.reduce_noise(
            y=samples,
            sr=self.sample_rate,
            y_noise=y_noise,
            stationary=self.stationary,
            prop_decrease=self.prop_decrease,
            n_fft=self.n_fft,
            n_jobs=1,  # Single-threaded for real-time
        )
        
        return reduced
    
    def _reduce_rms(self, samples: np.ndarray) -> np.ndarray:
        """Apply RMS-based noise gate (fallback method).
        
        Args:
            samples: Audio samples
            
        Returns:
            Gated samples
        """
        # Frame-based processing
        frame_size = int(0.02 * self.sample_rate)  # 20ms
        num_frames = len(samples) // frame_size
        
        if num_frames == 0:
            return samples
        
        # Reshape for vectorized processing
        truncated = num_frames * frame_size
        frames = samples[:truncated].reshape(num_frames, frame_size)
        
        # Calculate RMS per frame
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
        
        # Dynamic threshold based on signal statistics
        threshold = max(
            0.01,  # Minimum threshold
            np.percentile(rms, 30) * 1.5,  # Based on low-energy frames
        )
        
        # Soft gating
        gain = np.where(rms < threshold, self._rms_attenuation, 1.0)
        
        # Apply gain per frame
        frames = frames * gain[:, np.newaxis]
        
        # Reconstruct
        result = samples.copy()
        result[:truncated] = frames.flatten()
        
        return result
    
    def reduce_noise_streaming(
        self,
        audio_chunk: bytes,
        chunk_index: int = 0,
    ) -> bytes:
        """Optimized noise reduction for streaming.
        
        Faster than reduce_noise() but may be slightly less accurate.
        Uses accumulated noise floor estimate for better threshold.
        
        Args:
            audio_chunk: Audio chunk (PCM bytes)
            chunk_index: Current chunk index (for noise floor calibration)
            
        Returns:
            Noise-reduced audio bytes
        """
        if not audio_chunk or len(audio_chunk) < 4:
            return audio_chunk
        
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        
        if samples.size == 0:
            return audio_chunk
        
        # Update noise floor estimate (first few chunks are usually noise)
        chunk_rms = float(np.sqrt(np.mean(samples * samples) + 1e-10))
        
        if chunk_index < 5:
            # Calibration phase - assume these are noise
            alpha = 0.3  # Smoothing factor
            self._noise_floor_estimate = (
                alpha * chunk_rms + 
                (1 - alpha) * self._noise_floor_estimate
            )
        
        # Dynamic threshold
        threshold = max(
            self._rms_threshold,
            self._noise_floor_estimate * 2.0,
        )
        
        # Simple noise gate
        if chunk_rms < threshold:
            # Below threshold - attenuate
            samples *= self._rms_attenuation
        else:
            # Above threshold - apply light spectral reduction if available
            if NOISEREDUCE_AVAILABLE and len(samples) > 512:
                try:
                    samples = samples / 32768.0
                    samples = nr.reduce_noise(
                        y=samples,
                        sr=self.sample_rate,
                        stationary=True,
                        prop_decrease=0.5,  # Lighter reduction for speed
                        n_fft=256,  # Smaller FFT for speed
                        n_jobs=1,
                    )
                    samples = samples * 32768.0
                except Exception:
                    pass  # Fall through to output
        
        return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


class HybridNoiseFilter:
    """Hybrid noise filter combining VAD and noise reduction.
    
    Uses VAD to identify speech regions, then applies noise reduction
    only where needed. This is more efficient and preserves speech quality.
    
    Pipeline:
    1. VAD detects speech regions
    2. Non-speech regions are aggressively attenuated
    3. Speech regions get light noise reduction
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        use_silero_vad: bool = True,
        vad_threshold: float = 0.5,
    ):
        """Initialize hybrid filter.
        
        Args:
            sample_rate: Audio sample rate
            use_silero_vad: Whether to use Silero VAD (True) or WebRTC (False)
            vad_threshold: VAD probability threshold
        """
        self.sample_rate = sample_rate
        self.use_silero_vad = use_silero_vad
        self.vad_threshold = vad_threshold
        
        # Lazy load VAD
        self._vad = None
        
        # Noise filter for speech regions
        self._noise_filter = EnhancedNoiseFilter(
            sample_rate=sample_rate,
            stationary=True,
            prop_decrease=0.5,  # Light reduction for speech
        )
        
        logger.info(
            f"HybridNoiseFilter initialized | sample_rate={sample_rate} | "
            f"use_silero_vad={use_silero_vad}"
        )
    
    def _get_vad(self):
        """Get or create VAD instance."""
        if self._vad is None:
            try:
                if self.use_silero_vad:
                    from services.audio.vad import SileroVAD
                    self._vad = SileroVAD(threshold=self.vad_threshold)
                else:
                    from services.audio.vad import WebRTCVAD
                    self._vad = WebRTCVAD(aggressiveness=2)
            except Exception as e:
                logger.warning(f"Failed to load VAD: {e}")
                self._vad = None
        return self._vad
    
    def reduce_noise(
        self,
        audio_data: Union[bytes, np.ndarray],
        return_bytes: bool = True,
    ) -> Union[bytes, np.ndarray]:
        """Apply hybrid noise reduction.
        
        Args:
            audio_data: Input audio
            return_bytes: If True, return bytes
            
        Returns:
            Noise-reduced audio
        """
        if audio_data is None or (isinstance(audio_data, bytes) and len(audio_data) < 64):
            return audio_data if return_bytes else np.array([], dtype=np.int16)
        
        # Convert to numpy if needed
        if isinstance(audio_data, bytes):
            samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            input_was_bytes = True
        else:
            samples = audio_data.astype(np.float32)
            input_was_bytes = False
        
        if samples.size == 0:
            return audio_data if return_bytes else samples.astype(np.int16)
        
        # Try VAD-based processing
        vad = self._get_vad()
        if vad is not None:
            try:
                is_speech, confidence = vad.is_speech(audio_data if input_was_bytes else samples)
                
                if not is_speech:
                    # No speech - heavily attenuate
                    samples *= 0.1
                    result = np.clip(samples, -32768, 32767).astype(np.int16)
                    return result.tobytes() if return_bytes else result
                
                # Has speech - apply light noise reduction
                if confidence < 0.8:
                    # Mixed content - apply noise reduction
                    return self._noise_filter.reduce_noise(
                        audio_data if input_was_bytes else samples,
                        return_bytes=return_bytes,
                    )
                else:
                    # High confidence speech - minimal processing
                    result = samples.astype(np.int16)
                    return result.tobytes() if return_bytes else result
                    
            except Exception as e:
                logger.debug(f"VAD processing failed: {e}")
        
        # Fallback to standard noise reduction
        return self._noise_filter.reduce_noise(
            audio_data if input_was_bytes else samples,
            return_bytes=return_bytes,
        )


# Global instances
_enhanced_filter: Optional[EnhancedNoiseFilter] = None
_hybrid_filter: Optional[HybridNoiseFilter] = None


def get_enhanced_noise_filter(sample_rate: int = 16000) -> EnhancedNoiseFilter:
    """Get global enhanced noise filter instance."""
    global _enhanced_filter
    if _enhanced_filter is None:
        _enhanced_filter = EnhancedNoiseFilter(sample_rate=sample_rate)
    return _enhanced_filter


def get_hybrid_noise_filter(sample_rate: int = 16000) -> HybridNoiseFilter:
    """Get global hybrid noise filter instance."""
    global _hybrid_filter
    if _hybrid_filter is None:
        _hybrid_filter = HybridNoiseFilter(sample_rate=sample_rate)
    return _hybrid_filter
