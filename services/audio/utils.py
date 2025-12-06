"""Audio processing utilities - Noise filtering and audio transformations."""

from __future__ import annotations

import io
import wave
from typing import Tuple

import numpy as np


class NoiseFilter:
    """Minimal noise filter optimized for Azure Speech quality.
    
    CRITICAL: Azure Speech đã có noise suppression tốt, nên filter này
    chỉ làm nhẹ nhàng để tránh làm hỏng audio đầu vào.
    
    - KHÔNG apply gain normalization (làm hỏng dynamic range)
    - Chỉ attenuate VERY low energy frames (< 50 RMS)
    - Giữ nguyên speech quality cho Azure Speech xử lý
    """

    def __init__(self, sample_rate: int = 16000, voice_threshold: float = 0.002):
        """
        Initialize noise filter.

        Args:
            sample_rate: Audio sample rate in Hz
            voice_threshold: RMS threshold for voice detection (used as a floor)
        """
        self.sample_rate = sample_rate
        self.voice_threshold = voice_threshold
        # Pre-calculated constants for faster processing
        self._frame_size = max(1, int(0.02 * sample_rate))  # 20ms frames
        # VERY low threshold - chỉ filter tiếng ồn rất nhỏ
        self._threshold_scale = 50.0  # Fixed low threshold in int16 scale
        self._atten = 0.5  # Gentler attenuation (was 0.3)

    def reduce_noise(self, audio_data: bytes) -> bytes:
        """
        Apply MINIMAL noise reduction - let Azure Speech handle the rest.
        
        Azure Speech đã có noise suppression tốt, nên chỉ filter
        những frame THỰC SỰ là noise (energy rất thấp).

        Args:
            audio_data: Raw PCM audio bytes (int16)

        Returns:
            Lightly filtered audio bytes (int16)
        """
        # Fast path for empty data
        if not audio_data or len(audio_data) < 4:
            return audio_data
        
        # Convert to float32 samples in int16 scale
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return audio_data

        # For short audio, return as-is (don't damage it)
        frame_size = self._frame_size
        num_frames = samples.size // frame_size
        
        if num_frames == 0:
            return audio_data
        
        # Reshape for vectorized RMS calculation
        truncated_size = num_frames * frame_size
        frames = samples[:truncated_size].reshape(num_frames, frame_size)
        
        # Vectorized RMS calculation (much faster than loop)
        rms_per_frame = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
        
        # VERY conservative threshold - only filter obvious noise
        # Use fixed low threshold, NOT adaptive (to preserve speech)
        threshold = self._threshold_scale  # Fixed 50.0
        
        # Only attenuate frames with VERY low energy (obvious silence/noise)
        low_energy_mask = rms_per_frame < threshold
        frames[low_energy_mask] *= self._atten
        
        # Rebuild samples
        samples[:truncated_size] = frames.flatten()
        
        # NO gain normalization - let Azure Speech handle dynamic range
        # This was causing audio distortion and STT errors!
        
        return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


class AudioQualityAnalyzer:
    """Analyze audio quality metrics."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio quality analyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
    
    def _sliding_frames(
        self,
        signal: np.ndarray,
        frame_len: int,
        hop: int,
    ) -> Tuple[np.ndarray, int]:
        """Create sliding window frames from signal."""
        if signal.size < frame_len:
            pad = np.zeros(frame_len - signal.size, dtype=signal.dtype)
            signal = np.concatenate([signal, pad])
        
        num_frames = 1 + max(0, (len(signal) - frame_len) // hop)
        frames = np.lib.stride_tricks.as_strided(
            signal,
            shape=(num_frames, frame_len),
            strides=(signal.strides[0] * hop, signal.strides[0]),
            writeable=False,
        )
        return frames.copy(), num_frames
    
    def estimate_quality(
        self,
        signal: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Estimate audio quality metrics.
        
        Args:
            signal: Normalized audio signal (float32, [-1, 1])
        
        Returns:
            Tuple of (rms, voiced_ratio, snr_db, clipping_ratio)
        """
        # RMS calculation
        rms = float(np.sqrt(np.mean(signal * signal)) + 1e-10)
        
        # Frame-based analysis
        frame_len = int(0.02 * self.sample_rate)  # 20ms frames
        hop = int(0.01 * self.sample_rate)  # 10ms hop
        frames, count = self._sliding_frames(signal, frame_len, hop)
        
        # Energy per frame
        energy = np.mean(frames * frames, axis=1) + 1e-10
        
        # Estimate noise floor and SNR
        sorted_energy = np.sort(energy)
        noise_floor = float(np.mean(sorted_energy[: max(1, int(0.2 * len(sorted_energy)))]))
        speech_energy = float(np.mean(sorted_energy[int(0.8 * len(sorted_energy)):]))
        
        if noise_floor <= 0:
            noise_floor = 1e-10
        
        snr_db = float(10.0 * np.log10(max(speech_energy, noise_floor) / noise_floor))
        
        # Voiced ratio (frames above energy threshold)
        # Slightly more permissive to avoid false "quiet" flags on real-world mics
        energy_threshold = max(noise_floor * 2.0, np.percentile(energy, 50))
        voiced_ratio = float(np.mean(energy > energy_threshold)) if count > 0 else 0.0
        
        # Clipping ratio
        clipping_ratio = float(np.mean(np.abs(signal) >= 0.98))
        
        return rms, voiced_ratio, snr_db, clipping_ratio


def bytes_to_mono(
    audio_bytes: bytes,
    target_sample_rate: int = 16000,
) -> Tuple[np.ndarray, int]:
    """
    Convert audio bytes (WAV or raw PCM) to mono signal.
    
    Args:
        audio_bytes: Audio data (WAV or raw PCM)
        target_sample_rate: Target sample rate for resampling
    
    Returns:
        Tuple of (mono_signal, original_sample_rate)
    """
    # Check if WAV format
    if audio_bytes.startswith(b"RIFF"):
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
    else:
        # Assume raw PCM 16kHz mono
        sample_rate = target_sample_rate
        channels = 1
        frames = audio_bytes
    
    # Convert to float32
    raw = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    
    # Convert to mono if stereo
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1)
    
    return raw, sample_rate


def resample_audio(
    signal: np.ndarray,
    src_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio signal to target sample rate.
    
    Args:
        signal: Input audio signal
        src_sr: Source sample rate
        target_sr: Target sample rate
    
    Returns:
        Resampled signal
    """
    if src_sr == target_sr or signal.size == 0:
        return signal
    
    ratio = target_sr / float(src_sr)
    idx = (np.arange(int(len(signal) * ratio)) / ratio).astype(np.int64)
    idx = np.clip(idx, 0, len(signal) - 1)
    return signal[idx]


def pcm_to_wav(pcm: bytes, sample_rate: int = 16000, sample_width: int = 2) -> bytes:
    """
    Convert raw PCM to WAV format.
    
    Args:
        pcm: Raw PCM bytes
        sample_rate: Sample rate in Hz
        sample_width: Sample width in bytes (2 for 16-bit)
    
    Returns:
        WAV file bytes
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buffer.getvalue()


def calculate_zero_crossing_rate(audio_bytes: bytes) -> float:
    """
    Calculate Zero Crossing Rate (ZCR) for acoustic change detection.
    
    Higher ZCR indicates different speaker or acoustic properties.
    
    Args:
        audio_bytes: Raw PCM audio bytes (int16)
    
    Returns:
        ZCR value (0.0 to 1.0)
    
    Examples:
        >>> # Pure tone has low ZCR
        >>> tone = (np.sin(2 * np.pi * 100 * np.arange(1000) / 16000) * 10000).astype(np.int16).tobytes()
        >>> zcr = calculate_zero_crossing_rate(tone)
        >>> 0.0 < zcr < 0.2
        True
    """
    if not audio_bytes or len(audio_bytes) < 4:
        return 0.0
    
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if samples.size < 2:
        return 0.0
    
    # Count sign changes
    signs = np.sign(samples)
    sign_changes = np.diff(signs) != 0
    zcr = float(np.sum(sign_changes)) / max(1, len(samples) - 1)
    
    return zcr


def detect_energy_spike(
    current_audio: bytes,
    previous_audio: bytes,
    spike_threshold: float = 1.5,
    sample_rate: int = 16000,
) -> bool:
    """
    Detect energy spike that may indicate speaker interruption.
    
    Args:
        current_audio: Current audio frame (PCM bytes)
        previous_audio: Previous audio frame (PCM bytes)
        spike_threshold: RMS ratio threshold (default 1.5 = 50% increase)
        sample_rate: Audio sample rate
    
    Returns:
        True if energy spike detected
    
    Example:
        >>> # Person A speaking quietly, Person B interrupts loudly
        >>> is_interruption = detect_energy_spike(loud_audio, quiet_audio, 1.5)
        >>> # True if current RMS > previous RMS * 1.5
    """
    if not current_audio or not previous_audio:
        return False
    
    # Convert to numpy arrays
    current = np.frombuffer(current_audio, dtype=np.int16).astype(np.float32)
    previous = np.frombuffer(previous_audio, dtype=np.int16).astype(np.float32)
    
    if current.size == 0 or previous.size == 0:
        return False
    
    # Calculate RMS for both
    current_rms = float(np.sqrt(np.mean(current * current) + 1e-10))
    previous_rms = float(np.sqrt(np.mean(previous * previous) + 1e-10))
    
    # Avoid division by zero
    if previous_rms < 1.0:
        previous_rms = 1.0
    
    # Check ratio
    ratio = current_rms / previous_rms
    
    # Also require absolute threshold (avoid false positives on silence)
    min_abs_rms = 500.0  # Minimum absolute RMS to consider
    
    return ratio >= spike_threshold and current_rms >= min_abs_rms


def calculate_rms(audio_bytes: bytes) -> float:
    """
    Calculate RMS (Root Mean Square) energy of audio.
    
    Args:
        audio_bytes: PCM audio bytes
    
    Returns:
        RMS value
    """
    if not audio_bytes:
        return 0.0
    
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    
    return float(np.sqrt(np.mean(samples * samples) + 1e-10))


def detect_acoustic_change(
    current_audio: bytes,
    previous_audio: bytes,
    change_threshold: float = 0.3,
    sample_rate: int = 16000,
) -> bool:
    """
    Detect acoustic change (pitch/timbre shift) indicating speaker change.
    
    Uses zero-crossing rate as a simple acoustic feature.
    
    Args:
        current_audio: Current audio frame
        previous_audio: Previous audio frame
        change_threshold: Relative change threshold (0-1)
        sample_rate: Audio sample rate
    
    Returns:
        True if significant acoustic change detected
    """
    if not current_audio or not previous_audio:
        return False
    
    current = np.frombuffer(current_audio, dtype=np.int16).astype(np.float32)
    previous = np.frombuffer(previous_audio, dtype=np.int16).astype(np.float32)
    
    if current.size == 0 or previous.size == 0:
        return False
    
    # Calculate zero-crossing rate (simple pitch indicator)
    def zcr(signal: np.ndarray) -> float:
        if signal.size < 2:
            return 0.0
        signs = np.sign(signal)
        crossings = np.sum(np.abs(np.diff(signs))) / 2.0
        return float(crossings / signal.size)
    
    current_zcr = zcr(current)
    previous_zcr = zcr(previous)
    
    # Avoid division by zero
    if previous_zcr < 1e-6:
        return False
    
    # Calculate relative change
    change = abs(current_zcr - previous_zcr) / previous_zcr
    
    return change >= change_threshold
