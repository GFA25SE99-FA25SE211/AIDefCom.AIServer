"""Audio processing utilities - Noise filtering and audio transformations."""

from __future__ import annotations

import io
import wave
from typing import Tuple

import numpy as np


class NoiseFilter:
    """Simple RMS-based noise gate with soft threshold."""
    
    def __init__(self, sample_rate: int = 16000, voice_threshold: float = 0.002):
        """
        Initialize noise filter.
        
        Args:
            sample_rate: Audio sample rate in Hz
            voice_threshold: RMS threshold for voice detection
        """
        self.sample_rate = sample_rate
        self.voice_threshold = voice_threshold
    
    def reduce_noise(self, audio_data: bytes) -> bytes:
        """
        Apply noise reduction to audio data.
        
        Args:
            audio_data: Raw PCM audio bytes (int16)
        
        Returns:
            Filtered audio bytes (int16)
        """
        # Convert to float32 samples
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return audio_data
        
        # Remove DC offset
        samples -= np.mean(samples)
        samples /= 32768.0  # Normalize to [-1, 1]
        
        # Calculate RMS
        rms = float(np.sqrt(np.mean(samples * samples)) + 1e-8)
        
        # Soft gate: attenuate if below threshold
        gate_threshold = max(self.voice_threshold, 0.002)
        if rms < gate_threshold:
            samples *= 0.65  # Attenuate instead of muting
        
        # Clip and convert back to int16
        samples = np.clip(samples, -1.0, 1.0)
        return (samples * 32768.0).astype(np.int16).tobytes()


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
        energy_threshold = max(noise_floor * 3.0, np.percentile(energy, 60))
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
