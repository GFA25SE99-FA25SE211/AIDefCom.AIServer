"""Audio processing utilities.

This module contains audio-related utilities:
- buffer_manager: Audio buffering and framing
- noise_filter: Enhanced noise reduction
- utils: Audio conversion and analysis utilities
- vad: Voice Activity Detection (Silero, WebRTC)
- streaming_upload: Streaming audio upload helpers
"""

from services.audio.buffer_manager import AudioBufferManager, AudioBufferConfig
from services.audio.noise_filter import EnhancedNoiseFilter, HybridNoiseFilter
from services.audio.vad import SileroVAD, WebRTCVAD, is_speech as vad_is_speech
from services.audio.utils import (
    NoiseFilter,
    AudioQualityAnalyzer,
    pcm_to_wav,
    calculate_rms,
    detect_energy_spike,
    detect_acoustic_change,
)

__all__ = [
    # Buffer management
    "AudioBufferManager",
    "AudioBufferConfig",
    # Noise filtering
    "NoiseFilter",
    "EnhancedNoiseFilter",
    "HybridNoiseFilter",
    # VAD
    "SileroVAD",
    "WebRTCVAD",
    "vad_is_speech",
    # Utilities
    "AudioQualityAnalyzer",
    "pcm_to_wav",
    "calculate_rms",
    "detect_energy_spike",
    "detect_acoustic_change",
]
