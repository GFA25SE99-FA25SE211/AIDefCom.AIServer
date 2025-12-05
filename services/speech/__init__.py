"""Speech processing modules.

This module contains speech-related utilities:
- recognition_handler: Azure Speech streaming recognition
- text_utils: Text filtering and normalization for Vietnamese
- interruption_detector: Speaker interruption detection
"""

from services.speech.recognition_handler import RecognitionStreamHandler
from services.speech.text_utils import (
    filter_filler_words,
    normalize_vietnamese_text,
    should_log_transcript,
    calculate_speech_confidence,
)
from services.speech.interruption_detector import (
    EmbeddingInterruptionDetector,
    CombinedInterruptionDetector,
    InterruptionEvent,
)

__all__ = [
    # Recognition
    "RecognitionStreamHandler",
    # Text utilities
    "filter_filler_words",
    "normalize_vietnamese_text",
    "should_log_transcript",
    "calculate_speech_confidence",
    # Interruption detection
    "EmbeddingInterruptionDetector",
    "CombinedInterruptionDetector",
    "InterruptionEvent",
]
