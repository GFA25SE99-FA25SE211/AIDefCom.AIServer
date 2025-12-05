"""Voice processing and speaker identification modules.

Contains:
- speaker_tracker: Multi-speaker tracking
- speaker_identifier: Speaker identification from audio
- score_normalizer: S-Norm score normalization
- vector_index: FAISS-based vector similarity search
- embedding_model: Speaker embedding extraction (ML model)
"""

from services.voice.speaker_tracker import MultiSpeakerTracker, SpeakerSegment
from services.voice.speaker_identifier import (
    SpeakerIdentificationManager,
    SpeakerIdentificationConfig,
    IdentificationResult,
    normalize_speaker_label,
)
from services.voice.score_normalizer import ScoreNormalizer, ImposterCohort
from services.voice.vector_index import (
    VectorIndex,
    FAISSIndex,
    NumpyFallbackIndex,
    SessionVectorIndex,
    create_vector_index,
)
from services.voice.embedding_model import EmbeddingModel, EmbeddingModelRepository

__all__ = [
    # Speaker tracking
    "MultiSpeakerTracker",
    "SpeakerSegment",
    # Speaker identification
    "SpeakerIdentificationManager",
    "SpeakerIdentificationConfig",
    "IdentificationResult",
    "normalize_speaker_label",
    # Score normalization
    "ScoreNormalizer",
    "ImposterCohort",
    # Vector index
    "VectorIndex",
    "FAISSIndex",
    "NumpyFallbackIndex",
    "SessionVectorIndex",
    "create_vector_index",
    # Embedding model
    "EmbeddingModel",
    "EmbeddingModelRepository",
]
