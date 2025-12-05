"""Score Normalization for Speaker Verification/Identification.

Implements various normalization techniques to make similarity scores more robust
and comparable across different conditions:

1. **Z-Norm (Zero Normalization)**: Normalize using impostor cohort statistics
2. **T-Norm (Test Normalization)**: Normalize using test utterance against cohort
3. **S-Norm (Symmetric Normalization)**: Combines Z-Norm and T-Norm

Why S-Norm?
- Raw cosine scores are affected by speaker characteristics
- With small cohorts (1-2 speakers), z-score is unreliable
- S-Norm provides stable, calibrated scores regardless of session size

Reference:
- Auckenthaler et al., "Score Normalization for Text-Independent Speaker Verification Systems"
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# Default imposter cohort size
DEFAULT_COHORT_SIZE = 100


@dataclass
class CohortStats:
    """Statistics for a cohort of imposters."""
    mean: float
    std: float
    count: int
    
    def normalize(self, score: float) -> float:
        """Apply Z-normalization to a score."""
        if self.std < 1e-6:
            return score - self.mean
        return (score - self.mean) / self.std


@dataclass
class ImposterCohort:
    """Imposter cohort for score normalization.
    
    Contains embeddings from diverse speakers used as reference
    for normalizing similarity scores.
    """
    embeddings: np.ndarray  # Shape: (N, dim)
    speaker_ids: List[str] = field(default_factory=list)
    embedding_dim: int = 256
    
    # Precomputed statistics per enrolled speaker
    _speaker_stats: Dict[str, CohortStats] = field(default_factory=dict)
    
    @classmethod
    def create_random(cls, dim: int, size: int = DEFAULT_COHORT_SIZE) -> "ImposterCohort":
        """Create a random cohort for testing.
        
        In production, this should be replaced with real speaker embeddings.
        """
        # Generate random unit vectors (simulating diverse speakers)
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((size, dim)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        speaker_ids = [f"imposter_{i}" for i in range(size)]
        
        return cls(embeddings=embeddings, speaker_ids=speaker_ids, embedding_dim=dim)
    
    @classmethod
    def from_embeddings(cls, embeddings: List[np.ndarray], speaker_ids: Optional[List[str]] = None) -> "ImposterCohort":
        """Create cohort from list of embeddings."""
        if not embeddings:
            raise ValueError("Cannot create cohort from empty embeddings")
        
        matrix = np.stack(embeddings, axis=0).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / (norms + 1e-8)
        
        dim = matrix.shape[1]
        
        if speaker_ids is None:
            speaker_ids = [f"speaker_{i}" for i in range(len(embeddings))]
        
        return cls(embeddings=matrix, speaker_ids=speaker_ids, embedding_dim=dim)
    
    def compute_scores(self, probe: np.ndarray) -> np.ndarray:
        """Compute similarity scores between probe and all cohort members.
        
        Args:
            probe: Query embedding (dim,) - should be L2-normalized
            
        Returns:
            Array of cosine similarities (N,)
        """
        probe_norm = probe.reshape(-1) / (np.linalg.norm(probe) + 1e-8)
        scores = self.embeddings @ probe_norm
        return np.clip(scores, -1.0, 1.0)
    
    def get_stats(self, probe: np.ndarray) -> CohortStats:
        """Get normalization statistics for a probe against this cohort.
        
        This is the T-Norm statistics (test-dependent).
        """
        scores = self.compute_scores(probe)
        return CohortStats(
            mean=float(np.mean(scores)),
            std=float(np.std(scores)),
            count=len(scores),
        )
    
    def precompute_speaker_stats(self, speaker_id: str, speaker_embedding: np.ndarray) -> CohortStats:
        """Precompute Z-Norm statistics for a speaker.
        
        Call this once when a speaker enrolls to cache their cohort statistics.
        
        Args:
            speaker_id: Unique speaker identifier
            speaker_embedding: Speaker's mean embedding (normalized)
            
        Returns:
            CohortStats for this speaker against the imposter cohort
        """
        scores = self.compute_scores(speaker_embedding)
        stats = CohortStats(
            mean=float(np.mean(scores)),
            std=float(np.std(scores)),
            count=len(scores),
        )
        self._speaker_stats[speaker_id] = stats
        return stats
    
    def get_speaker_stats(self, speaker_id: str) -> Optional[CohortStats]:
        """Get precomputed Z-Norm statistics for a speaker."""
        return self._speaker_stats.get(speaker_id)


class ScoreNormalizer:
    """Score normalizer using S-Norm (Symmetric Normalization).
    
    S-Norm combines:
    - Z-Norm: Normalize score using enrolled speaker's statistics
    - T-Norm: Normalize score using test utterance's statistics
    
    Formula:
        s_znorm = (s - μ_z) / σ_z
        s_tnorm = (s - μ_t) / σ_t
        s_snorm = (s_znorm + s_tnorm) / 2
    
    Benefits:
    - Robust to channel/session variations
    - Works well with small cohorts (1-2 speakers)
    - Provides calibrated, comparable scores
    """
    
    # Class-level cohort (shared across instances)
    _cohort: Optional[ImposterCohort] = None
    _cohort_lock = threading.Lock()
    
    def __init__(
        self,
        embedding_dim: int = 256,
        cohort: Optional[ImposterCohort] = None,
        use_adaptive_snorm: bool = True,
        fallback_std: float = 0.15,  # Fallback std when cohort is too small
    ):
        """Initialize score normalizer.
        
        Args:
            embedding_dim: Dimension of embeddings
            cohort: Pre-built imposter cohort (will create default if None)
            use_adaptive_snorm: Whether to use adaptive S-Norm
            fallback_std: Fallback standard deviation for normalization
        """
        self.embedding_dim = embedding_dim
        self.use_adaptive_snorm = use_adaptive_snorm
        self.fallback_std = fallback_std
        
        # Initialize or use provided cohort
        if cohort is not None:
            self._cohort = cohort
        elif ScoreNormalizer._cohort is None:
            with ScoreNormalizer._cohort_lock:
                if ScoreNormalizer._cohort is None:
                    # Create default random cohort
                    ScoreNormalizer._cohort = ImposterCohort.create_random(
                        dim=embedding_dim,
                        size=DEFAULT_COHORT_SIZE,
                    )
    
    @property
    def cohort(self) -> ImposterCohort:
        """Get the imposter cohort."""
        return ScoreNormalizer._cohort or ImposterCohort.create_random(self.embedding_dim)
    
    def z_normalize(
        self,
        raw_score: float,
        speaker_embedding: np.ndarray,
        speaker_id: Optional[str] = None,
    ) -> Tuple[float, CohortStats]:
        """Apply Z-Normalization (speaker-dependent).
        
        Normalizes score based on how the enrolled speaker compares to imposters.
        
        Args:
            raw_score: Raw cosine similarity score
            speaker_embedding: Enrolled speaker's mean embedding
            speaker_id: Optional speaker ID for caching
            
        Returns:
            Tuple of (normalized_score, stats)
        """
        # Try to get cached stats
        stats = None
        if speaker_id:
            stats = self.cohort.get_speaker_stats(speaker_id)
        
        if stats is None:
            # Compute on-the-fly
            stats = self.cohort.get_stats(speaker_embedding)
            if speaker_id:
                self.cohort._speaker_stats[speaker_id] = stats
        
        z_score = stats.normalize(raw_score)
        return z_score, stats
    
    def t_normalize(
        self,
        raw_score: float,
        probe_embedding: np.ndarray,
    ) -> Tuple[float, CohortStats]:
        """Apply T-Normalization (test-dependent).
        
        Normalizes score based on how the test utterance compares to imposters.
        
        Args:
            raw_score: Raw cosine similarity score
            probe_embedding: Test utterance embedding
            
        Returns:
            Tuple of (normalized_score, stats)
        """
        stats = self.cohort.get_stats(probe_embedding)
        t_score = stats.normalize(raw_score)
        return t_score, stats
    
    def s_normalize(
        self,
        raw_score: float,
        probe_embedding: np.ndarray,
        speaker_embedding: np.ndarray,
        speaker_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply S-Normalization (symmetric).
        
        Combines Z-Norm and T-Norm for robust score calibration.
        
        Args:
            raw_score: Raw cosine similarity score
            probe_embedding: Test utterance embedding
            speaker_embedding: Enrolled speaker's mean embedding
            speaker_id: Optional speaker ID for caching
            
        Returns:
            Dictionary with normalized scores and statistics
        """
        # Z-Norm (speaker-dependent)
        z_score, z_stats = self.z_normalize(raw_score, speaker_embedding, speaker_id)
        
        # T-Norm (test-dependent)
        t_score, t_stats = self.t_normalize(raw_score, probe_embedding)
        
        # S-Norm (symmetric average)
        s_score = (z_score + t_score) / 2.0
        
        return {
            "raw_score": raw_score,
            "z_score": z_score,
            "t_score": t_score,
            "s_score": s_score,
            "z_mean": z_stats.mean,
            "z_std": z_stats.std,
            "t_mean": t_stats.mean,
            "t_std": t_stats.std,
        }
    
    def normalize_candidates(
        self,
        probe_embedding: np.ndarray,
        candidates: List[Dict[str, Any]],
        speaker_embeddings: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Normalize scores for all candidates using S-Norm.
        
        Args:
            probe_embedding: Test utterance embedding
            candidates: List of candidate dicts with 'cosine' and 'id' keys
            speaker_embeddings: Dict mapping speaker_id -> mean_embedding
            
        Returns:
            Candidates with added normalized scores
        """
        # Get T-Norm stats once (same for all candidates)
        t_stats = self.cohort.get_stats(probe_embedding)
        
        for candidate in candidates:
            speaker_id = candidate.get("id") or candidate.get("user_id")
            raw_score = candidate.get("cosine", 0.0)
            
            # Get speaker embedding
            speaker_emb = speaker_embeddings.get(speaker_id)
            if speaker_emb is None:
                # Fallback: use raw score
                candidate["s_score"] = raw_score
                candidate["z_score"] = raw_score
                candidate["t_score"] = raw_score
                continue
            
            # Z-Norm
            z_stats = self.cohort.get_speaker_stats(speaker_id)
            if z_stats is None:
                z_stats = self.cohort.precompute_speaker_stats(speaker_id, speaker_emb)
            
            z_score = z_stats.normalize(raw_score)
            t_score = t_stats.normalize(raw_score)
            s_score = (z_score + t_score) / 2.0
            
            candidate["z_score"] = z_score
            candidate["t_score"] = t_score
            candidate["s_score"] = s_score
            candidate["z_mean"] = z_stats.mean
            candidate["z_std"] = z_stats.std
        
        return candidates
    
    def adaptive_accept(
        self,
        candidate: Dict[str, Any],
        cohort_size: int,
        raw_threshold: float = 0.45,
        snorm_threshold: float = 1.5,
    ) -> Tuple[bool, List[str]]:
        """Make adaptive acceptance decision based on cohort size.
        
        When cohort is small (1-2 speakers), relies more on S-Norm.
        When cohort is large (3+), uses combination of raw and normalized scores.
        
        Args:
            candidate: Candidate dict with scores
            cohort_size: Number of speakers in current session
            raw_threshold: Threshold for raw cosine score
            snorm_threshold: Threshold for S-Norm score (in std units)
            
        Returns:
            Tuple of (accepted, reasons)
        """
        reasons = []
        raw_score = candidate.get("cosine", 0.0)
        s_score = candidate.get("s_score")
        
        # Always check raw score minimum
        if raw_score < raw_threshold:
            reasons.append(f"raw_score_low ({raw_score:.3f} < {raw_threshold})")
            return False, reasons
        
        # For small cohorts, rely heavily on S-Norm
        if cohort_size <= 2 and s_score is not None:
            if s_score < snorm_threshold:
                reasons.append(f"snorm_low ({s_score:.2f} < {snorm_threshold})")
                return False, reasons
            return True, []
        
        # For larger cohorts, use combination
        if cohort_size >= 3:
            # Use traditional z-score if available
            z_score = candidate.get("zscore")
            if z_score is not None and z_score < 1.8:
                reasons.append(f"zscore_low ({z_score:.2f} < 1.8)")
                return False, reasons
        
        # Default to S-Norm check
        if s_score is not None and s_score < snorm_threshold * 0.8:
            reasons.append(f"snorm_marginal ({s_score:.2f})")
            return False, reasons
        
        return True, []


# Global normalizer instance (lazy initialization)
_normalizer: Optional[ScoreNormalizer] = None


def get_score_normalizer(embedding_dim: int = 256) -> ScoreNormalizer:
    """Get or create global score normalizer.
    
    Args:
        embedding_dim: Embedding dimension
        
    Returns:
        ScoreNormalizer instance
    """
    global _normalizer
    if _normalizer is None:
        _normalizer = ScoreNormalizer(embedding_dim=embedding_dim)
    return _normalizer


def load_imposter_cohort_from_file(path: Path) -> Optional[ImposterCohort]:
    """Load imposter cohort from numpy file.
    
    Expected format: .npz file with:
    - 'embeddings': (N, dim) array
    - 'speaker_ids': (N,) array of strings (optional)
    
    Args:
        path: Path to .npz file
        
    Returns:
        ImposterCohort or None if failed
    """
    try:
        data = np.load(path, allow_pickle=True)
        embeddings = data['embeddings']
        speaker_ids = data.get('speaker_ids', None)
        
        if speaker_ids is not None:
            speaker_ids = speaker_ids.tolist()
        
        cohort = ImposterCohort.from_embeddings(
            embeddings=[embeddings[i] for i in range(len(embeddings))],
            speaker_ids=speaker_ids,
        )
        
        logger.info(f"Loaded imposter cohort from {path} | size={len(embeddings)}")
        return cohort
        
    except Exception as e:
        logger.warning(f"Failed to load imposter cohort from {path}: {e}")
        return None


def save_imposter_cohort_to_file(cohort: ImposterCohort, path: Path) -> bool:
    """Save imposter cohort to numpy file.
    
    Args:
        cohort: Cohort to save
        path: Output path (.npz)
        
    Returns:
        True if successful
    """
    try:
        np.savez(
            path,
            embeddings=cohort.embeddings,
            speaker_ids=np.array(cohort.speaker_ids),
        )
        logger.info(f"Saved imposter cohort to {path} | size={len(cohort.embeddings)}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save imposter cohort to {path}: {e}")
        return False
