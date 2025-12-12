"""Voice Service - Business logic for voice authentication."""

from __future__ import annotations

import gc
import logging
import math
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Protocol

import httpx
import numpy as np

from core.exceptions import VoiceProfileNotFoundError, VoiceAuthenticationError, AudioValidationError
from repositories.interfaces import IVoiceProfileRepository
from services.interfaces.i_voice_service import IVoiceService
from services.audio.utils import (
    AudioQualityAnalyzer,
    NoiseFilter,
    bytes_to_mono,
    resample_audio,
)
from services.voice.score_normalizer import ScoreNormalizer, get_score_normalizer

logger = logging.getLogger(__name__)

# Constants
MIN_ENROLL_SAMPLES = 3
ANGLE_CAP_DEG = 45.0

# Memory limits
MAX_EMBEDDING_CACHE_SIZE = 100  # Max users in embedding cache
MAX_MEAN_CACHE_SIZE = 100  # Max users in mean cache
MAX_SESSION_CACHE_SIZE = 20  # Max sessions in profile cache
CACHE_TTL_SECONDS = 3600  # 1 hour TTL for caches


class LRUCache:
    """Simple LRU cache with TTL and size limit.
    
    Automatically evicts oldest items when size limit is reached.
    Items expire after TTL seconds.
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, None if missing or expired."""
        if key not in self._cache:
            self._misses += 1
            return None
        
        value, timestamp = self._cache[key]
        
        # Check TTL
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with current timestamp."""
        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            old_value = self._cache.pop(oldest_key)
            # Help GC for numpy arrays
            if isinstance(old_value[0], np.ndarray):
                del old_value
        
        self._cache[key] = (value, time.time())
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all items."""
        self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired items. Returns count removed."""
        now = time.time()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
        for k in expired:
            del self._cache[k]
        return len(expired)
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        if key not in self._cache:
            return False
        _, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return False
        return True
    
    def keys(self):
        return self._cache.keys()
    
    def items(self):
        return [(k, v[0]) for k, v in self._cache.items()]
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""
    embedding_dim: int
    model_tag: str
    sample_rate: int
    
    def extract_embedding(self, audio_signal: np.ndarray) -> np.ndarray: ...
    def get_embedding_dim(self) -> int: ...
    def get_model_tag(self) -> str: ...


@dataclass
class QualityThresholds:
    """Audio quality thresholds for enrollment and verification."""
    min_duration: float = 1.5  # Reduced from 2.0s for faster first-utterance identification
    min_enroll_duration: float = 10.0
    rms_floor: float = 0.005  # Very tolerant for streaming audio
    voiced_floor: float = 0.10  # Lower threshold - allow more silence
    snr_floor_db: float = 8.0  # Lower SNR threshold for noisy environments
    clip_ceiling: float = 0.03

class VoiceService(IVoiceService):
    """Voice authentication business logic service."""
    
    def __init__(
        self,
        voice_profile_repo: IVoiceProfileRepository,
        model_repo: EmbeddingModel,
        thresholds: QualityThresholds | None = None,
        azure_blob_repo: Any = None,
        sql_server_repo: Any = None,
    ) -> None:
        """
        Initialize voice service.
        
        Args:
            voice_profile_repo: Interface for voice profile persistence
            model_repo: Repository for embedding model (Pyannote/WeSpeaker)
            thresholds: Quality thresholds for audio validation
            azure_blob_repo: Optional Azure Blob Storage repository
            sql_server_repo: Optional SQL Server repository
        """
        self.profile_repo = voice_profile_repo
        self.model_repo = model_repo
        self.thresholds = thresholds or QualityThresholds()
        self.azure_blob_repo = azure_blob_repo
        self.sql_server_repo = sql_server_repo
        
        # .NET API base URL for user info
        try:
            from app.config import Config as _AppConfig
            self.auth_api_base = getattr(_AppConfig, "AUTH_SERVICE_BASE_URL", "").strip()
            self.auth_verify_ssl = getattr(_AppConfig, "AUTH_SERVICE_VERIFY_SSL", False)
        except Exception:
            self.auth_api_base = ""
            self.auth_verify_ssl = False

        # Audio processing helpers
        self.noise_filter = NoiseFilter()
        self.quality_analyzer = AudioQualityAnalyzer()
        
        # Get model info
        self.embedding_dim = model_repo.get_embedding_dim()
        self.model_tag = model_repo.get_model_tag()
        
        # Thresholds - read from Config for tuning flexibility
        try:
            from app.config import Config as _AppConfig
            self.cosine_threshold = float(getattr(_AppConfig, "VOICE_COSINE_THRESHOLD", 0.50))
            self.speaker_lock_decay = float(getattr(_AppConfig, "VOICE_SPEAKER_LOCK_DECAY_SECONDS", 8.0))
            self.speaker_switch_margin = float(getattr(_AppConfig, "VOICE_SPEAKER_SWITCH_MARGIN", 0.10))
            self.speaker_switch_hits_required = int(getattr(_AppConfig, "VOICE_SPEAKER_SWITCH_HITS_REQUIRED", 4))
        except Exception:
            self.cosine_threshold = 0.50  # Default for production
            self.speaker_lock_decay = 8.0
            self.speaker_switch_margin = 0.10
            self.speaker_switch_hits_required = 4
        
        # Other thresholds (keep hardcoded as they're model-specific)
        self.enrollment_threshold = 0.76
        self.verification_threshold = 0.55
        self.angle_cap = float(np.deg2rad(75))
        self.z_threshold = 1.8
        self.enroll_min_similarity = 0.65
        self.enroll_decay_per_sample = 0.035
        
        # Adaptive gain and relax controls (from Config via environment)
        try:
            from app.config import Config as _AppConfig
            # Adaptive gain controls
            self.gain_target_pctl: int = int(getattr(_AppConfig, "VOICE_GAIN_TARGET_PCTL", 98))
            self.gain_target_peak: float = float(getattr(_AppConfig, "VOICE_GAIN_TARGET_PEAK", 0.80))
            self.gain_max: float = float(getattr(_AppConfig, "VOICE_GAIN_MAX", 10.0))
            self.dynamic_rms_relax: bool = bool(getattr(_AppConfig, "VOICE_DYNAMIC_RMS_RELAX", True))
            # Override threshold dataclass values from environment if present
            self.thresholds.rms_floor = float(getattr(_AppConfig, "VOICE_RMS_FLOOR", self.thresholds.rms_floor))
            self.thresholds.voiced_floor = float(getattr(_AppConfig, "VOICE_VOICED_FLOOR", self.thresholds.voiced_floor))
            self.thresholds.snr_floor_db = float(getattr(_AppConfig, "VOICE_SNR_FLOOR_DB", self.thresholds.snr_floor_db))
            self.thresholds.clip_ceiling = float(getattr(_AppConfig, "VOICE_CLIP_CEILING", self.thresholds.clip_ceiling))
        except Exception:
            # Safe fallbacks
            self.gain_target_pctl = 98
            self.gain_target_peak = 0.80
            self.gain_max = 10.0
            self.dynamic_rms_relax = True
        
        logger.info(
            f"Voice Service initialized | model={self.model_tag} | dim={self.embedding_dim}"
        )
        # Embedding caches for performance with LRU eviction
        self._embedding_cache: LRUCache = LRUCache(
            max_size=MAX_EMBEDDING_CACHE_SIZE, 
            ttl_seconds=CACHE_TTL_SECONDS
        )
        self._mean_cache: LRUCache = LRUCache(
            max_size=MAX_MEAN_CACHE_SIZE,
            ttl_seconds=CACHE_TTL_SECONDS
        )
        self._profiles_preloaded: bool = False
        # Session-specific profile cache with TTL
        self._session_profiles_cache: LRUCache = LRUCache(
            max_size=MAX_SESSION_CACHE_SIZE,
            ttl_seconds=CACHE_TTL_SECONDS
        )
        # Track last cleanup time
        self._last_cleanup_ts: float = time.time()
        self._cleanup_interval: int = 300  # Cleanup every 5 minutes
        
        # Score normalizer for S-Norm (Adaptive Symmetric Normalization)
        self._score_normalizer = get_score_normalizer(embedding_dim=self.embedding_dim)
        # S-Norm thresholds
        self.snorm_threshold = 1.5  # S-Norm score threshold (in std units)
        self.use_snorm = True  # Enable S-Norm by default
    
    def _maybe_cleanup_caches(self) -> None:
        """Periodically cleanup expired cache entries."""
        now = time.time()
        if now - self._last_cleanup_ts < self._cleanup_interval:
            return
        
        self._last_cleanup_ts = now
        expired_emb = self._embedding_cache.cleanup_expired()
        expired_mean = self._mean_cache.cleanup_expired()
        expired_session = self._session_profiles_cache.cleanup_expired()
        
        if expired_emb + expired_mean + expired_session > 0:
            logger.info(
                f"🧹 Cache cleanup: embedding={expired_emb}, mean={expired_mean}, session={expired_session}"
            )
            # Force garbage collection after cleanup
            gc.collect()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for monitoring."""
        return {
            "embedding_cache": self._embedding_cache.stats(),
            "mean_cache": self._mean_cache.stats(),
            "session_cache": {
                "size": len(self._session_profiles_cache),
                "max_size": MAX_SESSION_CACHE_SIZE,
            },
        }
    
    def preload_enrolled_profiles(self) -> int:
        """Preload all enrolled profiles into memory cache.
        
        Call this on startup to avoid cold-start delays during identification.
        Returns the number of profiles preloaded.
        """
        if self._profiles_preloaded:
            return len(self._embedding_cache)
        
        try:
            profiles = self._get_enrolled_profiles_batch()
            count = len(profiles)
            self._profiles_preloaded = True
            logger.info(f"✅ Preloaded {count} enrolled profiles into cache")
            return count
        except Exception as e:
            logger.warning(f"Failed to preload profiles: {e}")
            return 0
    
    def preload_session_profiles(
        self, 
        defense_session_id: str,
        user_ids: List[str],
        user_info_map: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Preload và cache profiles cho một defense session.
        Gọi 1 lần khi bắt đầu session, dùng lại suốt phiên.
        
        Args:
            defense_session_id: ID của defense session
            user_ids: List of user IDs in this session
            user_info_map: Optional dict mapping user_id -> {"name", "role", "display_name"}
                           If provided, profiles will use display_name for speaker identification
        
        Returns:
            List of profiles với embeddings đã normalize, sẵn sàng cho identification
        """
        # Periodically cleanup expired entries
        self._maybe_cleanup_caches()
        
        # Check if already cached
        cache_key = defense_session_id
        cached = self._session_profiles_cache.get(cache_key)
        if cached is not None:
            
            # Update display names if user_info_map provided (may have been fetched after initial cache)
            if user_info_map:
                for profile in cached:
                    user_id = profile.get("user_id")
                    if user_id and user_id in user_info_map:
                        info = user_info_map[user_id]
                        profile["name"] = info.get("display_name") or info.get("name") or profile.get("name")
                        profile["role"] = info.get("role", "")
            
            return cached
        
        # Load profiles for this session
        profiles = self._get_enrolled_profiles_batch(user_ids)
        
        # Override profile names with display_name from user_info_map (if provided)
        if user_info_map:
            logger.info(f"🏷️ Applying display names from user_info_map: {list(user_info_map.keys())}")
            for profile in profiles:
                user_id = profile.get("user_id")
                if user_id and user_id in user_info_map:
                    info = user_info_map[user_id]
                    old_name = profile.get("name")
                    new_name = info.get("display_name") or info.get("name") or profile.get("name")
                    # Use display_name: "Tên (Vai trò)" for speaker identification
                    profile["name"] = new_name
                    profile["role"] = info.get("role", "")
                    logger.info(f"🏷️ Profile {user_id}: '{old_name}' -> '{new_name}' (role={profile['role']})")
                else:
                    logger.warning(f"⚠️ Profile {user_id} not in user_info_map, keeping name='{profile.get('name')}'")
        else:
            logger.warning("⚠️ No user_info_map provided, using profile names as-is")
            for profile in profiles:
                logger.info(f"📋 Profile: user_id={profile.get('user_id')}, name='{profile.get('name')}'")
        
        # Cache for this session
        self._session_profiles_cache.set(cache_key, profiles)
        
        return profiles
    
    def get_session_profiles(self, defense_session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached profiles for a session, or None if not preloaded."""
        return self._session_profiles_cache.get(defense_session_id)
    
    def clear_session_cache(self, defense_session_id: str) -> None:
        """Clear cached profiles for a session (call when session ends)."""
        if self._session_profiles_cache.delete(defense_session_id):
            logger.info(f"🧹 Cleared profile cache for session {defense_session_id}")
            # Force GC to release numpy arrays
            gc.collect()
    
    def _invalidate_user_caches(self, user_id: str) -> None:
        """Invalidate all caches for a user after enrollment/update.
        
        This forces the next identification to reload fresh embeddings from storage.
        """
        # Clear embedding cache
        if self._embedding_cache.delete(user_id):
            logger.info(f"🧹 Invalidated embedding cache for {user_id}")
        
        # Clear mean cache
        if self._mean_cache.delete(user_id):
            logger.info(f"🧹 Invalidated mean cache for {user_id}")
        
        # Note: LRUCache doesn't support iteration by session_id,
        # so we clear all session caches when user re-enrolls
        # This is acceptable as re-enrollment is rare
        
        # Reset preloaded flag so next global load will refresh
        self._profiles_preloaded = False
        logger.info(f"🔄 Reset profiles preloaded flag - will reload on next startup/request")

    def identify_speaker_with_cache(
        self,
        audio_bytes: bytes,
        preloaded_profiles: List[Dict[str, Any]],
        pre_filtered: bool = False,
    ) -> Dict[str, Any]:
        """
        Identify speaker using pre-loaded profiles (no DB/Blob I/O).
        Much faster than identify_speaker() for streaming use cases.
        
        Args:
            audio_bytes: Audio to identify
            preloaded_profiles: Output from preload_session_profiles()
            pre_filtered: True if audio already noise-filtered
            
        Returns:
            Identification result (same format as identify_speaker)
        """
        # Process audio
        try:
            embedding, quality = self._process_audio(audio_bytes, skip_noise_filter=pre_filtered)
        except Exception as e:
            logger.exception("Audio processing failed during identify_with_cache")
            return {
                "type": "identify",
                "success": False,
                "identified": False,
                "speaker_id": None,
                "speaker_name": None,
                "score": 0.0,
                "confidence": 0.0,
                "message": f"Failed to process audio: {e}",
            }
        
        # Check quality
        if not quality.get("ok"):
            return {
                "type": "identify",
                "success": False,
                "identified": False,
                "speaker_id": None,
                "speaker_name": None,
                "score": 0.0,
                "confidence": 0.0,
                "message": quality.get("reason", "Audio quality check failed"),
                "quality": quality,
            }
        
        if not preloaded_profiles:
            return {
                "type": "identify",
                "success": False,
                "identified": False,
                "speaker_id": None,
                "speaker_name": None,
                "score": 0.0,
                "confidence": 0.0,
                "message": "No enrolled profiles in session cache",
                "quality": quality,
            }
        
        # Calculate scores using cached profiles (NO DB/Blob I/O!)
        candidates = self._calculate_candidate_scores(embedding, preloaded_profiles)
        
        if not candidates:
            return {
                "type": "identify",
                "success": False,
                "identified": False,
                "speaker_id": None,
                "speaker_name": None,
                "score": 0.0,
                "confidence": 0.0,
                "message": "No valid embeddings in cached profiles",
                "quality": quality,
            }
        
        # Sort by cosine similarity
        candidates.sort(key=lambda x: x["cosine"], reverse=True)
        best = candidates[0]
        
        # Calculate z-score for cohort analysis
        cohort_scores = [c["cosine"] for c in candidates[1:]]
        if cohort_scores:
            cohort_mean = float(np.mean(cohort_scores))
            cohort_std = float(np.std(cohort_scores))
            if cohort_std < 1e-5:
                cohort_std = 1e-5
            zscore = (best["cosine"] - cohort_mean) / cohort_std
        else:
            zscore = None
        cohort_size = len(cohort_scores)
        
        # Build speaker embeddings map for S-Norm
        speaker_embeddings = {}
        for profile in preloaded_profiles:
            user_id = profile.get("user_id")
            mean_emb = profile.get("mean_embedding_norm")
            if user_id and mean_emb is not None:
                speaker_embeddings[user_id] = mean_emb
        
        # Decision logic with S-Norm for small cohorts
        accepted, reasons = self._accept_identification(
            best, zscore, cohort_size,
            probe_embedding=embedding,
            speaker_embeddings=speaker_embeddings,
        )
        
        logger.info(
            "🔍 Best candidate (cached): %s | cosine=%.3f | angle=%.1f° | accepted=%s",
            best.get("name"),
            best.get("cosine", 0),
            float(np.rad2deg(best.get("angle", 0))),
            accepted,
        )
        
        # Format candidates for response
        all_candidates = [
            {
                "name": c.get("name"),
                "id": c.get("id"),
                "user_id": c.get("user_id"),
                "cosine": c.get("cosine"),
                "mean_cosine": c.get("mean_cosine"),
                "max_sample_cosine": c.get("max_sample_cosine"),
                "angle_deg": float(np.rad2deg(c.get("angle", 0))),
                "score_source": c.get("score_source", "mean"),
            }
            for c in candidates[:5]
        ]
        
        if accepted:
            confidence = self._score_confidence(best["cosine"])
            return {
                "type": "identify",
                "success": True,
                "identified": True,
                "speaker_id": best.get("id"),
                "speaker": best.get("name"),
                "user_id": best.get("user_id") or best.get("id"),
                "score": best.get("cosine"),
                "confidence": confidence,
                "angular_dist": best.get("angle"),
                "zscore": float(zscore) if zscore is not None else None,
                "quality": quality,
                "candidates": all_candidates,
                "message": "Speaker identified successfully (cached)",
            }
        
        return {
            "type": "identify",
            "success": False,
            "identified": False,
            "speaker_id": None,
            "speaker_name": None,
            "score": best.get("cosine"),
            "confidence": 0.0,
            "angular_dist": best.get("angle"),
            "zscore": float(zscore) if zscore is not None else None,
            "message": "Voice does not match any enrolled user",
            "reasons": reasons,
            "quality": quality,
            "candidates": all_candidates,
        }
    
    def _get_user_from_api(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user info from .NET API."""
        if not self.auth_api_base:
            return None
        try:
            import requests
            url = f"{self.auth_api_base}/api/auth/users/{user_id}"
            resp = requests.get(url, verify=self.auth_verify_ssl, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                # Extract from ApiResponse wrapper if present
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                return data
            logger.warning(f"User API returned {resp.status_code} for user_id={user_id}")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch user info from API | user_id={user_id} | error={e}")
            return None
    
    def _check_voice_uniqueness(
        self, 
        embedding: np.ndarray, 
        exclude_user_id: str,
        similarity_threshold: float = 0.70
    ) -> Dict[str, Any]:
        """
        Check if voice embedding is unique (not too similar to existing enrolled users).
        
        This prevents enrollment of duplicate/similar voices which cause identification confusion.
        
        Args:
            embedding: New voice embedding to check
            exclude_user_id: User ID to exclude from check (the user being enrolled)
            similarity_threshold: Max allowed similarity with other users (default 0.70)
        
        Returns:
            Dict with:
                - unique: True if voice is unique enough
                - similar_to: List of similar user IDs if not unique
                - max_similarity: Highest similarity score found
                - message: Human-readable message
        """
        try:
            # Normalize input embedding
            probe = np.asarray(embedding, dtype=np.float32).reshape(-1)
            probe_norm = probe / (np.linalg.norm(probe) + 1e-6)
            
            # Get all enrolled profiles
            all_profiles = self.profile_repo.list_profiles()
            similar_users = []
            max_sim = 0.0
            max_sim_user = None
            
            for uid in all_profiles:
                if uid == exclude_user_id:
                    continue
                
                try:
                    profile = self.profile_repo.load_profile(uid)
                    if profile.get("enrollment_status") != "enrolled":
                        continue
                    
                    embeddings = self.profile_repo.get_embeddings(uid)
                    if not embeddings or len(embeddings) == 0:
                        continue
                    
                    # Calculate similarity with mean embedding
                    mat = np.array(embeddings, dtype=np.float32)
                    norms = np.linalg.norm(mat, axis=1, keepdims=True)
                    norm_mat = mat / (norms + 1e-6)
                    mean_emb = np.mean(norm_mat, axis=0)
                    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)
                    
                    similarity = float(np.dot(probe_norm, mean_emb))
                    
                    if similarity > max_sim:
                        max_sim = similarity
                        max_sim_user = uid
                    
                    if similarity > similarity_threshold:
                        user_name = profile.get("name", uid)
                        similar_users.append({
                            "user_id": uid,
                            "name": user_name,
                            "similarity": round(similarity, 3),
                        })
                        
                except Exception as e:
                    logger.warning(f"Error checking similarity with {uid}: {e}")
                    continue
            
            if similar_users:
                # Sort by similarity descending
                similar_users.sort(key=lambda x: x["similarity"], reverse=True)
                top_match = similar_users[0]
                return {
                    "unique": False,
                    "similar_to": similar_users,
                    "max_similarity": max_sim,
                    "max_similarity_user": max_sim_user,
                    "message": f"Giọng nói quá giống với user '{top_match['name']}' (similarity: {top_match['similarity']:.0%}). Vui lòng ghi âm lại với giọng tự nhiên của bạn.",
                }
            
            return {
                "unique": True,
                "similar_to": [],
                "max_similarity": max_sim,
                "max_similarity_user": max_sim_user,
                "message": "Voice is unique",
            }
            
        except Exception as e:
            logger.exception(f"Error in voice uniqueness check: {e}")
            # On error, allow enrollment to proceed (fail-open)
            return {
                "unique": True,
                "similar_to": [],
                "max_similarity": 0.0,
                "message": "Uniqueness check skipped due to error",
                "error": str(e),
            }

    def get_or_create_profile(self, user_id: str, name: str | None = None) -> Dict[str, Any]:
        """
        Get existing profile or create new one.
        
        Args:
            user_id: User identifier
            name: User's name
        
        Returns:
            Profile information
        """
        name = name or user_id
        
        if self.profile_repo.profile_exists(user_id):
            try:
                profile = self.profile_repo.load_profile(user_id)
                embeddings = self.profile_repo.get_embeddings(user_id)
                
                logger.info(f"Profile loaded | user_id={user_id}")
                return {
                    "status": "existing",
                    "user_id": user_id,
                    "name": profile.get("name", name),
                    "enrollment_status": profile.get("enrollment_status", "not_enrolled"),
                    "enrollment_count": len(embeddings),
                }
            except VoiceProfileNotFoundError:
                pass
        
        # Create new profile
        profile = self.profile_repo.create_profile(user_id, name, self.embedding_dim)
        logger.info(f"Profile created | user_id={user_id}")
        
        return {
            "status": "created",
            "user_id": user_id,
            "name": name,
            "enrollment_status": "not_enrolled",
            "enrollment_count": 0,
        }
    
    def create_profile(self, user_id: str, name: str | None = None) -> Dict[str, Any]:
        """Create new voice profile."""
        return self.get_or_create_profile(user_id, name or user_id)
    
    def enroll_voice(self, user_id: str, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Enroll voice sample for user (max 3 enrollments).
        
        Args:
            user_id: User identifier
            audio_bytes: Audio data (WAV or PCM)
        
        Returns:
            Enrollment result
        """
        try:
            # Validate input
            if not audio_bytes:
                return {
                    "error": "Empty audio data",
                    "user_id": user_id,
                }
            
            # Fetch user info from .NET API to validate user exists
            user_info = self._get_user_from_api(user_id)
            if user_info is None:
                logger.warning(f"User API unavailable or user not found: {user_id}. Proceeding with enrollment anyway.")
                user_name = user_id  # Fallback to user_id as name
            else:
                # Extract name from API response (could be in "data" wrapper)
                user_name = user_info.get("fullName") or user_info.get("name") or user_id
            
            # Load or auto-create profile
            try:
                profile = self.profile_repo.load_profile(user_id)
            except VoiceProfileNotFoundError:
                # Auto-create profile with name from .NET API
                profile = self.profile_repo.create_profile(user_id, user_name, self.embedding_dim)
            except Exception as e:
                logger.exception(f"Failed to load/create profile for {user_id}")
                return {
                    "error": f"Failed to load or create profile: {str(e)}",
                    "user_id": user_id,
                }
            
            # Check if maximum enrollments reached
            current_count = profile.get("enrollment_count", 0)
            max_count = getattr(self, "max_enrollment_count", 3)
            
            if current_count >= max_count:
                return {
                    "error": f"Maximum enrollment limit reached ({max_count} samples)",
                    "user_id": user_id,
                    "enrollment_count": current_count,
                    "max_enrollment_count": max_count,
                    "completed": True,
                    "can_record_next": False,
                    "next_sample_number": current_count,
                    "message": f"Đã đủ {max_count} samples. Không cần ghi thêm.",
                }
            
            # Process audio
            try:
                embedding, quality = self._process_audio_for_enrollment(audio_bytes)
            except AudioValidationError as e:
                return {
                    "error": str(e),
                    "id": user_id,
                    "quality": {"ok": False, "reason": str(e)},
                }
            except Exception as e:
                logger.exception(f"Audio processing failed | user_id={user_id}")
                return {"error": "Failed to process audio input", "id": user_id, "details": str(e)}
            
            # Check quality
            if not quality["ok"]:
                return {
                    "error": quality.get("reason") or "Audio quality below requirements",
                    "id": user_id,
                    "quality": quality,
                }
            
            # Check consistency with existing samples
            embeddings = self.profile_repo.get_embeddings(user_id)
            consistency = self._check_enrollment_consistency(embedding, embeddings, profile)
            
            if consistency and not consistency["passed"]:
                return {
                    "error": "Voice sample inconsistent with previous samples",
                    "id": user_id,
                    "consistency": consistency,
                    "quality": quality,
                }
            
            # Check voice uniqueness - prevent enrolling voice too similar to existing users
            # Only check on first sample (subsequent samples should be consistent with first)
            current_count = profile.get("enrollment_count", 0)
            if current_count == 0:
                uniqueness = self._check_voice_uniqueness(embedding, user_id)
                if not uniqueness["unique"]:
                    return {
                        "error": uniqueness["message"],
                        "error_code": "VOICE_NOT_UNIQUE",
                        "id": user_id,
                        "uniqueness": uniqueness,
                        "quality": quality,
                    }
            
            # Add sample to profile
            try:
                self.profile_repo.add_voice_sample(
                    user_id,
                    embedding,
                    metrics=quality,
                )
            except Exception as e:
                logger.exception(f"Failed to add voice sample for {user_id}")
                return {
                    "error": f"Failed to save voice sample: {str(e)}",
                    "user_id": user_id,
                }
            
            # Update profile statistics
            try:
                embeddings = self.profile_repo.get_embeddings(user_id)
                self._update_profile_stats(user_id, embeddings)
            except Exception as e:
                logger.warning(f"Failed to update profile stats for {user_id}: {e}")
                # Non-fatal, continue
            
            # CRITICAL: Invalidate caches for this user to force reload with new embeddings
            self._invalidate_user_caches(user_id)
            
            # Update enrollment status
            enrollment_count = len(embeddings)
            if enrollment_count >= MIN_ENROLL_SAMPLES:
                self.profile_repo.update_enrollment_status(user_id, "enrolled")
                enrollment_status = "enrolled"
                remaining = max(0, max_count - enrollment_count)
            else:
                self.profile_repo.update_enrollment_status(user_id, "partial")
                enrollment_status = "partial"
                remaining = max(0, max_count - enrollment_count)
            
            # Upload logic: first sample -> upload & DB update; subsequent samples -> just overwrite blob; final sample -> ensure blob updated
            blob_url = None
            initial_upload = False
            final_upload = False
            db_updated = False
            if self.azure_blob_repo and self.sql_server_repo:
                try:
                    updated_profile = self.profile_repo.load_profile(user_id)
                    should_upload = True  # always keep blob in sync
                    if current_count == 0:
                        # First sample just added (now enrollment_count will be 1 after add)
                        initial_upload = True
                    if enrollment_count == max_count:
                        final_upload = True
                    if should_upload:
                        blob_url = self.azure_blob_repo.upload_voice_profile(user_id, updated_profile)
                        logger.info(
                            "Voice profile uploaded to Azure Blob | user_id=%s | url=%s | initial=%s | final=%s",
                            user_id, blob_url, initial_upload, final_upload,
                        )
                        # Always update DB URL to ensure consistency
                        db_updated = self.sql_server_repo.update_voice_sample_path(user_id, blob_url)
                        if db_updated:
                            logger.info("VoiceSamplePath updated in database | user_id=%s | url=%s", user_id, blob_url)
                        else:
                            logger.warning("Failed to update VoiceSamplePath in database | user_id=%s", user_id)
                except Exception as e:
                    import traceback
                    logger.error(
                        "Failed Azure Blob operation | user_id=%s | error=%s | initial=%s | final=%s | traceback=%s",
                        user_id, e, initial_upload, final_upload, traceback.format_exc(),
                    )
                    # Non-fatal
            
            response = {
                "id": user_id,
                "name": profile.get("name", user_id),
                "enrollment_status": enrollment_status,
                "enrollment_count": enrollment_count,
                "max_enrollment_count": max_count,
                "remaining_samples": remaining,
                "quality": quality,
                "completed": enrollment_count >= max_count,
                "can_record_next": enrollment_count < max_count,
                "next_sample_number": enrollment_count + 1 if enrollment_count < max_count else enrollment_count,
            }
            if enrollment_count >= max_count:
                response["message"] = f"✅ Hoàn tất. Đã đủ {enrollment_count}/{max_count} samples."
            else:
                response["message"] = f"✅ Sample {enrollment_count} OK. Còn {max_count - enrollment_count} sample nữa."
            
            if blob_url:
                response["blob_url"] = blob_url
                response["initial_upload"] = initial_upload
                response["final_upload"] = final_upload
                response["db_updated"] = db_updated
            
            if consistency:
                response["consistency"] = consistency
            
            # Explicit memory cleanup
            del embedding
            del embeddings
            import gc
            gc.collect()
            
            return response
            
        except Exception as e:
            # Catch-all for any unexpected errors
            logger.exception(f"Unexpected error in enroll_voice for user {user_id}")
            return {
                "error": f"Unexpected error during enrollment: {str(e)}",
                "user_id": user_id,
            }
    
    async def get_defense_session_users(self, session_id: str) -> Optional[List[str]]:
        """Fetch list of user IDs from defense session.
        
        Args:
            session_id: Defense session ID
            
        Returns:
            List of user IDs or None if failed
        """
        result = await self.get_defense_session_users_with_info(session_id)
        if result is None:
            return None
        return list(result.keys())
    
    async def get_defense_session_users_with_info(self, session_id: str) -> Optional[Dict[str, Dict[str, str]]]:
        """Fetch user info (id, name, role) from defense session.
        
        Args:
            session_id: Defense session ID
            
        Returns:
            Dict mapping user_id -> {"name": "...", "role": "...", "display_name": "Tên (Vai trò)"}
            or None if failed
        """
        from app.config import Config
        
        if not Config.AUTH_SERVICE_BASE_URL:
            logger.warning("AUTH_SERVICE_BASE_URL not configured, skipping whitelist fetch")
            return None
        
        url = f"{Config.AUTH_SERVICE_BASE_URL}/api/defense-sessions/{session_id}/users"
        
        try:
            logger.info(f"🔍 Fetching defense session users | url={url}")
            
            async with httpx.AsyncClient(
                verify=Config.AUTH_SERVICE_VERIFY_SSL,
                timeout=httpx.Timeout(5.0, connect=3.0)  # 5s total, 3s connect
            ) as client:
                response = await client.get(url)
                
                if response.status_code != 200:
                    logger.warning(
                        f"⚠️ Defense session API returned {response.status_code} | "
                        f"url={url} | body={response.text[:200] if response.text else 'empty'}"
                    )
                    return None
                
                data = response.json()
                logger.info(f"📦 Defense session API response: {str(data)[:500]}")
                
                users_info: Dict[str, Dict[str, str]] = {}
                
                # Extract user info from response
                users_list = []
                if "data" in data and isinstance(data["data"], list):
                    users_list = data["data"]
                elif isinstance(data, list):
                    users_list = data
                
                for user in users_list:
                    if not isinstance(user, dict) or "id" not in user:
                        continue
                    
                    user_id = str(user["id"])
                    # Try to get name from various fields
                    name = (
                        user.get("fullName") or 
                        user.get("name") or 
                        user.get("displayName") or 
                        user.get("userName") or
                        user_id
                    )
                    # Try to get role
                    role = (
                        user.get("role") or 
                        user.get("roleName") or 
                        user.get("roleInSession") or
                        user.get("position") or
                        ""
                    )
                    
                    # Create display name: "Tên (Vai trò)" or just "Tên"
                    if role:
                        display_name = f"{name} ({role})"
                    else:
                        display_name = name
                    
                    users_info[user_id] = {
                        "name": name,
                        "role": role,
                        "display_name": display_name,
                    }
                
                if users_info:
                    return users_info
                
                logger.warning(f"⚠️ No valid users in response: {str(data)[:200]}")
                return None
                
        except httpx.ConnectError as e:
            logger.warning(f"⚠️ Cannot connect to AUTH_SERVICE: {e}")
            return None
        except httpx.TimeoutException as e:
            logger.warning(f"⚠️ Timeout fetching defense session users: {e}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Error fetching defense session users: {e}")
            return None
    
    def identify_speaker(self, audio_bytes: bytes, whitelist_user_ids: Optional[List[str]] = None, pre_filtered: bool = False) -> Dict[str, Any]:
        """
        Identify speaker from audio.
        
        Args:
            audio_bytes: Audio data (WAV or PCM)
            whitelist_user_ids: Optional list of user IDs to filter
            pre_filtered: If True, audio has already been noise-filtered (skip filtering)
        
        Returns:
            Identification result
        """
        # Process audio - skip noise filter if already filtered
        try:
            embedding, quality = self._process_audio(audio_bytes, skip_noise_filter=pre_filtered)
        except Exception as e:
            logger.exception("Audio processing failed during identify")
            return {
                "type": "identify",
                "success": False,
                "identified": False,
                "speaker_id": None,
                "speaker_name": None,
                "score": 0.0,
                "confidence": 0.0,
                "message": "Failed to process audio",
                "error": str(e),
            }
        
        # Check quality
        if not quality["ok"]:
            return {
                "type": "identify",
                "success": False,
                "identified": False,
                "speaker_id": None,
                "speaker_name": None,
                "score": 0.0,
                "confidence": 0.0,
                "message": quality.get("reason") or "Audio quality requirements not met",
                "quality": quality,
            }
        
        # Get enrolled profiles (batch optimized)
        enrolled_profiles = self._get_enrolled_profiles_batch(whitelist_user_ids)
        logger.info(f"🔍 Identify: Using {len(enrolled_profiles)} enrolled profiles"
        )
        
        if not enrolled_profiles:
            return {
                "type": "identify",
                "success": False,
                "identified": False,
                "speaker_id": None,
                "speaker_name": None,
                "score": 0.0,
                "confidence": 0.0,
                "message": "No enrolled users found. Please enroll at least one user with 3 samples.",
                "quality": quality,
            }
        
        # Calculate scores for all candidates
        candidates = self._calculate_candidate_scores(embedding, enrolled_profiles)
        logger.info(f"🔍 Identify: {len(candidates)} candidates scored")
        
        if not candidates:
            return {
                "type": "identify",
                "success": False,
                "identified": False,
                "speaker_id": None,
                "speaker_name": None,
                "score": 0.0,
                "confidence": 0.0,
                "message": "No valid embeddings",
                "quality": quality,
            }
        
        # Sort by cosine similarity
        candidates.sort(key=lambda x: x["cosine"], reverse=True)
        best = candidates[0]
        
        # Calculate z-score for cohort analysis
        cohort_scores = [c["cosine"] for c in candidates[1:]]
        if cohort_scores:
            cohort_mean = float(np.mean(cohort_scores))
            cohort_std = float(np.std(cohort_scores))
            if cohort_std < 1e-5:
                cohort_std = 1e-5
            zscore = (best["cosine"] - cohort_mean) / cohort_std
        else:
            cohort_mean = 0.0
            cohort_std = 0.0
            zscore = None
        cohort_size = len(cohort_scores)
        
        # Build speaker embeddings map for S-Norm
        speaker_embeddings = {}
        for profile in enrolled_profiles:
            user_id = profile.get("user_id")
            mean_emb = profile.get("mean_embedding_norm")
            if user_id and mean_emb is not None:
                speaker_embeddings[user_id] = mean_emb
        
        # Decision logic with S-Norm for small cohorts
        accepted, reasons = self._accept_identification(
            best, zscore, cohort_size,
            probe_embedding=embedding,
            speaker_embeddings=speaker_embeddings,
        )
        logger.info(
            "🔍 Best candidate: %s | cosine=%.3f | angle=%.1f° | source=%s | zscore=%s | accepted=%s",
            best["name"],
            best["cosine"],
            np.rad2deg(best["angle"]),
            best.get("score_source", "mean"),
            f"{zscore:.2f}" if zscore is not None else "N/A",
            accepted,
        )
        
        # Format candidates
        all_candidates = [
            {
                "name": c["name"],
                "id": c["id"],
                "cosine": c["cosine"],
                "mean_cosine": c.get("mean_cosine"),
                "max_sample_cosine": c.get("max_sample_cosine"),
                "angle_deg": float(np.rad2deg(c["angle"])),
                "score_source": c.get("score_source", "mean"),
            }
            for c in candidates[:3]
        ]
        
        if accepted:
            confidence = self._score_confidence(best["cosine"])
            return {
                "type": "identify",
                "success": True,
                "identified": True,
                "speaker_id": best["id"],
                "speaker_name": best["name"],
                "score": best["cosine"],
                "confidence": confidence,
                "angular_dist": best["angle"],
                "zscore": float(zscore) if zscore is not None else None,
                "quality": quality,
                "candidates": all_candidates,
                "message": "Speaker identified successfully",
            }

        return {
            "type": "identify",
            "success": False,
            "identified": False,
            "speaker_id": None,
            "speaker_name": None,
            "score": best["cosine"],
            "confidence": 0.0,
            "angular_dist": best["angle"],
            "zscore": float(zscore) if zscore is not None else None,
            "message": "Voice does not match any enrolled user",
            "reasons": reasons,
            "quality": quality,
            "candidates": all_candidates,
        }
    
    def verify_voice(self, user_id: str, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Verify if audio matches user's voice profile.
        
        Args:
            user_id: User identifier
            audio_bytes: Audio data (WAV or PCM)
        
        Returns:
            Verification result
        """
        # Load profile
        try:
            profile = self.profile_repo.load_profile(user_id)
        except VoiceProfileNotFoundError:
            return {
                "type": "verify",
                "success": False,
                "verified": False,
                "claimed_id": user_id,
                "match": False,
                "message": "Profile not found",
            }
        
        # Check enrollment status
        if profile.get("enrollment_status") != "enrolled":
            return {
                "type": "verify",
                "success": False,
                "verified": False,
                "claimed_id": user_id,
                "match": False,
                "message": "Profile not enrolled yet. Please enroll first.",
            }
        
        # Check enrollment count
        if profile.get("enrollment_count", 0) < 3:
            return {
                "type": "verify",
                "success": False,
                "verified": False,
                "claimed_id": user_id,
                "match": False,
                "message": f"Profile needs {3 - profile.get('enrollment_count', 0)} more samples. Please complete enrollment first.",
            }
        
        # Process audio
        try:
            embedding, quality = self._process_audio(audio_bytes)
        except Exception as e:
            logger.exception(f"Audio processing failed | user_id={user_id}")
            return {
                "type": "verify",
                "success": False,
                "verified": False,
                "claimed_id": user_id,
                "match": False,
                "message": "Failed to process audio input",
                "error": str(e),
            }
        
        # Check quality
        if not quality["ok"]:
            return {
                "type": "verify",
                "success": False,
                "verified": False,
                "claimed_id": user_id,
                "match": False,
                "message": quality.get("reason") or "Audio quality requirements not met",
                "quality": quality,
            }
        
        # Get embeddings
        embeddings = self.profile_repo.get_embeddings(user_id)
        if not embeddings:
            return {
                "type": "verify",
                "success": False,
                "verified": False,
                "claimed_id": user_id,
                "match": False,
                "message": "No stored embeddings",
            }
        
        # Calculate scores
        probe_norm = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        # Score against mean
        mean_embedding = np.asarray(profile.get("mean_embedding"), dtype=np.float32).reshape(-1)
        cos_mean = float(np.clip(np.dot(probe_norm, mean_embedding), -1.0, 1.0))
        
        # Score against all samples
        sample_scores = []
        for vec in embeddings:
            norm_vec = vec / (np.linalg.norm(vec) + 1e-6)
            sample_scores.append(float(np.clip(np.dot(probe_norm, norm_vec), -1.0, 1.0)))
        
        max_similarity = float(max(sample_scores))
        avg_similarity = float(np.mean(sample_scores))
        
        # Verification decision
        verified = max_similarity >= self.verification_threshold and cos_mean >= self.cosine_threshold
        confidence = self._score_confidence(max_similarity)
        
        return {
            "type": "verify",
            "success": verified,  # success chỉ True khi verified=True
            "verified": verified,
            "claimed_id": user_id,
            "match": verified,
            "score": max_similarity,
            "mean_score": cos_mean,
            "avg_score": avg_similarity,
            "confidence": confidence,
            "quality": quality,
            "message": "Voice match" if verified else "Voice does not match claimed identity",
        }
    
    def reset_enrollment(self, user_id: str) -> Dict[str, Any]:
        """
        Reset enrollment for user - delete Azure blob profile and clear database link.
        
        Args:
            user_id: User identifier to reset
            
        Returns:
            Dict with success, message, details
        """
        try:
            results = {
                "user_id": user_id,
                "blob_deleted": False,
                "db_cleared": False,
                "cache_cleared": False,
            }
            
            # 1. Delete profile from Azure Blob Storage
            try:
                blob_deleted = self.profile_repo.delete_profile(user_id)
                results["blob_deleted"] = blob_deleted
                if blob_deleted:
                    logger.info(f"✅ Deleted blob profile for user {user_id}")
                else:
                    logger.warning(f"⚠️ Blob profile not found for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to delete blob profile for {user_id}: {e}")
                results["blob_error"] = str(e)
            
            # 2. Clear VoiceSamplePath in SQL database
            try:
                if self.sql_server_repo:
                    # Set to empty string to clear the link
                    db_updated = self.sql_server_repo.update_voice_sample_path(user_id, "")
                    results["db_cleared"] = db_updated
                    if db_updated:
                        logger.info(f"✅ Cleared VoiceSamplePath in DB for user {user_id}")
                    else:
                        logger.warning(f"⚠️ Failed to clear DB path for user {user_id}")
                else:
                    results["db_cleared"] = None  # No SQL repo configured
                    logger.info("SQL repository not configured, skipping DB update")
            except Exception as e:
                logger.warning(f"Failed to clear DB path for {user_id}: {e}")
                results["db_error"] = str(e)
            
            # 3. Invalidate all caches for this user
            try:
                self._invalidate_user_caches(user_id)
                results["cache_cleared"] = True
                logger.info(f"✅ Invalidated caches for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to invalidate caches for {user_id}: {e}")
                results["cache_error"] = str(e)
            
            # Determine overall success
            success = results["blob_deleted"] or results.get("blob_error") is None
            
            return {
                "success": success,
                "user_id": user_id,
                "message": f"Enrollment reset {'successful' if success else 'partially failed'} for user {user_id}",
                "details": results,
            }
            
        except Exception as e:
            logger.exception(f"Error resetting enrollment for user {user_id}: {e}")
            return {
                "success": False,
                "user_id": user_id,
                "error": str(e),
                "message": f"Failed to reset enrollment: {e}",
            }

    def list_all_profiles(self) -> List[Dict[str, Any]]:
        """List all voice profiles."""
        profiles = []
        for user_id in self.profile_repo.list_profiles():
            try:
                profile = self.profile_repo.load_profile(user_id)
                embeddings = self.profile_repo.get_embeddings(user_id)
                profiles.append({
                    "id": user_id,
                    "name": profile.get("name", user_id),
                    "enrollment_status": profile.get("enrollment_status", "unknown"),
                    "enrollment_count": len(embeddings),
                })
            except Exception as e:
                logger.warning(f"Failed to load profile {user_id}: {e}")
        return profiles

    def get_profile_status(self, user_id: str) -> Dict[str, Any]:
        """Return consolidated status of a profile: local counts, blob presence, DB URL."""
        try:
            profile = self.profile_repo.load_profile(user_id)
        except VoiceProfileNotFoundError:
            return {
                "exists": False,
                "id": user_id,
                "message": "Local profile not found",
            }

        embeddings = self.profile_repo.get_embeddings(user_id)
        status: Dict[str, Any] = {
            "exists": True,
            "id": user_id,
            "name": profile.get("name", user_id),
            "enrollment_status": profile.get("enrollment_status", "unknown"),
            "enrollment_count": len(embeddings),
        }
        # Only include local file path if repository has that method
        file_path = getattr(self.profile_repo, "_profile_path", None)
        if callable(file_path):
            try:
                status["file_path"] = str(file_path(user_id))
            except Exception:
                pass

        # Check blob presence
        if self.azure_blob_repo is not None:
            try:
                status["blob_exists"] = bool(self.azure_blob_repo.profile_exists_in_blob(user_id))
            except Exception as e:
                status["blob_exists_error"] = str(e)

        # Get DB URL
        if self.sql_server_repo is not None:
            try:
                status["db_voice_sample_path"] = self.sql_server_repo.get_voice_sample_path(user_id)
            except Exception as e:
                status["db_error"] = str(e)

        return status
    
    def get_profile_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get profile information."""
        try:
            profile = self.profile_repo.load_profile(user_id)
            embeddings = self.profile_repo.get_embeddings(user_id)
            return {
                "id": user_id,
                "name": profile.get("name", user_id),
                "enrollment_status": profile.get("enrollment_status", "not_enrolled"),
                "enrollment_count": len(embeddings),
            }
        except VoiceProfileNotFoundError:
            return None
    
    def delete_profile(self, user_id: str) -> Dict[str, Any]:
        """Delete voice profile."""
        try:
            profile = self.profile_repo.load_profile(user_id)
            name = profile.get("name", user_id)
        except VoiceProfileNotFoundError:
            return {"error": "Profile not found", "id": user_id}
        
        if self.profile_repo.delete_profile(user_id):
            return {
                "success": True,
                "id": user_id,
                "name": name,
                "message": f"Profile {name} deleted successfully",
            }
        return {"error": "Failed to delete profile", "id": user_id}
    
    # Private helper methods
    
    def _process_audio(self, audio_bytes: bytes, skip_noise_filter: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio and extract embedding with quality check.
        
        Args:
            audio_bytes: Raw audio bytes (WAV or PCM)
            skip_noise_filter: If True, skip noise filtering (audio already filtered)
        """
        # Convert to mono
        signal, sr = bytes_to_mono(audio_bytes)
        
        # Apply noise filter ONLY if not already filtered
        # CRITICAL: Double filtering destroys voice characteristics!
        if skip_noise_filter:
            # Audio already filtered by speech_service, just convert to float32
            signal_filtered = signal.astype(np.float32)
        else:
            # Apply noise filter (works on int16 PCM)
            filtered = self.noise_filter.reduce_noise(
                (signal.astype(np.float32)).astype(np.int16).tobytes()
            )
            signal_filtered = np.frombuffer(filtered, dtype=np.int16).astype(np.float32)
        
        # Resample if needed
        signal_filtered = resample_audio(signal_filtered, sr, self.noise_filter.sample_rate)
        
        # Normalize to [-1, 1] and remove DC
        signal_norm = signal_filtered / 32768.0
        if signal_norm.size > 0:
            signal_norm -= np.mean(signal_norm)
        
        # Rescue path: if RMS too low, try adaptive gain (server-side)
        if signal_norm.size == 0 or float(np.max(np.abs(signal_norm)) if signal_norm.size else 0.0) < 1e-7:
            # No mic signal detected
            quality = {
                "rms": 0.0,
                "voiced_ratio": 0.0,
                "snr_db": 0.0,
                "clipping_ratio": 0.0,
                "duration_sec": 0.0,
                "ok": False,
                "reason": "No microphone signal detected",
                "model": self.model_tag,
            }
            return np.zeros(self.embedding_dim, dtype=np.float32), quality
        
        pre_rms = float(np.sqrt(np.mean(signal_norm * signal_norm)) + 1e-10)
        rescued = False
        gain_scale = 1.0
        pctl = 0.0
        if pre_rms < self.thresholds.rms_floor:
            abs_sig = np.abs(signal_norm)
            pctl = float(np.percentile(abs_sig, self.gain_target_pctl)) if abs_sig.size else 0.0
            if pctl > 1e-6:
                target = self.gain_target_peak
                gain_scale = min(self.gain_max, target / pctl)
                if gain_scale > 1.02:  # apply only if meaningful
                    signal_norm = np.clip(signal_norm * gain_scale, -1.0, 1.0)
                    rescued = True
            # Recenter after rescue attempt
            if signal_norm.size:
                signal_norm -= np.mean(signal_norm)
        
        # Calculate quality metrics
        duration = len(signal_norm) / self.noise_filter.sample_rate
        rms, voiced_ratio, snr_db, clipping_ratio = self.quality_analyzer.estimate_quality(signal_norm)
        
        # Check quality thresholds
        rms_floor = self.thresholds.rms_floor
        
        quality_ok = True
        reason = None
        
        if duration < self.thresholds.min_duration:
            quality_ok = False
            reason = f"Recording too short (>={self.thresholds.min_duration:.0f}s required)"
        elif rms < rms_floor:
            quality_ok = False
            reason = f"RMS below floor ({rms:.4f} < {rms_floor:.4f})"
        elif voiced_ratio < self.thresholds.voiced_floor:
            quality_ok = False
            reason = f"Voiced ratio too low ({voiced_ratio:.2f})"
        elif snr_db < self.thresholds.snr_floor_db:
            quality_ok = False
            reason = f"SNR below {self.thresholds.snr_floor_db:.0f}dB ({snr_db:.1f}dB)"
        elif clipping_ratio > self.thresholds.clip_ceiling:
            quality_ok = False
            reason = f"Signal clipping too high ({clipping_ratio:.2%})"
        
        # Optional dynamic relax if only RMS is failing but speech looks valid
        rms_relaxed = False
        if (not quality_ok and self.dynamic_rms_relax and reason and reason.startswith("RMS below floor")):
            # More permissive relax: voiced can be slightly under floor, SNR can be 7dB under.
            voiced_ok = voiced_ratio >= max(self.thresholds.voiced_floor - 0.07, 0.12)
            snr_ok = snr_db >= (self.thresholds.snr_floor_db - 7.0)
            if voiced_ok and snr_ok and duration >= self.thresholds.min_duration:
                logger.info(
                    "Audio accepted by dynamic RMS relax | pre_rms=%.5f post_rms=%.5f voiced=%.2f snr=%.1f duration=%.2fs",
                    pre_rms, rms, voiced_ratio, snr_db, duration,
                )
                quality_ok = True
                reason = None
                rms_relaxed = True

        quality = {
            "rms": rms,
            "rms_floor": rms_floor,
            "voiced_ratio": voiced_ratio,
            "snr_db": snr_db,
            "clipping_ratio": clipping_ratio,
            "duration_sec": duration,
            "ok": quality_ok,
            "reason": reason,
            "model": self.model_tag,
            "pre_rms": pre_rms,
            "rms_relaxed": rms_relaxed,
            "gain_target_pctl": self.gain_target_pctl,
            "gain_target_peak": self.gain_target_peak,
            "gain_scale": gain_scale,
            "gain_rescued": rescued,
            "gain_pctl_value": pctl,
        }
        
        if not quality_ok:
            # Cleanup before returning
            del signal_norm, signal_filtered
            return np.zeros(self.embedding_dim, dtype=np.float32), quality
        
        # Extract embedding
        try:
            embedding = self.model_repo.extract_embedding(signal_norm)
            # Cleanup intermediate arrays
            del signal_norm, signal_filtered
        except Exception as e:
            quality["ok"] = False
            quality["reason"] = f"Embedding extraction failed: {e}"
            del signal_norm, signal_filtered
            return np.zeros(self.embedding_dim, dtype=np.float32), quality
        
        return embedding, quality
    
    def _process_audio_for_enrollment(self, audio_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio for enrollment with stricter quality checks."""
        embedding, quality = self._process_audio(audio_bytes)
        
        # Additional check for enrollment
        if quality["ok"] and quality["duration_sec"] < self.thresholds.min_enroll_duration:
            quality["ok"] = False
            quality["reason"] = f"Recording too short for enrollment (>={self.thresholds.min_enroll_duration:.0f}s required)"
        
        return embedding, quality
    
    def _check_enrollment_consistency(
        self,
        embedding: np.ndarray,
        existing_embeddings: List[np.ndarray],
        profile: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Check if new enrollment sample is consistent with existing samples."""
        if not existing_embeddings:
            return None
        
        # Calculate similarities
        comparisons = []
        for vec in existing_embeddings:
            norm_vec = vec / (np.linalg.norm(vec) + 1e-6)
            norm_emb = embedding / (np.linalg.norm(embedding) + 1e-6)
            comparisons.append(float(np.dot(norm_emb, norm_vec)))
        
        avg_similarity = float(np.mean(comparisons))
        min_similarity = float(np.min(comparisons))
        
        # Flexible threshold that decreases with more samples
        flex_threshold = max(
            self.enroll_min_similarity,
            self.enrollment_threshold - self.enroll_decay_per_sample * len(existing_embeddings),
        )
        
        passed = avg_similarity >= flex_threshold or min_similarity >= (flex_threshold - 0.05)
        
        return {
            "avg_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "threshold": flex_threshold,
            "passed": passed,
        }
    
    def _update_profile_stats(self, user_id: str, embeddings: List[np.ndarray]) -> None:
        """Update profile statistics (mean embedding, variance)."""
        if not embeddings:
            return
        
        matrix = np.stack(embeddings, axis=0)
        mean_vec = matrix.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec) + 1e-6)
        mean_unit = mean_vec / norm
        
        diffs = matrix - mean_vec
        within_var = float(np.mean(np.sum(diffs * diffs, axis=1)))
        sigma = float(math.sqrt(within_var / max(len(embeddings), 1)))
        
        self.profile_repo.update_mean_embedding(user_id, mean_unit, within_var, sigma)
    
    def _get_enrolled_profiles(self) -> List[Dict[str, Any]]:
        """Get all enrolled profiles with at least 3 samples."""
        enrolled = []
        for user_id in self.profile_repo.list_profiles():
            try:
                profile = self.profile_repo.load_profile(user_id)
                # Chỉ lấy profile đã enrolled và có đủ 3 samples
                if profile.get("enrollment_status") == "enrolled" and profile.get("enrollment_count", 0) >= 3:
                    embeddings = self.profile_repo.get_embeddings(user_id)
                    if embeddings:
                        profile["embeddings"] = embeddings
                        enrolled.append(profile)
            except Exception as e:
                logger.warning(f"Failed to load profile {user_id}: {e}")
        return enrolled

    def _get_enrolled_profiles_batch(
        self,
        whitelist_user_ids: Optional[List[str]] = None,
        max_workers: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get enrolled profiles in parallel (thread pool) and precompute normalized vectors."""
        import concurrent.futures

        all_ids = self.profile_repo.list_profiles()
        logger.info(f"📋 All profile IDs in repo: {all_ids}")
        
        if whitelist_user_ids:
            logger.info(f"📋 Whitelist user IDs: {whitelist_user_ids}")
            id_set = set(str(uid) for uid in whitelist_user_ids)  # Ensure string comparison
            target_ids = [uid for uid in all_ids if str(uid) in id_set]
            logger.info(f"📋 Target IDs (intersection): {target_ids}")
        else:
            target_ids = all_ids
            logger.info(f"📋 No whitelist, using all {len(target_ids)} profiles")

        def load_one(uid: str) -> Optional[Dict[str, Any]]:
            try:
                profile = self.profile_repo.load_profile(uid)
                enrollment_status = profile.get("enrollment_status")
                enrollment_count = profile.get("enrollment_count", 0)
                
                if enrollment_status != "enrolled" or enrollment_count < 3:
                    return None
                    
                embeddings = self.profile_repo.get_embeddings(uid)
                if not embeddings:
                    return None
                    
                # Normalize embeddings matrix
                mat = np.stack(embeddings, axis=0).astype(np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-6
                norm_mat = mat / norms
                profile["embeddings"] = embeddings  # keep raw if needed
                profile["embeddings_norm"] = norm_mat

                # Cache normalized matrix using LRUCache
                self._embedding_cache.set(uid, norm_mat)

                # Prepare mean embedding if present
                raw_mean = profile.get("mean_embedding")
                if raw_mean is not None:
                    mean_arr = np.asarray(raw_mean, dtype=np.float32).reshape(-1)
                    if mean_arr.size == self.embedding_dim:
                        mean_norm = mean_arr / (np.linalg.norm(mean_arr) + 1e-6)
                        profile["mean_embedding_norm"] = mean_norm
                        self._mean_cache.set(uid, mean_norm)
                return profile
            except Exception as e:
                logger.warning(f"Failed to load profile {uid}: {e}")
                return None

        enrolled: List[Dict[str, Any]] = []
        if not target_ids:
            return enrolled
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for prof in ex.map(load_one, target_ids):
                if prof:
                    enrolled.append(prof)
        return enrolled
    
    def _calculate_candidate_scores(
        self,
        probe: np.ndarray,
        profiles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Calculate similarity scores for all candidate profiles.
        
        OPTIMIZED: Uses matrix multiplication instead of Python loop.
        
        Algorithm:
        1. Stack all profile embeddings into a single matrix
        2. Compute all similarities via single matrix-vector multiply: O(1)
        3. Aggregate scores per profile using vectorized operations
        
        Complexity: O(total_embeddings) matrix ops vs O(N * samples) Python loop
        """
        probe_norm = probe / (np.linalg.norm(probe) + 1e-6)
        
        logger.info(f"🔍 Calculating scores for {len(profiles)} profiles | probe_dim={probe.shape} | expected_dim={self.embedding_dim}")
        
        # Build mapping: profile_index -> (start_idx, end_idx) in stacked matrix
        profile_ranges: List[Tuple[int, int, Dict[str, Any]]] = []
        all_embeddings_list: List[np.ndarray] = []
        current_idx = 0
        
        for profile in profiles:
            sample_matrix = profile.get("embeddings_norm")
            mean_vec = profile.get("mean_embedding_norm")
            
            # Validate dimensions
            if mean_vec is not None and isinstance(mean_vec, np.ndarray):
                if mean_vec.size != self.embedding_dim:
                    logger.warning(f"⚠️ SKIP: Profile '{profile.get('name')}' has incompatible dim={mean_vec.size}, expected={self.embedding_dim}. RE-ENROLLMENT REQUIRED!")
                    continue
            
            if sample_matrix is None:
                # Fallback: build normalized matrix on the fly
                sample_vecs = []
                for emb in profile.get("embeddings", []):
                    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
                    if arr.size != self.embedding_dim:
                        continue
                    norm = np.linalg.norm(arr)
                    if norm < 1e-6:
                        continue
                    sample_vecs.append(arr / norm)
                
                if not sample_vecs and mean_vec is None:
                    continue
                if not sample_vecs and mean_vec is not None:
                    sample_vecs = [mean_vec]
                if not sample_vecs:
                    continue
                    
                sample_matrix = np.stack(sample_vecs, axis=0)
            else:
                # Validate preloaded matrix dimensions
                if hasattr(sample_matrix, 'shape') and len(sample_matrix.shape) >= 2:
                    sample_dim = sample_matrix.shape[1]
                    if sample_dim != self.embedding_dim:
                        logger.warning(f"⚠️ SKIP: Profile '{profile.get('name')}' has preloaded embeddings with dim={sample_dim}, expected={self.embedding_dim}. RE-ENROLLMENT REQUIRED!")
                        continue
            
            n_samples = sample_matrix.shape[0]
            profile_ranges.append((current_idx, current_idx + n_samples, profile))
            all_embeddings_list.append(sample_matrix)
            current_idx += n_samples
        
        if not all_embeddings_list:
            return []
        
        # VECTORIZED: Stack all embeddings into single matrix
        all_embeddings = np.vstack(all_embeddings_list)  # Shape: (total_samples, dim)
        
        # VECTORIZED: Single matrix-vector multiply for ALL similarities
        all_similarities = np.clip(all_embeddings @ probe_norm, -1.0, 1.0)  # Shape: (total_samples,)
        
        # Aggregate scores per profile
        candidates = []
        
        for start_idx, end_idx, profile in profile_ranges:
            sample_scores = all_similarities[start_idx:end_idx]
            sample_count = len(sample_scores)
            
            max_sample_cosine = float(np.max(sample_scores))
            avg_sample_cosine = float(np.mean(sample_scores))
            best_sample_index = int(np.argmax(sample_scores))
            best_sample_angle = float(np.arccos(np.clip(sample_scores[best_sample_index], -1.0, 1.0)))
            
            # Mean embedding score
            mean_vec = profile.get("mean_embedding_norm")
            if mean_vec is not None and isinstance(mean_vec, np.ndarray) and mean_vec.size == self.embedding_dim:
                mean_cosine = float(np.clip(np.dot(probe_norm, mean_vec), -1.0, 1.0))
                mean_angle = float(np.arccos(mean_cosine))
            else:
                mean_cosine = max_sample_cosine
                mean_angle = float(np.arccos(np.clip(mean_cosine, -1.0, 1.0)))
            
            # Use best of mean or sample scores
            score = max(mean_cosine, max_sample_cosine)
            score_source = "mean" if score == mean_cosine else "sample"
            angle = mean_angle if score_source == "mean" else best_sample_angle
            
            # Use display name if available
            # Check if name is actually a UUID (not a real name)
            raw_name = profile.get("name") or ""
            user_id_str = str(profile.get("user_id", ""))
            
            # Detect if name looks like a UUID or is same as user_id
            is_uuid_like = (
                len(raw_name) >= 32 and  # UUID length
                raw_name.replace("-", "").isalnum() and
                len(raw_name.replace("-", "")) == 32
            ) or raw_name == user_id_str or not raw_name.strip()
            
            if is_uuid_like:
                # Use short ID prefix instead of full UUID for readability
                # e.g. "User-e7ed" instead of "e7ed355d-aafb-4bc9-947c-73789419e220"
                short_id = user_id_str[:8] if len(user_id_str) >= 8 else user_id_str
                display_name = f"User-{short_id}"
            else:
                display_name = raw_name
            
            candidates.append({
                "id": profile["user_id"],
                "name": display_name,
                "user_id": profile.get("user_id"),
                "cosine": score,
                "angle": angle,
                "mean_cosine": mean_cosine,
                "mean_angle": mean_angle,
                "max_sample_cosine": max_sample_cosine,
                "best_sample_angle": best_sample_angle,
                "avg_sample_cosine": avg_sample_cosine,
                "score_source": score_source,
                "sample_count": sample_count,
            })
        
        return candidates
    
    def _accept_identification(
        self,
        candidate: Dict[str, Any],
        zscore: Optional[float],
        cohort_size: int,
        probe_embedding: Optional[np.ndarray] = None,
        speaker_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[bool, List[str]]:
        """Determine if identification should be accepted.
        
        Uses Adaptive S-Norm when cohort is small (1-2 speakers) for more
        reliable score calibration. Falls back to traditional z-score
        for larger cohorts.
        
        Args:
            candidate: Best candidate dict with scores
            zscore: Z-score from cohort analysis (may be None for small cohorts)
            cohort_size: Number of speakers in current session
            probe_embedding: Test utterance embedding (for S-Norm)
            speaker_embeddings: Dict of speaker_id -> mean_embedding (for S-Norm)
        """
        reasons = []
        accepted = True
        
        # Always check raw cosine threshold first
        if candidate["cosine"] < self.cosine_threshold:
            accepted = False
            reasons.append(f"low_cosine ({candidate['cosine']:.3f} < {self.cosine_threshold})")
        
        if candidate["angle"] > self.angle_cap:
            accepted = False
            reasons.append(f"angle_cap ({np.rad2deg(candidate['angle']):.1f}° > {np.rad2deg(self.angle_cap):.1f}°)")
        
        # Use S-Norm for small cohorts (more reliable than z-score)
        if self.use_snorm and cohort_size <= 2 and probe_embedding is not None and speaker_embeddings is not None:
            speaker_id = candidate.get("id") or candidate.get("user_id")
            speaker_emb = speaker_embeddings.get(speaker_id)
            
            if speaker_emb is not None:
                try:
                    snorm_result = self._score_normalizer.s_normalize(
                        raw_score=candidate["cosine"],
                        probe_embedding=probe_embedding,
                        speaker_embedding=speaker_emb,
                        speaker_id=speaker_id,
                    )
                    
                    s_score = snorm_result["s_score"]
                    candidate["s_score"] = s_score
                    candidate["z_norm_score"] = snorm_result["z_score"]
                    candidate["t_norm_score"] = snorm_result["t_score"]
                    
                    logger.info(
                        f"📊 S-Norm: raw={candidate['cosine']:.3f} | "
                        f"z_norm={snorm_result['z_score']:.2f} | "
                        f"t_norm={snorm_result['t_score']:.2f} | "
                        f"s_score={s_score:.2f}"
                    )
                    
                    # S-Norm threshold check (in std units)
                    if accepted and s_score < self.snorm_threshold:
                        accepted = False
                        reasons.append(f"snorm_low ({s_score:.2f} < {self.snorm_threshold})")
                    
                except Exception as e:
                    logger.warning(f"S-Norm computation failed: {e}")
                    # Fall through to z-score check
        
        # Traditional z-score for larger cohorts (3+ speakers)
        elif zscore is not None and cohort_size >= 3 and zscore < self.z_threshold:
            accepted = False
            reasons.append(f"zscore ({zscore:.2f} < {self.z_threshold})")
        
        logger.info(f"🔍 Accept check: accepted={accepted} | cohort_size={cohort_size} | reasons={reasons}")
        return accepted, reasons
    
    def _score_confidence(self, cosine: float) -> str:
        """Calculate confidence level from cosine similarity."""
        if cosine >= self.cosine_threshold + 0.12:
            return "High"
        if cosine >= self.cosine_threshold + 0.05:
            return "Medium"
        return "Low"
