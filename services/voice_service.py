"""Voice Service - Business logic for voice authentication."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from core.exceptions import VoiceProfileNotFoundError, VoiceAuthenticationError, AudioValidationError
from repositories.interfaces.i_voice_profile_repository import IVoiceProfileRepository
from repositories.models.speechbrain_model import SpeechBrainModelRepository
from services.interfaces.i_voice_service import IVoiceService
from services.audio_processing.audio_utils import (
    AudioQualityAnalyzer,
    NoiseFilter,
    bytes_to_mono,
    resample_audio,
)

logger = logging.getLogger(__name__)

# Constants
MIN_ENROLL_SAMPLES = 3
ANGLE_CAP_DEG = 45.0


@dataclass
class QualityThresholds:
    """Audio quality thresholds for enrollment and verification."""
    min_duration: float = 2.0  # Lowered from 5.0 for streaming compatibility
    min_enroll_duration: float = 10.0
    # Lower RMS floors to reduce false "nói nhỏ" (quiet speech) detections on typical consumer mics.
    # Can still be overridden via environment (Config VOICE_RMS_FLOOR_*).
    rms_floor_ecapa: float = 0.008  # was 0.012
    rms_floor_xvector: float = 0.010  # was 0.015
    voiced_floor: float = 0.20
    snr_floor_db: float = 15.0
    clip_ceiling: float = 0.02

class VoiceService(IVoiceService):
    """Voice authentication business logic service."""
    
    def __init__(
        self,
        voice_profile_repo: IVoiceProfileRepository,
        model_repo: SpeechBrainModelRepository,
        thresholds: QualityThresholds | None = None,
        azure_blob_repo: Any = None,
        sql_server_repo: Any = None,
    ) -> None:
        """
        Initialize voice service.
        
        Args:
            voice_profile_repo: Interface for voice profile persistence
            model_repo: Repository for SpeechBrain model
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
        
        # Thresholds - IMPROVED for better accuracy
        self.enrollment_threshold = 0.76
        # Increased from 0.70/0.78 to 0.75/0.80 for better speaker discrimination
        self.cosine_threshold = 0.80 if self.model_tag == "xvector" else 0.75
        self.verification_threshold = max(self.cosine_threshold + 0.05, 0.82)
        # NEW: Margin threshold for speaker switching (prevent false switches)
        self.speaker_switch_margin = 0.06  # Top-2 margin requirement
        self.speaker_switch_hits_required = 3  # Require 3 consecutive hits
        self.angle_cap = float(np.deg2rad(ANGLE_CAP_DEG))
        self.z_threshold = 2.2
        self.enroll_min_similarity = 0.63 if self.model_tag == "ecapa" else 0.68
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
            self.thresholds.rms_floor_ecapa = float(getattr(_AppConfig, "VOICE_RMS_FLOOR_ECAPA", self.thresholds.rms_floor_ecapa))
            self.thresholds.rms_floor_xvector = float(getattr(_AppConfig, "VOICE_RMS_FLOOR_XVECTOR", self.thresholds.rms_floor_xvector))
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
        # Embedding caches for performance
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._mean_cache: Dict[str, np.ndarray] = {}
    
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
        from app.config import Config
        
        try:
            url = f"{Config.AUTH_SERVICE_BASE_URL}/api/defense-sessions/{session_id}/users"
            
            async with httpx.AsyncClient(
                verify=Config.AUTH_SERVICE_VERIFY_SSL,
                timeout=Config.AUTH_SERVICE_TIMEOUT
            ) as client:
                response = await client.get(url)
                
                if response.status_code != 200:
                    logger.error(f"Failed to fetch defense session users: {response.status_code}")
                    return None
                
                data = response.json()
                
                # Extract user IDs from response
                if "data" in data and isinstance(data["data"], list):
                    user_ids = [user["id"] for user in data["data"] if "id" in user]
                    logger.info(f"✅ Fetched {len(user_ids)} users from defense session {session_id}")
                    return user_ids
                
                logger.warning(f"Invalid response format from defense session API")
                return None
                
        except Exception as e:
            logger.exception(f"Error fetching defense session users: {e}")
            return None
    
    def identify_speaker(self, audio_bytes: bytes, whitelist_user_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Identify speaker from audio.
        
        Args:
            audio_bytes: Audio data (WAV or PCM)
        
        Returns:
            Identification result
        """
        # Process audio
        try:
            embedding, quality = self._process_audio(audio_bytes)
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
        
        # Decision logic
        accepted, reasons = self._accept_identification(best, zscore, cohort_size)
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
    
    def _process_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio and extract embedding with quality check."""
        # Convert to mono
        signal, sr = bytes_to_mono(audio_bytes)
        
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
        if pre_rms < (self.thresholds.rms_floor_ecapa if self.model_tag == "ecapa" else self.thresholds.rms_floor_xvector):
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
            logger.debug(
                "Gain rescue attempt | pre_rms=%.6f pctl%d=%.6f scale=%.2f rescued=%s",
                pre_rms, self.gain_target_pctl, pctl, gain_scale, rescued,
            )
        
        # Calculate quality metrics
        duration = len(signal_norm) / self.noise_filter.sample_rate
        rms, voiced_ratio, snr_db, clipping_ratio = self.quality_analyzer.estimate_quality(signal_norm)
        
        # Check quality thresholds
        rms_floor = (
            self.thresholds.rms_floor_ecapa
            if self.model_tag == "ecapa"
            else self.thresholds.rms_floor_xvector
        )
        
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
            else:
                logger.debug(
                    "Dynamic RMS relax NOT applied | voiced=%.2f need>=%.2f snr=%.1f need>=%.1f duration=%.2fs rescued=%s",
                    voiced_ratio, max(self.thresholds.voiced_floor - 0.07, 0.12),
                    snr_db, self.thresholds.snr_floor_db - 7.0,
                    duration, rescued,
                )

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
        if whitelist_user_ids:
            id_set = set(whitelist_user_ids)
            target_ids = [uid for uid in all_ids if uid in id_set]
        else:
            target_ids = all_ids

        def load_one(uid: str) -> Optional[Dict[str, Any]]:
            try:
                profile = self.profile_repo.load_profile(uid)
                if profile.get("enrollment_status") != "enrolled" or profile.get("enrollment_count", 0) < 3:
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

                # Cache normalized matrix
                self._embedding_cache[uid] = norm_mat

                # Prepare mean embedding if present
                raw_mean = profile.get("mean_embedding")
                if raw_mean is not None:
                    mean_arr = np.asarray(raw_mean, dtype=np.float32).reshape(-1)
                    if mean_arr.size == self.embedding_dim:
                        mean_norm = mean_arr / (np.linalg.norm(mean_arr) + 1e-6)
                        profile["mean_embedding_norm"] = mean_norm
                        self._mean_cache[uid] = mean_norm
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
        """Calculate similarity scores for all candidate profiles."""
        probe_norm = probe / (np.linalg.norm(probe) + 1e-6)
        candidates = []
        
        for profile in profiles:
            mean_vec = profile.get("mean_embedding_norm")
            mean_cosine = None
            mean_angle = None
            if isinstance(mean_vec, np.ndarray) and mean_vec.size == self.embedding_dim:
                mean_cosine = float(np.clip(np.dot(probe_norm, mean_vec), -1.0, 1.0))
                mean_angle = float(np.arccos(mean_cosine))

            sample_matrix = profile.get("embeddings_norm")
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
                sample_matrix = np.stack(sample_vecs, axis=0)

            sample_cosines = np.clip(sample_matrix @ probe_norm, -1.0, 1.0)
            max_sample_cosine = float(np.max(sample_cosines))
            avg_sample_cosine = float(np.mean(sample_cosines))
            best_sample_index = int(np.argmax(sample_cosines))
            best_sample_angle = float(np.arccos(sample_cosines[best_sample_index]))
            
            if mean_cosine is None:
                mean_cosine = max_sample_cosine
                mean_angle = float(np.arccos(np.clip(mean_cosine, -1.0, 1.0)))
            
            score = max(mean_cosine, max_sample_cosine)
            score_source = "mean" if score == mean_cosine else "sample"
            angle = mean_angle if score_source == "mean" else best_sample_angle
            
            candidates.append({
                "id": profile["user_id"],
                "name": profile.get("name", profile["user_id"]),
                "cosine": score,
                "angle": angle,
                "mean_cosine": mean_cosine,
                "mean_angle": mean_angle,
                "max_sample_cosine": max_sample_cosine,
                "best_sample_angle": best_sample_angle,
                "avg_sample_cosine": avg_sample_cosine,
                "score_source": score_source,
                "sample_count": len(sample_vecs),
            })
        
        return candidates
    
    def _accept_identification(
        self,
        candidate: Dict[str, Any],
        zscore: Optional[float],
        cohort_size: int,
    ) -> Tuple[bool, List[str]]:
        """Determine if identification should be accepted."""
        reasons = []
        accepted = True
        
        if candidate["cosine"] < self.cosine_threshold:
            accepted = False
            reasons.append(f"low_cosine ({candidate['cosine']:.3f} < {self.cosine_threshold})")
        
        if candidate["angle"] > self.angle_cap:
            accepted = False
            reasons.append(f"angle_cap ({np.rad2deg(candidate['angle']):.1f}° > {np.rad2deg(self.angle_cap):.1f}°)")
        
        if zscore is not None and cohort_size >= 2 and zscore < self.z_threshold:
            accepted = False
            reasons.append(f"zscore ({zscore:.2f} < {self.z_threshold})")
        
        logger.info(f"🔍 Accept check: accepted={accepted} | reasons={reasons}")
        return accepted, reasons
    
    def _score_confidence(self, cosine: float) -> str:
        """Calculate confidence level from cosine similarity."""
        if cosine >= self.cosine_threshold + 0.12:
            return "High"
        if cosine >= self.cosine_threshold + 0.05:
            return "Medium"
        return "Low"
