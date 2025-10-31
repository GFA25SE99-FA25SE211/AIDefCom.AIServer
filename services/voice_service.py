"""Voice Service - Business logic for voice authentication."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


import numpy as np

from core.exceptions import VoiceProfileNotFoundError, VoiceAuthenticationError, AudioValidationError
from repositories.voice_profile_repository import VoiceProfileRepository
from repositories.models.speechbrain_model import SpeechBrainModelRepository
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
    min_duration: float = 5.0
    min_enroll_duration: float = 10.0
    rms_floor_ecapa: float = 0.012
    rms_floor_xvector: float = 0.015
    voiced_floor: float = 0.20
    snr_floor_db: float = 15.0
    clip_ceiling: float = 0.02

class VoiceService:
    """Voice authentication business logic service."""
    
    def __init__(
        self,
        voice_profile_repo: VoiceProfileRepository,
        model_repo: SpeechBrainModelRepository,
        thresholds: QualityThresholds | None = None,
    ) -> None:
        """
        Initialize voice service.
        
        Args:
            voice_profile_repo: Repository for voice profile persistence
            model_repo: Repository for SpeechBrain model
            thresholds: Quality thresholds for audio validation
        """
        self.profile_repo = voice_profile_repo
        self.model_repo = model_repo
        self.thresholds = thresholds or QualityThresholds()
        
        self.noise_filter = NoiseFilter()
        self.quality_analyzer = AudioQualityAnalyzer()
        
        # Get model info
        self.embedding_dim = model_repo.get_embedding_dim()
        self.model_tag = model_repo.get_model_tag()
        
        # Thresholds
        self.enrollment_threshold = 0.76
        self.cosine_threshold = 0.78 if self.model_tag == "xvector" else 0.70
        self.verification_threshold = max(self.cosine_threshold + 0.05, 0.80)
        self.angle_cap = float(np.deg2rad(ANGLE_CAP_DEG))
        self.z_threshold = 2.2
        self.enroll_min_similarity = 0.63 if self.model_tag == "ecapa" else 0.68
        self.enroll_decay_per_sample = 0.035
        
        logger.info(
            f"Voice Service initialized | model={self.model_tag} | dim={self.embedding_dim}"
        )
    
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
        Enroll voice sample for user.
        
        Args:
            user_id: User identifier
            audio_bytes: Audio data (WAV or PCM)
        
        Returns:
            Enrollment result
        """
        # Load profile
        try:
            profile = self.profile_repo.load_profile(user_id)
        except VoiceProfileNotFoundError:
            return {"error": "Profile not found. Create profile first.", "user_id": user_id}
        
        # Process audio
        try:
            embedding, quality = self._process_audio_for_enrollment(audio_bytes)
        except AudioValidationError as e:
            return {
                "error": str(e),
                "user_id": user_id,
                "quality": {"ok": False, "reason": str(e)},
            }
        except Exception as e:
            logger.exception(f"Audio processing failed | user_id={user_id}")
            return {"error": "Failed to process audio input", "user_id": user_id, "details": str(e)}
        
        # Check quality
        if not quality["ok"]:
            return {
                "error": quality.get("reason") or "Audio quality below requirements",
                "user_id": user_id,
                "quality": quality,
            }
        
        # Check consistency with existing samples
        embeddings = self.profile_repo.get_embeddings(user_id)
        consistency = self._check_enrollment_consistency(embedding, embeddings, profile)
        
        if consistency and not consistency["passed"]:
            return {
                "error": "Voice sample inconsistent with previous samples",
                "user_id": user_id,
                "consistency": consistency,
                "quality": quality,
            }
        
        # Add sample to profile
        self.profile_repo.add_voice_sample(
            user_id,
            embedding,
            metrics=quality,
        )
        
        # Update profile statistics
        embeddings = self.profile_repo.get_embeddings(user_id)
        self._update_profile_stats(user_id, embeddings)
        
        # Update enrollment status
        enrollment_count = len(embeddings)
        if enrollment_count >= MIN_ENROLL_SAMPLES:
            self.profile_repo.update_enrollment_status(user_id, "enrolled")
            enrollment_status = "enrolled"
            remaining = 0
        else:
            self.profile_repo.update_enrollment_status(user_id, "partial")
            enrollment_status = "partial"
            remaining = MIN_ENROLL_SAMPLES - enrollment_count
        
        response = {
            "user_id": user_id,
            "name": profile.get("name", user_id),
            "enrollment_status": enrollment_status,
            "enrollment_count": enrollment_count,
            "remaining_samples": remaining,
            "quality": quality,
        }
        
        if consistency:
            response["consistency"] = consistency
        
        return response
    
    def identify_speaker(self, audio_bytes: bytes) -> Dict[str, Any]:
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
                "identified": False,
                "speaker": "Unknown",
                "error": "Failed to process audio",
                "details": str(e),
            }
        
        # Check quality
        if not quality["ok"]:
            return {
                "identified": False,
                "speaker": "Guest",
                "score": 0.0,
                "message": quality.get("reason") or "Audio quality requirements not met",
                "quality": quality,
            }
        
        # Get all enrolled profiles
        enrolled_profiles = self._get_enrolled_profiles()
        logger.info(f"🔍 Identify: Found {len(enrolled_profiles)} enrolled profiles")
        
        if not enrolled_profiles:
            return {
                "identified": False,
                "speaker": "Guest",
                "score": 0.0,
                "message": "No enrolled users found",
                "quality": quality,
            }
        
        # Calculate scores for all candidates
        candidates = self._calculate_candidate_scores(embedding, enrolled_profiles)
        logger.info(f"🔍 Identify: {len(candidates)} candidates scored")
        
        if not candidates:
            return {
                "identified": False,
                "speaker": "Guest",
                "score": 0.0,
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
                "user_id": c["user_id"],
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
                "identified": True,
                "speaker": best["name"],
                "user_id": best["user_id"],
                "score": best["cosine"],
                "angular_dist": best["angle"],
                "zscore": float(zscore) if zscore is not None else None,
                "confidence": confidence,
                "quality": quality,
                "cohort_mean": cohort_mean,
                "cohort_std": cohort_std,
                "score_source": best.get("score_source", "mean"),
                "mean_cosine": best.get("mean_cosine"),
                "max_sample_cosine": best.get("max_sample_cosine"),
                "candidates": all_candidates,
            }
        
        return {
            "identified": False,
            "speaker": "Guest",
            "score": best["cosine"],
            "angular_dist": best["angle"],
            "zscore": float(zscore) if zscore is not None else None,
            "message": "Voice does not match any enrolled user",
            "reasons": reasons,
            "quality": quality,
            "cohort_mean": cohort_mean,
            "cohort_std": cohort_std,
            "score_source": best.get("score_source", "mean"),
            "mean_cosine": best.get("mean_cosine"),
            "max_sample_cosine": best.get("max_sample_cosine"),
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
            return {"error": "Profile not found", "verified": False}
        
        # Check enrollment status
        if profile.get("enrollment_status") != "enrolled":
            return {
                "error": "Profile not enrolled yet",
                "verified": False,
                "message": "Please complete enrollment first",
            }
        
        # Process audio
        try:
            embedding, quality = self._process_audio(audio_bytes)
        except Exception as e:
            logger.exception(f"Audio processing failed | user_id={user_id}")
            return {"error": "Failed to process audio input", "verified": False, "user_id": user_id}
        
        # Check quality
        if not quality["ok"]:
            return {"verified": False, "quality": quality, "message": quality.get("reason")}
        
        # Get embeddings
        embeddings = self.profile_repo.get_embeddings(user_id)
        if not embeddings:
            return {"error": "No stored embeddings", "verified": False}
        
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
            "user_id": user_id,
            "verified": verified,
            "score": max_similarity,
            "mean_score": cos_mean,
            "avg_score": avg_similarity,
            "confidence": confidence,
            "quality": quality,
        }
    
    def list_all_profiles(self) -> List[Dict[str, Any]]:
        """List all voice profiles."""
        profiles = []
        for user_id in self.profile_repo.list_profiles():
            try:
                profile = self.profile_repo.load_profile(user_id)
                embeddings = self.profile_repo.get_embeddings(user_id)
                profiles.append({
                    "user_id": user_id,
                    "name": profile.get("name", user_id),
                    "enrollment_status": profile.get("enrollment_status", "unknown"),
                    "enrollment_count": len(embeddings),
                })
            except Exception as e:
                logger.warning(f"Failed to load profile {user_id}: {e}")
        return profiles
    
    def get_profile_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get profile information."""
        try:
            profile = self.profile_repo.load_profile(user_id)
            embeddings = self.profile_repo.get_embeddings(user_id)
            return {
                "user_id": user_id,
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
            return {"error": "Profile not found", "user_id": user_id}
        
        if self.profile_repo.delete_profile(user_id):
            return {
                "success": True,
                "user_id": user_id,
                "name": name,
                "message": f"Profile {name} deleted successfully",
            }
        return {"error": "Failed to delete profile", "user_id": user_id}
    
    # Private helper methods
    
    def _process_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process audio and extract embedding with quality check."""
        # Convert to mono
        signal, sr = bytes_to_mono(audio_bytes)
        
        # Apply noise filter
        filtered = self.noise_filter.reduce_noise((signal / 32768.0 * 32768.0).astype(np.int16).tobytes())
        signal_filtered = np.frombuffer(filtered, dtype=np.int16).astype(np.float32)
        
        # Resample if needed
        signal_filtered = resample_audio(signal_filtered, sr, self.noise_filter.sample_rate)
        
        # Normalize
        signal_norm = signal_filtered / 32768.0
        signal_norm -= np.mean(signal_norm)
        
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
            reason = f"RMS below floor ({rms:.4f} < {rms_floor})"
        elif voiced_ratio < self.thresholds.voiced_floor:
            quality_ok = False
            reason = f"Voiced ratio too low ({voiced_ratio:.2f})"
        elif snr_db < self.thresholds.snr_floor_db:
            quality_ok = False
            reason = f"SNR below {self.thresholds.snr_floor_db:.0f}dB ({snr_db:.1f}dB)"
        elif clipping_ratio > self.thresholds.clip_ceiling:
            quality_ok = False
            reason = f"Signal clipping too high ({clipping_ratio:.2%})"
        
        quality = {
            "rms": rms,
            "voiced_ratio": voiced_ratio,
            "snr_db": snr_db,
            "clipping_ratio": clipping_ratio,
            "duration_sec": duration,
            "ok": quality_ok,
            "reason": reason,
            "model": self.model_tag,
        }
        
        if not quality_ok:
            return np.zeros(self.embedding_dim, dtype=np.float32), quality
        
        # Extract embedding
        try:
            embedding = self.model_repo.extract_embedding(signal_norm)
        except Exception as e:
            quality["ok"] = False
            quality["reason"] = f"Embedding extraction failed: {e}"
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
        """Get all enrolled profiles."""
        enrolled = []
        for user_id in self.profile_repo.list_profiles():
            try:
                profile = self.profile_repo.load_profile(user_id)
                if profile.get("enrollment_status") == "enrolled":
                    embeddings = self.profile_repo.get_embeddings(user_id)
                    if embeddings:
                        profile["embeddings"] = embeddings
                        enrolled.append(profile)
            except Exception as e:
                logger.warning(f"Failed to load profile {user_id}: {e}")
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
            raw_mean = profile.get("mean_embedding")
            mean_emb = None
            mean_cosine = None
            mean_angle = None
            if raw_mean is not None:
                mean_arr = np.asarray(raw_mean, dtype=np.float32).reshape(-1)
                if mean_arr.size == self.embedding_dim:
                    norm = np.linalg.norm(mean_arr) + 1e-6
                    mean_vec = mean_arr / norm
                    mean_cosine = float(np.clip(np.dot(probe_norm, mean_vec), -1.0, 1.0))
                    mean_angle = float(np.arccos(mean_cosine))
                    mean_emb = mean_vec
            
            sample_vecs = []
            for emb in profile.get("embeddings", []):
                arr = np.asarray(emb, dtype=np.float32).reshape(-1)
                if arr.size != self.embedding_dim:
                    continue
                norm = np.linalg.norm(arr)
                if norm < 1e-6:
                    continue
                sample_vecs.append(arr / norm)
            
            if not sample_vecs and mean_cosine is None:
                continue
            
            if not sample_vecs and mean_emb is not None:
                sample_vecs = [mean_emb]
            
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
                "user_id": profile["user_id"],
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
