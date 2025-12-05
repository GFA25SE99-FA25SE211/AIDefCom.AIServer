"""Multi-speaker tracking - State machine for managing concurrent speakers."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def _get_speaker_config():
    """Get speaker tracking config with fallback defaults."""
    try:
        from app.config import Config
        return {
            "max_speakers": Config.SPEAKER_MAX_CONCURRENT,
            "inactivity_timeout": Config.SPEAKER_INACTIVITY_TIMEOUT_SECONDS,
        }
    except Exception:
        return {
            "max_speakers": 4,
            "inactivity_timeout": 30.0,
        }


@dataclass
class SpeakerSegment:
    """Represents a speaking segment by one speaker."""
    
    speaker: str
    user_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    text: str = ""
    confidence: Optional[float] = None
    
    def duration(self) -> float:
        """Get segment duration in seconds."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "speaker": self.speaker,
            "user_id": self.user_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass
class SpeakerState:
    """State information for a single speaker."""
    
    speaker: str
    user_id: Optional[str]
    last_active: float
    total_duration: float = 0.0
    segment_count: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    
    def average_confidence(self) -> Optional[float]:
        """Get average confidence score."""
        if not self.confidence_scores:
            return None
        return sum(self.confidence_scores) / len(self.confidence_scores)


class MultiSpeakerTracker:
    """
    Track multiple concurrent speakers with state management.
    
    Supports:
    - 2-4 simultaneous speakers
    - Speaker transitions with history
    - Segment management
    - Speaker statistics
    """
    
    def __init__(self, max_speakers: int | None = None, inactivity_timeout: float | None = None):
        """
        Initialize multi-speaker tracker.
        
        Args:
            max_speakers: Maximum number of concurrent speakers to track (default from Config)
            inactivity_timeout: Time in seconds before speaker is considered inactive (default from Config)
        """
        config = _get_speaker_config()
        self.max_speakers = max_speakers if max_speakers is not None else config["max_speakers"]
        self.inactivity_timeout = inactivity_timeout if inactivity_timeout is not None else config["inactivity_timeout"]
        
        # Current active speaker
        self.active_speaker: str = "Khách"
        self.active_user_id: Optional[str] = None
        self.active_since: float = 0.0
        
        # Speaker history (recent speakers)
        self.recent_speakers: deque[str] = deque(maxlen=max_speakers)
        
        # Speaker states
        self.speaker_states: Dict[str, SpeakerState] = {}
        
        # Speaking segments
        self.segments: List[SpeakerSegment] = []
        self.current_segment: Optional[SpeakerSegment] = None
    
    def switch_speaker(
        self,
        new_speaker: str,
        new_user_id: Optional[str],
        timestamp: float,
        confidence: Optional[float] = None,
        reason: str = "unknown"
    ) -> bool:
        """
        Switch to a new speaker.
        
        Args:
            new_speaker: New speaker name
            new_user_id: New speaker user ID
            timestamp: Current timestamp
            confidence: Confidence score for identification
            reason: Reason for switch (for debugging)
        
        Returns:
            True if switched, False if same speaker
        """
        # No change
        if new_speaker == self.active_speaker and new_user_id == self.active_user_id:
            # Update last active time
            self._update_speaker_state(new_speaker, new_user_id, timestamp, confidence)
            return False
        
        # End current segment
        if self.current_segment:
            self.current_segment.end_time = timestamp
            self.segments.append(self.current_segment)
        
        # Switch speaker
        old_speaker = self.active_speaker
        self.active_speaker = new_speaker
        self.active_user_id = new_user_id
        self.active_since = timestamp
        
        # Update recent speakers
        if new_speaker not in self.recent_speakers:
            self.recent_speakers.append(new_speaker)
        
        # Start new segment
        self.current_segment = SpeakerSegment(
            speaker=new_speaker,
            user_id=new_user_id,
            start_time=timestamp,
            confidence=confidence,
        )
        
        # Update state
        self._update_speaker_state(new_speaker, new_user_id, timestamp, confidence)
        
        return True
    
    def append_text(self, text: str) -> None:
        """
        Append recognized text to current segment.
        
        Args:
            text: Recognized text to append
        """
        if self.current_segment:
            if self.current_segment.text:
                self.current_segment.text += " " + text
            else:
                self.current_segment.text = text
    
    def _update_speaker_state(
        self,
        speaker: str,
        user_id: Optional[str],
        timestamp: float,
        confidence: Optional[float]
    ) -> None:
        """Update speaker state with new activity."""
        if speaker not in self.speaker_states:
            self.speaker_states[speaker] = SpeakerState(
                speaker=speaker,
                user_id=user_id,
                last_active=timestamp,
            )
        
        state = self.speaker_states[speaker]
        state.last_active = timestamp
        state.segment_count += 1
        
        if confidence is not None:
            state.confidence_scores.append(confidence)
            # Keep only last 10 scores
            if len(state.confidence_scores) > 10:
                state.confidence_scores = state.confidence_scores[-10:]
    
    def get_active_speakers(self, timestamp: float) -> List[str]:
        """
        Get list of currently active speakers (spoke within timeout window).
        
        Args:
            timestamp: Current timestamp
        
        Returns:
            List of active speaker names
        """
        active = []
        for speaker, state in self.speaker_states.items():
            if timestamp - state.last_active <= self.inactivity_timeout:
                active.append(speaker)
        return active
    
    def get_speaker_stats(self, speaker: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific speaker.
        
        Args:
            speaker: Speaker name
        
        Returns:
            Dictionary of statistics or None if not found
        """
        if speaker not in self.speaker_states:
            return None
        
        state = self.speaker_states[speaker]
        
        # Calculate total speaking time from segments
        total_duration = sum(
            seg.duration() for seg in self.segments if seg.speaker == speaker
        )
        if self.current_segment and self.current_segment.speaker == speaker:
            # Add current segment duration
            from time import time
            total_duration += time() - self.current_segment.start_time
        
        return {
            "speaker": speaker,
            "user_id": state.user_id,
            "total_duration": total_duration,
            "segment_count": state.segment_count,
            "average_confidence": state.average_confidence(),
            "last_active": state.last_active,
        }
    
    def get_session_summary(self, timestamp: float) -> Dict[str, Any]:
        """
        Get summary of entire session.
        
        Args:
            timestamp: Current timestamp
        
        Returns:
            Session summary dictionary
        """
        active_speakers = self.get_active_speakers(timestamp)
        
        speaker_summaries = []
        for speaker in self.speaker_states.keys():
            stats = self.get_speaker_stats(speaker)
            if stats:
                speaker_summaries.append(stats)
        
        # Sort by total duration (descending)
        speaker_summaries.sort(key=lambda x: x["total_duration"], reverse=True)
        
        return {
            "active_speaker": self.active_speaker,
            "active_user_id": self.active_user_id,
            "total_segments": len(self.segments),
            "current_segment": self.current_segment.to_dict() if self.current_segment else None,
            "active_speakers": active_speakers,
            "speaker_count": len(self.speaker_states),
            "speakers": speaker_summaries,
        }
    
    def finalize(self, timestamp: float) -> List[SpeakerSegment]:
        """
        Finalize tracking and return all segments.
        
        Args:
            timestamp: Final timestamp
        
        Returns:
            List of all segments
        """
        # End current segment
        if self.current_segment:
            self.current_segment.end_time = timestamp
            self.segments.append(self.current_segment)
            self.current_segment = None
        
        return self.segments
