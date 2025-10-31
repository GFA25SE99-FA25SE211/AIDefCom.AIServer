"""Speech API schemas - WebSocket message types."""

from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel


class STTPartialEvent(BaseModel):
    """Partial STT result event."""
    type: Literal["partial"] = "partial"
    text: str
    speaker: str
    user_id: Optional[str] = None
    display: str


class STTResultEvent(BaseModel):
    """Final STT result event."""
    type: Literal["result"] = "result"
    text: str
    speaker: str
    user_id: Optional[str] = None
    display: str


class STTNoMatchEvent(BaseModel):
    """No match event."""
    type: Literal["nomatch"] = "nomatch"
    speaker: str


class STTErrorEvent(BaseModel):
    """Error event."""
    type: Literal["error"] = "error"
    error: str


class PingEvent(BaseModel):
    """Ping event to keep WebSocket alive."""
    type: Literal["ping"] = "ping"
