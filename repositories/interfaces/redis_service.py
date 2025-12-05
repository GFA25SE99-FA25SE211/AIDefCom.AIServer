"""Redis Service Interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional


class IRedisService(ABC):
    """Interface for Redis caching operations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value from Redis."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in Redis with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> int:
        """Delete key from Redis."""
        pass
