"""Redis Service - Azure Cache for Redis integration with async support."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis

from app.config import Config
from repositories.interfaces.i_redis_service import IRedisService

logger = logging.getLogger(__name__)


class RedisService(IRedisService):
    """Service for Azure Cache for Redis operations with async support."""
    
    def __init__(self) -> None:
        """Initialize Redis connection."""
        self.client: Optional[aioredis.Redis] = None
        self._connection_pool: Optional[aioredis.ConnectionPool] = None
        self._initialized = False
    
    async def _ensure_connection(self) -> None:
        """Ensure Redis connection is established (lazy init)."""
        if self._initialized:
            return
        
        self._initialized = True  # Mark as attempted to avoid retry loops
        
        if not Config.REDIS_PASSWORD:
            logger.warning("⚠️ REDIS_PASSWORD not set. Running without cache.")
            self.client = None
            return
        
        try:
            logger.info(f"🔌 Connecting to Redis: {Config.REDIS_HOST}:{Config.REDIS_PORT} (SSL: {Config.REDIS_SSL})")
            
            # Create client with from_url (rediss:// automatically enables SSL)
            redis_url = f"{'rediss' if Config.REDIS_SSL else 'redis'}://{Config.REDIS_HOST}:{Config.REDIS_PORT}/{Config.REDIS_DB}"
            
            self.client = await aioredis.from_url(
                redis_url,
                password=Config.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10,
                socket_keepalive=True,
                health_check_interval=30,
                retry_on_timeout=True,
            )
            
            # Test connection with timeout
            await asyncio.wait_for(self.client.ping(), timeout=15)
            logger.info(f"✅ Connected to Redis at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
            
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Redis connection timeout. Running without cache.")
            self.client = None
        except aioredis.AuthenticationError as e:
            logger.warning(f"⚠️ Redis authentication failed: {e}. Check REDIS_PASSWORD. Running without cache.")
            self.client = None
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Running without cache.")
            self.client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (async)."""
        if not self._initialized:
            await self._ensure_connection()
        
        if not self.client:
            return None
        
        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.debug(f"Redis get error for '{key}': {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache with TTL (async)."""
        if not self._initialized:
            await self._ensure_connection()
        
        if not self.client:
            return False
        
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            ttl_seconds = ttl if ttl is not None else Config.REDIS_TTL_SECONDS
            
            if ttl_seconds > 0:
                await self.client.setex(key, ttl_seconds, serialized)
            else:
                await self.client.set(key, serialized)
            
            return True
        except Exception as e:
            logger.debug(f"Redis set error for '{key}': {e}")
            return False
    
    async def delete(self, *keys: str) -> bool:
        """Delete keys from cache (async)."""
        if not self._initialized:
            await self._ensure_connection()
        
        if not self.client or not keys:
            return False
        
        try:
            await self.client.delete(*keys)
            return True
        except Exception as e:
            logger.debug(f"Redis delete error: {e}")
            return False
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if exists, False otherwise
        """
        if not self._initialized:
            await self._ensure_connection()
        
        if not self.client:
            return False
        
        try:
            return bool(await self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking key '{key}' in Redis: {e}")
            return False
    
    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment counter in cache.
        
        Args:
            key: Cache key
            amount: Amount to increment
            
        Returns:
            New value after increment or None if failed
        """
        if not self._initialized:
            await self._ensure_connection()
        
        if not self.client:
            return None
        
        try:
            return await self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing key '{key}' in Redis: {e}")
            return None
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds, -1 if no TTL, -2 if key doesn't exist, None if error
        """
        if not self._initialized:
            await self._ensure_connection()
        
        if not self.client:
            return None
        
        try:
            return await self.client.ttl(key)
        except Exception as e:
            logger.error(f"Error getting TTL for key '{key}' from Redis: {e}")
            return None
    
    async def set_hash(self, name: str, mapping: dict[str, Any]) -> bool:
        """
        Set hash in cache.
        
        Args:
            name: Hash name
            mapping: Dictionary to store
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            await self._ensure_connection()
        
        if not self.client:
            return False
        
        try:
            # Serialize values in mapping
            serialized_mapping = {k: json.dumps(v) for k, v in mapping.items()}
            await self.client.hset(name, mapping=serialized_mapping)
            return True
        except Exception as e:
            logger.error(f"Error setting hash '{name}' in Redis: {e}")
            return False
    
    async def get_hash(self, name: str) -> Optional[dict[str, Any]]:
        """
        Get hash from cache.
        
        Args:
            name: Hash name
            
        Returns:
            Dictionary or None if not found
        """
        if not self._initialized:
            await self._ensure_connection()
        
        if not self.client:
            return None
        
        try:
            hash_data = await self.client.hgetall(name)
            if not hash_data:
                return None
            
            # Deserialize values
            return {k: json.loads(v) for k, v in hash_data.items()}
        except Exception as e:
            logger.error(f"Error getting hash '{name}' from Redis: {e}")
            return None
    
    async def lpush(self, key: str, *values: Any) -> Optional[int]:
        """
        Push values to the head of a list in cache.
        
        Args:
            key: List key
            values: Values to push (will be JSON-serialized)
            
        Returns:
            Length of list after push, or None if failed
        """
        if not self._initialized:
            await self._ensure_connection()
        
        if not self.client:
            return None
        
        try:
            # Serialize all values
            serialized_values = [json.dumps(v) for v in values]
            return await self.client.lpush(key, *serialized_values)
        except Exception as e:
            logger.error(f"Error pushing to list '{key}' in Redis: {e}")
            return None
    
    async def publish(self, channel: str, message: Any) -> bool:
        """
        Publish message to a Redis Pub/Sub channel.
        
        Args:
            channel: Channel name (e.g., 'transcript:session:123')
            message: Message to publish (will be JSON-serialized)
            
        Returns:
            True if published successfully, False otherwise
        """
        if not self._initialized:
            await self._ensure_connection()
        
        if not self.client:
            return False
        
        try:
            serialized = json.dumps(message, ensure_ascii=False)
            await self.client.publish(channel, serialized)
            return True
        except Exception as e:
            logger.debug(f"Redis publish error for channel '{channel}': {e}")
            return False
    
    async def subscribe(self, channel: str):
        """
        Subscribe to a Redis Pub/Sub channel.
        
        Args:
            channel: Channel name to subscribe to
            
        Returns:
            PubSub object for receiving messages, or None if failed
        """
        if not self._initialized:
            await self._ensure_connection()
        
        if not self.client:
            return None
        
        try:
            pubsub = self.client.pubsub()
            await pubsub.subscribe(channel)
            return pubsub
        except Exception as e:
            logger.error(f"Redis subscribe error for channel '{channel}': {e}")
            return None


# Global Redis service instance
_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    """Get or create global Redis service instance."""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service
