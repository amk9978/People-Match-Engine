import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Set

import redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class RedisCache:
    """General-purpose Redis cache service"""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = None
        self._connect()

    def _connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)

            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.info(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def ping(self) -> bool:
        """Test Redis connection"""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.ping()
        except Exception:
            return False

    def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        if not self.redis_client:
            return None
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None

    def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set value in Redis"""
        if not self.redis_client:
            return False
        try:
            if ttl:
                return self.redis_client.setex(key, ttl, value)
            else:
                return self.redis_client.set(key, value)
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False

    def delete(self, *keys: str) -> int:
        """Delete keys from Redis"""
        if not self.redis_client or not keys:
            return 0
        try:
            return self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Error deleting keys {keys}: {e}")
            return 0

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking key {key}: {e}")
            return False

    def sadd(self, key: str, *values: str) -> int:
        """Add values to a set"""
        if not self.redis_client or not values:
            return 0
        try:
            return self.redis_client.sadd(key, *values)
        except Exception as e:
            logger.error(f"Error adding to set {key}: {e}")
            return 0

    def srem(self, key: str, *values: str) -> int:
        """Remove values from a set"""
        if not self.redis_client or not values:
            return 0
        try:
            return self.redis_client.srem(key, *values)
        except Exception as e:
            logger.error(f"Error removing from set {key}: {e}")
            return 0

    def smembers(self, key: str) -> Set:
        """Get all members of a set"""
        if not self.redis_client:
            return set()
        try:
            return self.redis_client.smembers(key)
        except Exception as e:
            logger.error(f"Error getting set members {key}: {e}")
            return set()

    def sismember(self, key: str, value: str) -> bool:
        """Check if value is member of set"""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.sismember(key, value)
        except Exception as e:
            logger.error(f"Error checking set membership {key}: {e}")
            return False

    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        if not self.redis_client:
            return []
        try:
            return self.redis_client.keys(pattern)
        except Exception as e:
            logger.error(f"Error getting keys with pattern {pattern}: {e}")
            return []

    def clear_all(self) -> bool:
        """Clear all keys (DANGEROUS - use with caution)"""
        if not self.redis_client:
            return False
        try:
            self.redis_client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

    def get_info(self) -> Dict:
        """Get Redis server info"""
        if not self.redis_client:
            return {"status": "disconnected"}

        try:
            info = self.redis_client.info()
            return {
                "status": "connected",
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
            }
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            return {"status": "error", "error": str(e)}


class RedisEmbeddingCache:
    def __init__(self, redis_url: str = None, key_prefix: str = "embeddings"):
        """Initialize Redis cache for embeddings"""
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.key_prefix = key_prefix
        self.redis_client = None
        self._connect()

    def _connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)

            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.info(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def _make_key(self, tag: str) -> str:
        """Create Redis key for a tag"""
        tag_hash = hashlib.md5(tag.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}:{tag_hash}"

    def get(self, tag: str) -> Optional[List[float]]:
        """Get embedding from Redis cache"""
        if not self.redis_client:
            return None

        try:
            key = self._make_key(tag)
            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.info(f"Error getting from cache: {e}")

        return None

    def set(self, tag: str, embedding: List[float], ttl: int = None):
        """Store embedding in Redis cache"""
        if not self.redis_client:
            return False

        try:
            key = self._make_key(tag)
            value = json.dumps(embedding)

            if ttl:
                self.redis_client.setex(key, ttl, value)
            else:
                self.redis_client.set(key, value)

            return True
        except Exception as e:
            logger.info(f"Error setting cache: {e}")
            return False

    def exists(self, tag: str) -> bool:
        """Check if tag exists in cache"""
        if not self.redis_client:
            return False

        try:
            key = self._make_key(tag)
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.info(f"Error checking cache: {e}")
            return False

    def get_cache_info(self) -> Dict:
        """Get cache statistics"""
        if not self.redis_client:
            return {"status": "disconnected"}

        try:
            info = self.redis_client.info("memory")
            keyspace = self.redis_client.info("keyspace")

            pattern = f"{self.key_prefix}:*"
            embedding_keys = len(list(self.redis_client.scan_iter(match=pattern)))

            return {
                "status": "connected",
                "memory_used": info.get("used_memory_human", "N/A"),
                "embedding_keys": embedding_keys,
                "total_keys": (
                    keyspace.get("db0", {}).get("keys", 0) if "db0" in keyspace else 0
                ),
            }
        except Exception as e:
            logger.info(f"Error getting cache info: {e}")
            return {"status": "error", "error": str(e)}

    def clear_embeddings(self):
        """Clear all embedding keys"""
        if not self.redis_client:
            return 0

        try:
            pattern = f"{self.key_prefix}:*"
            keys = list(self.redis_client.scan_iter(match=pattern))
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} embedding keys")
                return deleted
            return 0
        except Exception as e:
            logger.info(f"Error clearing cache: {e}")
            return 0
