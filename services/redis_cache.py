
import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

import redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


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
            # Test connection
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

            # Count our embedding keys
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
