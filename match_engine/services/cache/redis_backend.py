import logging
from typing import Any, Dict, List, Optional, Set

import redis

logger = logging.getLogger(__name__)

MGET_CHUNK = 1000


class RedisUnavailable(RuntimeError):
    """Raised when a Redis server cannot be reached at construction time."""


class RedisBackend:
    """Cache backend backed by a Redis server.

    The constructor pings the server so an unreachable Redis fails here rather
    than turning every later call into a silent no-op.
    """

    def __init__(self, redis_url: str) -> None:
        try:
            self._client = redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
        except Exception as error:
            raise RedisUnavailable(f"cannot reach Redis at {redis_url}") from error
        self._redis_url = redis_url

    def get(self, key: str) -> Optional[str]:
        return self._client.get(key)

    def get_many(self, keys: List[str]) -> Dict[str, Optional[str]]:
        resolved: Dict[str, Optional[str]] = {}
        for start in range(0, len(keys), MGET_CHUNK):
            chunk = keys[start : start + MGET_CHUNK]
            resolved.update(zip(chunk, self._client.mget(chunk)))
        return resolved

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        assert isinstance(value, str), f"cache values must be str, got {type(value)}"
        if ttl is None:
            return bool(self._client.set(key, value))
        return bool(self._client.setex(key, ttl, value))

    def delete(self, keys: List[str]) -> int:
        if not keys:
            return 0
        return int(self._client.delete(*keys))

    def exists(self, key: str) -> bool:
        return self._client.exists(key) > 0

    def scan(self, pattern: str) -> List[str]:
        return list(self._client.scan_iter(match=pattern))

    def sadd(self, key: str, values: List[str]) -> int:
        if not values:
            return 0
        return int(self._client.sadd(key, *values))

    def srem(self, key: str, values: List[str]) -> int:
        if not values:
            return 0
        return int(self._client.srem(key, *values))

    def smembers(self, key: str) -> Set[str]:
        return set(self._client.smembers(key))

    def sismember(self, key: str, value: str) -> bool:
        return bool(self._client.sismember(key, value))

    def flush(self) -> bool:
        self._client.flushdb()
        return True

    def info(self) -> Dict[str, Any]:
        server = self._client.info()
        return {
            "status": "connected",
            "backend": "redis",
            "redis_version": server.get("redis_version"),
            "used_memory_human": server.get("used_memory_human"),
            "connected_clients": server.get("connected_clients"),
            "keyspace_hits": server.get("keyspace_hits"),
            "keyspace_misses": server.get("keyspace_misses"),
        }
