import logging
import threading
from typing import Optional

from match_engine import settings
from match_engine.services.cache.backend import CacheBackend
from match_engine.services.cache.memory import InMemoryBackend
from match_engine.services.cache.redis_backend import RedisBackend, RedisUnavailable

logger = logging.getLogger(__name__)

_backend: Optional[CacheBackend] = None
_backend_lock = threading.Lock()


def create_cache_backend(redis_url: Optional[str]) -> CacheBackend:
    """Return a Redis backend when one is reachable, otherwise an in-memory one."""
    if not redis_url:
        logger.info("No Redis URL configured, using in-memory cache")
        return InMemoryBackend()

    try:
        backend = RedisBackend(redis_url)
    except RedisUnavailable as error:
        logger.warning(f"{error}, falling back to in-memory cache")
        return InMemoryBackend()

    logger.info(f"Connected to Redis at {redis_url}")
    return backend


def get_cache_backend() -> CacheBackend:
    """Return the process-wide cache backend, creating it on first use.

    Services share one backend so an in-memory deployment behaves like a single
    Redis database rather than one isolated store per service.
    """
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = create_cache_backend(settings.REDIS_URL)
    return _backend


def set_cache_backend(backend: CacheBackend) -> None:
    """Replace the process-wide backend. Intended for tests and CLI entry points."""
    global _backend
    with _backend_lock:
        _backend = backend
