import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Set

from match_engine.services.cache.backend import CacheBackend

logger = logging.getLogger(__name__)


class Cache:
    """String cache over a backend, with one key derivation shared by every operation.

    A namespace hashes logical names into `<namespace>:<digest>` storage keys.
    Reads, writes, deletes and existence checks all derive the storage key the
    same way, so a name that was written can always be deleted.
    """

    def __init__(self, backend: CacheBackend, namespace: Optional[str] = None):
        self._backend = backend
        self._namespace = namespace

    @property
    def namespace(self) -> Optional[str]:
        return self._namespace

    def storage_key(self, name: str) -> str:
        if self._namespace is None:
            return name
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()
        return f"{self._namespace}:{digest}"

    def get(self, name: str) -> Optional[str]:
        return self._backend.get(self.storage_key(name))

    def get_many(self, names: List[str]) -> Dict[str, Optional[str]]:
        """Read many names in one backend round trip. Result is keyed by name."""
        by_storage_key = {self.storage_key(name): name for name in names}
        resolved = self._backend.get_many(list(by_storage_key.keys()))
        return {by_storage_key[key]: value for key, value in resolved.items()}

    def set(self, name: str, value: str, ttl: Optional[int] = None) -> bool:
        return self._backend.set(self.storage_key(name), value, ttl)

    def delete(self, name: str) -> int:
        return self._backend.delete([self.storage_key(name)])

    def exists(self, name: str) -> bool:
        return self._backend.exists(self.storage_key(name))

    def sadd(self, name: str, value: str) -> int:
        return self._backend.sadd(self.storage_key(name), [value])

    def srem(self, name: str, value: str) -> int:
        return self._backend.srem(self.storage_key(name), [value])

    def smembers(self, name: str) -> Set[str]:
        return self._backend.smembers(self.storage_key(name))

    def sismember(self, name: str, value: str) -> bool:
        return self._backend.sismember(self.storage_key(name), value)

    def delete_by_pattern(self, pattern: str) -> int:
        """Delete every key matching a glob pattern. Only valid without a namespace,
        because namespaced keys are digests and no pattern over names can match them."""
        assert (
            self._namespace is None
        ), "delete_by_pattern needs raw keys; use clear() on a namespaced cache"
        keys = self._backend.scan(pattern)
        return self._backend.delete(keys)

    def count_by_pattern(self, pattern: str) -> int:
        return len(self._backend.scan(pattern))

    def clear(self) -> int:
        """Delete every key in this cache's namespace."""
        if self._namespace is None:
            return self._backend.delete(self._backend.scan("*"))
        keys = self._backend.scan(f"{self._namespace}:*")
        return self._backend.delete(keys)

    def info(self) -> Dict[str, Any]:
        return self._backend.info()


class EmbeddingCache:
    """Vector cache keyed by the text that produced the vector."""

    def __init__(self, backend: CacheBackend, namespace: str = "embeddings"):
        self._cache = Cache(backend, namespace)

    def get(self, text: str) -> Optional[List[float]]:
        raw = self._cache.get(text)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Discarding malformed cached embedding for '{text[:60]}'")
            self._cache.delete(text)
            return None

    def set(self, text: str, embedding: List[float], ttl: Optional[int] = None) -> bool:
        return self._cache.set(text, json.dumps(embedding), ttl)

    def delete(self, text: str) -> int:
        return self._cache.delete(text)

    def exists(self, text: str) -> bool:
        return self._cache.exists(text)

    def clear(self) -> int:
        return self._cache.clear()

    def info(self) -> Dict[str, Any]:
        return self._cache.info()
