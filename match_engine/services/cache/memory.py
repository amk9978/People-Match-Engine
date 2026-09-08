import fnmatch
import threading
import time
from typing import Any, Dict, List, Optional, Set


class InMemoryBackend:
    """Process-local cache backend used when Redis is absent.

    Entries live for the lifetime of the process. Expiry is evaluated lazily on
    read, so an expired key occupies memory until it is next touched.
    """

    def __init__(self) -> None:
        self._values: Dict[str, str] = {}
        self._sets: Dict[str, Set[str]] = {}
        self._expiry: Dict[str, float] = {}
        self._lock = threading.RLock()

    def _expired(self, key: str) -> bool:
        deadline = self._expiry.get(key)
        if deadline is None:
            return False
        return deadline <= time.monotonic()

    def _drop_if_expired(self, key: str) -> None:
        if self._expired(key):
            self._values.pop(key, None)
            self._sets.pop(key, None)
            self._expiry.pop(key, None)

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            self._drop_if_expired(key)
            return self._values.get(key)

    def get_many(self, keys: List[str]) -> Dict[str, Optional[str]]:
        with self._lock:
            resolved = {}
            for key in keys:
                self._drop_if_expired(key)
                resolved[key] = self._values.get(key)
            return resolved

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        assert isinstance(value, str), f"cache values must be str, got {type(value)}"
        with self._lock:
            self._values[key] = value
            if ttl is None:
                self._expiry.pop(key, None)
            else:
                self._expiry[key] = time.monotonic() + ttl
            return True

    def delete(self, keys: List[str]) -> int:
        with self._lock:
            deleted = 0
            for key in keys:
                self._drop_if_expired(key)
                if self._values.pop(key, None) is not None:
                    deleted += 1
                elif self._sets.pop(key, None) is not None:
                    deleted += 1
                self._expiry.pop(key, None)
            return deleted

    def exists(self, key: str) -> bool:
        with self._lock:
            self._drop_if_expired(key)
            return key in self._values or key in self._sets

    def scan(self, pattern: str) -> List[str]:
        with self._lock:
            candidates = list(self._values.keys()) + list(self._sets.keys())
            live = [key for key in candidates if not self._expired(key)]
            return [key for key in live if fnmatch.fnmatchcase(key, pattern)]

    def sadd(self, key: str, values: List[str]) -> int:
        with self._lock:
            self._drop_if_expired(key)
            members = self._sets.setdefault(key, set())
            before = len(members)
            members.update(values)
            return len(members) - before

    def srem(self, key: str, values: List[str]) -> int:
        with self._lock:
            self._drop_if_expired(key)
            members = self._sets.get(key)
            if members is None:
                return 0
            before = len(members)
            members.difference_update(values)
            if not members:
                self._sets.pop(key, None)
            return before - len(members)

    def smembers(self, key: str) -> Set[str]:
        with self._lock:
            self._drop_if_expired(key)
            return set(self._sets.get(key, set()))

    def sismember(self, key: str, value: str) -> bool:
        with self._lock:
            self._drop_if_expired(key)
            return value in self._sets.get(key, set())

    def flush(self) -> bool:
        with self._lock:
            self._values.clear()
            self._sets.clear()
            self._expiry.clear()
            return True

    def info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": "connected",
                "backend": "memory",
                "value_keys": len(self._values),
                "set_keys": len(self._sets),
            }
