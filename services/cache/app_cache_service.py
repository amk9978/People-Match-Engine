import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from services.cache.backend import CacheBackend
from services.cache.cache import Cache, EmbeddingCache
from services.cache.factory import get_cache_backend

logger = logging.getLogger(__name__)

PERSON_EMBEDDING_PREFIX = "person_embedding"
PAIR_COMPLEMENTARITY_PREFIX = "complementarity"
PROFILE_DIGEST_CHARS = 16


def _digest(profile: str) -> str:
    return hashlib.md5(profile.encode("utf-8")).hexdigest()[:PROFILE_DIGEST_CHARS]


def _parse_score(raw: str, key: str, store: Cache) -> Optional[float]:
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Discarding malformed cached score at {key}")
        store.delete(key)
        return None


@dataclass(frozen=True)
class ComplementarityCacheStatus:
    """Which target-comparison pairs already have a score and which still need one."""

    cached: Dict[str, Dict[str, float]]
    missing: Dict[str, List[str]]


class AppCacheService:
    """Caching for per-person embeddings and per-profile complementarity scores.

    Keys are explicit and human-readable so they can be scanned and cleared by
    pattern. Text embeddings go to a separate namespaced cache keyed by content.
    """

    def __init__(self, backend: CacheBackend = None):
        self._backend = backend
        self._store: Optional[Cache] = None
        self._embeddings: Optional[EmbeddingCache] = None

    @property
    def store(self) -> Cache:
        if self._store is None:
            self._store = Cache(self._resolve_backend())
        return self._store

    @property
    def embeddings(self) -> EmbeddingCache:
        if self._embeddings is None:
            self._embeddings = EmbeddingCache(self._resolve_backend())
        return self._embeddings

    def _resolve_backend(self) -> CacheBackend:
        if self._backend is None:
            self._backend = get_cache_backend()
        return self._backend

    def _get_row_hash(self, row_data: Dict[str, Any]) -> str:
        content = json.dumps(row_data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _get_person_cache_key(self, row_hash: str, feature_type: str) -> str:
        return f"{PERSON_EMBEDDING_PREFIX}:{feature_type}:{row_hash}"

    def get_person_embedding(
        self, row_data: Dict[str, Any], feature_type: str
    ) -> Optional[List[float]]:
        cache_key = self._get_person_cache_key(
            self._get_row_hash(row_data), feature_type
        )
        cached_data = self.store.get(cache_key)
        if cached_data is None:
            return None
        try:
            return json.loads(cached_data)
        except json.JSONDecodeError:
            logger.warning(f"Discarding malformed cached embedding at {cache_key}")
            self.store.delete(cache_key)
            return None

    def set_person_embedding(
        self, row_data: Dict[str, Any], feature_type: str, embedding: List[float]
    ) -> bool:
        cache_key = self._get_person_cache_key(
            self._get_row_hash(row_data), feature_type
        )
        return self.store.set(cache_key, json.dumps(embedding))

    def get_dataset_embedding_cache_status(
        self, df: pd.DataFrame, feature_type: str, column: str
    ) -> Dict[str, Any]:
        """Split the dataset into people whose embedding is cached and those who need one.

        Both halves are keyed by row position, matching how the rest of the
        pipeline addresses people."""
        cached_embeddings = {}
        uncached_indices = []

        for position in range(len(df)):
            row_data = self._extract_row_data(df.iloc[position], feature_type, column)
            embedding = self.get_person_embedding(row_data, feature_type)
            if embedding:
                cached_embeddings[position] = embedding
            else:
                uncached_indices.append(position)

        logger.info(
            f"Feature {feature_type}: {len(cached_embeddings)} cached embeddings, "
            f"{len(uncached_indices)} to compute"
        )

        return {
            "cached_embeddings": cached_embeddings,
            "uncached_indices": uncached_indices,
        }

    def cache_dataset_embeddings(
        self,
        df: pd.DataFrame,
        feature_type: str,
        column: str,
        embeddings: Dict[int, List[float]],
    ) -> None:
        cached_count = 0

        for position, embedding in embeddings.items():
            if position >= len(df):
                continue
            row_data = self._extract_row_data(df.iloc[position], feature_type, column)
            if self.set_person_embedding(row_data, feature_type, embedding):
                cached_count += 1

        logger.info(f"Cached {cached_count} new {feature_type} embeddings")

    def _extract_row_data(
        self, row: pd.Series, feature_type: str, column: str
    ) -> Dict[str, Any]:
        """The row content a feature's embedding depends on.

        The column travels with the feature name so a cached vector belongs to
        one column of one dataset, not to a feature name that another dataset
        might reuse for different text."""
        return {"column": column, "value": str(row.get(column, ""))}

    def _pair_key(self, category: str, source: str, target: str) -> str:
        """Key one ordered pair of profiles for one feature.

        Pairs are keyed individually so a score survives any change to the rest
        of the roster. Keying a whole comparison set together meant adding one
        person invalidated every previously scored row."""
        return (
            f"{PAIR_COMPLEMENTARITY_PREFIX}:{category}:"
            f"{_digest(source)}:{_digest(target)}"
        )

    def get_pair_score(
        self, category: str, source: str, target: str
    ) -> Optional[float]:
        """Read one pair's score, falling back to the transpose when only it was scored."""
        for key in (
            self._pair_key(category, source, target),
            self._pair_key(category, target, source),
        ):
            raw = self.store.get(key)
            if raw is not None:
                return _parse_score(raw, key, self.store)
        return None

    def set_pair_score(
        self, category: str, source: str, target: str, score: float
    ) -> bool:
        return self.store.set(self._pair_key(category, source, target), repr(score))

    def get_complementarity_cache_status(
        self, target_profiles: List[str], comparison_profiles: List[str], category: str
    ) -> ComplementarityCacheStatus:
        """Split every target-comparison pair into what is cached and what is not.

        Self-comparison is never requested. Reads go out in one batch per
        direction, so the cost is two round trips rather than one per pair."""
        wanted = [
            (target, comparison)
            for target in target_profiles
            for comparison in comparison_profiles
            if target != comparison
        ]

        forward = {
            pair: self._pair_key(category, pair[0], pair[1]) for pair in wanted
        }
        reverse = {
            pair: self._pair_key(category, pair[1], pair[0]) for pair in wanted
        }
        stored = self.store.get_many(list(forward.values()))
        stored.update(self.store.get_many(list(reverse.values())))

        cached: Dict[str, Dict[str, float]] = {}
        missing: Dict[str, List[str]] = {}

        for pair in wanted:
            target, comparison = pair
            raw = stored.get(forward[pair])
            if raw is None:
                raw = stored.get(reverse[pair])

            score = None
            if raw is not None:
                score = _parse_score(raw, forward[pair], self.store)

            if score is None:
                missing.setdefault(target, []).append(comparison)
            else:
                cached.setdefault(target, {})[comparison] = score

        logger.info(
            f"Feature {category}: {sum(len(row) for row in cached.values())} cached pairs, "
            f"{sum(len(row) for row in missing.values())} to score"
        )
        return ComplementarityCacheStatus(cached=cached, missing=missing)

    def cache_complementarity_scores(
        self, results: Dict[str, Dict[str, float]], category: str
    ) -> int:
        stored = 0
        for source, scores in results.items():
            for target, score in scores.items():
                if source == target:
                    continue
                if self.set_pair_score(category, source, target, score):
                    stored += 1

        logger.info(f"Feature {category}: cached {stored} complementarity pairs")
        return stored

    def clear_complementarity(self, category: str = None) -> int:
        if category:
            pattern = f"{PAIR_COMPLEMENTARITY_PREFIX}:{category}:*"
        else:
            pattern = f"{PAIR_COMPLEMENTARITY_PREFIX}:*"
        return self.store.delete_by_pattern(pattern)

    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        return self.embeddings.get(text)

    def set_text_embedding(self, text: str, embedding: List[float]) -> bool:
        return self.embeddings.set(text, embedding)

    def clear_person_embeddings(self, feature_type: str = None) -> int:
        if feature_type:
            pattern = f"{PERSON_EMBEDDING_PREFIX}:{feature_type}:*"
        else:
            pattern = f"{PERSON_EMBEDDING_PREFIX}:*"
        return self.store.delete_by_pattern(pattern)

    def get_cache_stats(self) -> Dict[str, Any]:
        stats = self.store.info()
        stats.update(
            {
                "person_embedding_keys": self.store.count_by_pattern(
                    f"{PERSON_EMBEDDING_PREFIX}:*"
                ),
                "complementarity_pair_keys": self.store.count_by_pattern(
                    f"{PAIR_COMPLEMENTARITY_PREFIX}:*"
                ),
            }
        )
        return stats


app_cache_service = AppCacheService()
