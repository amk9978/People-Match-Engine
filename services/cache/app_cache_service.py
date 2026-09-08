import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from services.cache.backend import CacheBackend
from services.cache.cache import Cache, EmbeddingCache
from services.cache.factory import get_cache_backend

logger = logging.getLogger(__name__)

PERSON_EMBEDDING_PREFIX = "person_embedding"
PROFILE_COMPLEMENTARITY_PREFIX = "profile_complementarity"


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

    def _get_complementarity_cache_key(
        self, target_profile: str, comparison_profiles: List[str], category: str
    ) -> str:
        target_hash = hashlib.md5(target_profile.encode()).hexdigest()[:8]
        comparison_hash = hashlib.md5(
            str(sorted(comparison_profiles)).encode()
        ).hexdigest()[:8]
        return (
            f"{PROFILE_COMPLEMENTARITY_PREFIX}:{category}:"
            f"{target_hash}:vs:{comparison_hash}"
        )

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
        self, df: pd.DataFrame, feature_type: str
    ) -> Dict[str, Any]:
        """Split the dataset into people whose embedding is cached and those who need one."""
        cached_embeddings = {}
        uncached_indices = []

        for idx, row in df.iterrows():
            row_data = self._extract_row_data(row, feature_type)
            embedding = self.get_person_embedding(row_data, feature_type)
            if embedding:
                cached_embeddings[idx] = embedding
            else:
                uncached_indices.append(idx)

        logger.info(
            f"Feature {feature_type}: {len(cached_embeddings)} cached embeddings, "
            f"{len(uncached_indices)} to compute"
        )

        return {
            "cached_embeddings": cached_embeddings,
            "uncached_indices": uncached_indices,
        }

    def cache_dataset_embeddings(
        self, df: pd.DataFrame, feature_type: str, embeddings: Dict[int, List[float]]
    ) -> None:
        cached_count = 0

        for idx, embedding in embeddings.items():
            if idx not in df.index:
                continue
            row_data = self._extract_row_data(df.loc[idx], feature_type)
            if self.set_person_embedding(row_data, feature_type, embedding):
                cached_count += 1

        logger.info(f"Cached {cached_count} new {feature_type} embeddings")

    def _extract_row_data(self, row: pd.Series, feature_type: str) -> Dict[str, Any]:
        """Extract the row content a feature's embedding depends on."""
        if feature_type == "role":
            return {
                "role": str(row.get("Professional Identity - Role Specification", ""))
            }
        elif feature_type == "experience":
            return {
                "experience": str(
                    row.get("Professional Identity - Experience Level", "")
                )
            }
        elif feature_type == "persona":
            return {"personas": str(row.get("All Persona Titles", ""))}
        elif feature_type == "industry":
            return {
                "industry": str(
                    row.get("Company Identity - Industry Classification", "")
                )
            }
        elif feature_type == "market":
            return {"market": str(row.get("Company Market - Market Traction", ""))}
        elif feature_type == "offering":
            return {
                "offering": str(row.get("Company Offering - Value Proposition", ""))
            }
        else:
            return {col: str(row.get(col, "")) for col in row.index}

    def get_profile_complementarity(
        self, target_profile: str, comparison_profiles: List[str], category: str
    ) -> Optional[Dict[str, float]]:
        cache_key = self._get_complementarity_cache_key(
            target_profile, comparison_profiles, category
        )
        cached_data = self.store.get(cache_key)
        if cached_data is None:
            return None
        try:
            return json.loads(cached_data)
        except json.JSONDecodeError:
            logger.warning(f"Discarding malformed cached complementarity at {cache_key}")
            self.store.delete(cache_key)
            return None

    def set_profile_complementarity(
        self,
        target_profile: str,
        comparison_profiles: List[str],
        category: str,
        scores: Dict[str, float],
    ) -> bool:
        cache_key = self._get_complementarity_cache_key(
            target_profile, comparison_profiles, category
        )
        return self.store.set(cache_key, json.dumps(scores))

    def get_dataset_complementarity_cache_status(
        self, target_profiles: List[str], comparison_profiles: List[str], category: str
    ) -> Dict[str, Any]:
        """Split target profiles into those with cached scores and those needing computation."""
        cached_results = {}
        uncached_targets = []

        for target in target_profiles:
            cached_scores = self.get_profile_complementarity(
                target, comparison_profiles, category
            )
            if cached_scores:
                cached_results[target] = cached_scores
            else:
                uncached_targets.append(target)

        logger.info(
            f"Feature {category}: {len(cached_results)} cached complementarity rows, "
            f"{len(uncached_targets)} to compute"
        )

        return {
            "cached_results": cached_results,
            "uncached_targets": uncached_targets,
            "comparison_profiles": comparison_profiles,
        }

    def cache_dataset_complementarity_results(
        self,
        results: Dict[str, Dict[str, float]],
        comparison_profiles: List[str],
        category: str,
    ) -> None:
        cached_count = 0

        for target_profile, scores in results.items():
            if self.set_profile_complementarity(
                target_profile, comparison_profiles, category, scores
            ):
                cached_count += 1

        logger.info(f"Cached {cached_count} new {category} complementarity rows")

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

    def clear_profile_complementarity(self, category: str = None) -> int:
        if category:
            pattern = f"{PROFILE_COMPLEMENTARITY_PREFIX}:{category}:*"
        else:
            pattern = f"{PROFILE_COMPLEMENTARITY_PREFIX}:*"
        return self.store.delete_by_pattern(pattern)

    def get_cache_stats(self) -> Dict[str, Any]:
        stats = self.store.info()
        stats.update(
            {
                "person_embedding_keys": self.store.count_by_pattern(
                    f"{PERSON_EMBEDDING_PREFIX}:*"
                ),
                "profile_complementarity_keys": self.store.count_by_pattern(
                    f"{PROFILE_COMPLEMENTARITY_PREFIX}:*"
                ),
            }
        )
        return stats


app_cache_service = AppCacheService()
