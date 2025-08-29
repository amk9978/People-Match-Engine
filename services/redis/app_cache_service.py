import hashlib
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from services.redis.redis_cache import RedisEmbeddingCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class AppCacheService:
    """Unified caching service for embeddings and profile complementarity with row-based overlap handling"""

    def __init__(self):
        self.cache = RedisEmbeddingCache()

    def _get_row_hash(self, row_data: Dict[str, Any]) -> str:
        """Generate consistent hash for a row based on its content"""
        content = json.dumps(row_data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _get_person_cache_key(self, row_hash: str, feature_type: str) -> str:
        """Generate cache key for person-level data"""
        return f"person_embedding:{feature_type}:{row_hash}"

    def _get_complementarity_cache_key(
        self, target_profile: str, comparison_profiles: List[str], category: str
    ) -> str:
        """Generate cache key for profile complementarity (includes comparison set)"""
        target_hash = hashlib.md5(target_profile.encode()).hexdigest()[:8]
        comparison_hash = hashlib.md5(
            str(sorted(comparison_profiles)).encode()
        ).hexdigest()[:8]
        return f"profile_complementarity_{category}_{target_hash}_vs_{comparison_hash}"

    # Row-based embedding caching
    def get_person_embedding(
        self, row_data: Dict[str, Any], feature_type: str
    ) -> Optional[List[float]]:
        """Get cached embedding for a person's feature"""
        row_hash = self._get_row_hash(row_data)
        cache_key = self._get_person_cache_key(row_hash, feature_type)

        cached_data = self.cache.redis_client.get(cache_key)
        if cached_data:
            try:
                return json.loads(cached_data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid cached embedding for {cache_key}")
                return None
        return None

    def set_person_embedding(
        self, row_data: Dict[str, Any], feature_type: str, embedding: List[float]
    ) -> bool:
        """Cache embedding for a person's feature"""
        row_hash = self._get_row_hash(row_data)
        cache_key = self._get_person_cache_key(row_hash, feature_type)

        try:
            value = json.dumps(embedding)
            return self.cache.redis_client.set(cache_key, value)
        except Exception as e:
            logger.error(f"Error caching person embedding: {e}")
            return False

    def get_cached_people_embeddings(
        self, df: pd.DataFrame, feature_type: str
    ) -> Dict[int, List[float]]:
        """Get cached embeddings for people in dataset, returns {row_index: embedding}"""
        cached_embeddings = {}

        for idx, row in df.iterrows():
            row_data = self._extract_row_data(row, feature_type)
            embedding = self.get_person_embedding(row_data, feature_type)
            if embedding:
                cached_embeddings[idx] = embedding

        logger.info(f"Found {len(cached_embeddings)} cached {feature_type} embeddings out of {len(df)} people")
        return cached_embeddings

    def cache_people_embeddings(
        self, df: pd.DataFrame, feature_type: str, embeddings: Dict[int, List[float]]
    ) -> None:
        """Cache embeddings for people in dataset"""
        cached_count = 0

        for idx, embedding in embeddings.items():
            if idx < len(df):
                row = df.iloc[idx]
                row_data = self._extract_row_data(row, feature_type)
                if self.set_person_embedding(row_data, feature_type, embedding):
                    cached_count += 1

        logger.info(f"Cached {cached_count} new {feature_type} embeddings")

    def get_dataset_embedding_cache_status(
        self, df: pd.DataFrame, feature_type: str
    ) -> Dict[str, Any]:
        """Check which people in dataset have cached embeddings vs which need computing"""
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
            f"Found {len(cached_embeddings)} cached {feature_type} embeddings, {len(uncached_indices)} need computing"
        )

        return {
            "cached_embeddings": cached_embeddings,
            "uncached_indices": uncached_indices,
        }

    def cache_dataset_embeddings(
        self, df: pd.DataFrame, feature_type: str, embeddings: Dict[int, List[float]]
    ) -> None:
        """Cache embeddings for people in dataset (same as cache_people_embeddings but consistent naming)"""
        self.cache_people_embeddings(df, feature_type, embeddings)

    def _extract_row_data(self, row: pd.Series, feature_type: str) -> Dict[str, Any]:
        """Extract relevant row data for caching key generation"""
        if feature_type == "role":
            return {
                "role": str(
                    row.get("Professional Identity - Role Specification", "")
                )
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
            # Fallback: use all columns for unknown feature types
            return {col: str(row.get(col, "")) for col in row.index}

    # Profile complementarity caching
    def get_profile_complementarity(
        self, target_profile: str, comparison_profiles: List[str], category: str
    ) -> Optional[Dict[str, float]]:
        """Get cached complementarity scores for a target profile against specific comparison profiles"""
        cache_key = self._get_complementarity_cache_key(
            target_profile, comparison_profiles, category
        )

        cached_data = self.cache.redis_client.get(cache_key)
        if cached_data:
            try:
                return json.loads(cached_data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid cached complementarity for {cache_key}")
                return None
        return None

    def set_profile_complementarity(
        self,
        target_profile: str,
        comparison_profiles: List[str],
        category: str,
        scores: Dict[str, float],
    ) -> bool:
        """Cache complementarity scores for a target profile against specific comparison profiles"""
        cache_key = self._get_complementarity_cache_key(
            target_profile, comparison_profiles, category
        )

        try:
            value = json.dumps(scores)
            return self.cache.redis_client.set(cache_key, value)
        except Exception as e:
            logger.error(f"Error caching profile complementarity: {e}")
            return False

    def get_dataset_complementarity_cache_status(
        self, target_profiles: List[str], comparison_profiles: List[str], category: str
    ) -> Dict[str, Any]:
        """Check which target profiles have cached complementarity vs which need computing"""
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
            f"Found {len(cached_results)} cached {category} complementarity profiles, {len(uncached_targets)} need computing"
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
        """Cache complementarity results for a dataset"""
        cached_count = 0

        for target_profile, scores in results.items():
            if self.set_profile_complementarity(
                target_profile, comparison_profiles, category, scores
            ):
                cached_count += 1

        logger.info(f"Cached {cached_count} new {category} complementarity profiles")

    # Individual text embedding (existing functionality)
    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for individual text"""
        return self.cache.get(text)

    def set_text_embedding(self, text: str, embedding: List[float]) -> bool:
        """Cache embedding for individual text"""
        return self.cache.set(text, embedding)

    # Utility methods
    def clear_person_embeddings(self, feature_type: str = None) -> int:
        """Clear cached person embeddings"""
        if feature_type:
            pattern = f"person_embedding:{feature_type}:*"
        else:
            pattern = "person_embedding:*"
        return self.cache.delete_by_pattern(pattern)

    def clear_profile_complementarity(self, category: str = None) -> int:
        """Clear cached profile complementarity"""
        if category:
            pattern = f"profile_comp:{category}:*"
        else:
            pattern = "profile_comp:*"
        return self.cache.delete_by_pattern(pattern)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        base_stats = self.cache.get_cache_info()

        try:
            person_embedding_keys = len(
                list(self.cache.redis_client.scan_iter(match="person_embedding:*"))
            )
            profile_comp_keys = len(
                list(self.cache.redis_client.scan_iter(match="profile_comp:*"))
            )

            base_stats.update(
                {
                    "person_embedding_keys": person_embedding_keys,
                    "profile_complementarity_keys": profile_comp_keys,
                }
            )
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")

        return base_stats


# Global instance for easy import
app_cache_service = AppCacheService()
