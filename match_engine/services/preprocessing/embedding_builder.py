import asyncio
import logging
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from match_engine.services.cache.app_cache_service import app_cache_service
from match_engine.services.features.feature_set import Feature, FeatureSet
from match_engine.services.preprocessing.embedding_interface import (
    EmbeddingServiceProtocol,
)
from match_engine.services.preprocessing.fast_embedding_service import (
    FastEmbeddingService,
)

logger = logging.getLogger(__name__)


class EmbeddingBuilder:
    """Turns one text column per feature into one vector per person.

    People are addressed by position. Row `i` of a feature's matrix is person `i`
    of the loaded frame, which the loader guarantees is indexed 0..n-1.
    """

    def __init__(
        self,
        cache=None,
        embedding_service: EmbeddingServiceProtocol = None,
    ):
        self.cache = cache or app_cache_service
        self.embedding_service = embedding_service or FastEmbeddingService()

    @property
    def embedding_dim(self) -> int:
        return self.embedding_service.embedding_dim

    async def get_cached_embedding(self, tag: str) -> List[float]:
        return await self.embedding_service.get_embedding(tag)

    async def embed_features(
        self, df: pd.DataFrame, feature_set: FeatureSet
    ) -> Dict[str, np.ndarray]:
        assert df.index.equals(
            pd.RangeIndex(len(df))
        ), "people are addressed by position; the frame must be indexed 0..n-1"

        feature_embeddings = {}
        for feature in feature_set:
            feature_embeddings[feature.name] = await self._embed_one_feature(
                df, feature
            )

        logger.info(f"Cache after embedding: {self.cache.get_cache_stats()}")
        return feature_embeddings

    async def _embed_one_feature(
        self, df: pd.DataFrame, feature: Feature
    ) -> np.ndarray:
        feature_name = feature.name
        cache_status = self.cache.get_dataset_embedding_cache_status(
            df, feature_name, feature.column
        )
        person_embeddings = dict(cache_status["cached_embeddings"])
        uncached_positions = cache_status["uncached_indices"]

        if uncached_positions:
            values = self._collect_values(df, uncached_positions, feature)
            value_embeddings = await self._embed_values(values)

            computed = {
                position: self._person_vector(
                    feature.split(str(df.iloc[position][feature.column])),
                    value_embeddings,
                )
                for position in uncached_positions
            }
            person_embeddings.update(computed)
            self.cache.cache_dataset_embeddings(
                df, feature_name, feature.column, computed
            )

        matrix = self._assemble(df, person_embeddings, feature_name)
        logger.info(
            f"Feature {feature_name}: {matrix.shape[0]} people, {matrix.shape[1]} dimensions"
        )
        return matrix

    def _collect_values(
        self, df: pd.DataFrame, positions: List[int], feature: Feature
    ) -> Set[str]:
        values = set()
        for position in positions:
            values.update(feature.split(str(df.iloc[position][feature.column])))
        return values

    async def _embed_values(self, values: Set[str]) -> Dict[str, List[float]]:
        """Resolve every distinct tag to a vector, reusing whatever the cache holds."""
        embeddings = {}
        missing = []

        for value in values:
            if not value.strip():
                embeddings[value] = [0.0] * self.embedding_dim
                continue
            cached = self.cache.get_text_embedding(value)
            if cached:
                embeddings[value] = cached
            else:
                missing.append(value)

        logger.info(
            f"Tag embeddings: {len(embeddings)} cached, {len(missing)} to compute"
        )

        if missing:
            computed = await asyncio.gather(
                *[self.get_cached_embedding(value) for value in missing],
                return_exceptions=True,
            )
            for value, embedding in zip(missing, computed):
                if isinstance(embedding, Exception):
                    logger.error(f"Embedding failed for tag '{value}': {embedding}")
                    embeddings[value] = [0.0] * self.embedding_dim
                else:
                    embeddings[value] = embedding

        return embeddings

    def _person_vector(
        self, values: List[str], value_embeddings: Dict[str, List[float]]
    ) -> List[float]:
        """Sum a person's tag vectors and normalize to unit length."""
        vectors = [
            value_embeddings[value] for value in values if value in value_embeddings
        ]
        if not vectors:
            return [0.0] * self.embedding_dim

        summed = np.sum(np.array(vectors), axis=0)
        norm = np.linalg.norm(summed)
        if norm > 0:
            summed = summed / norm
        return summed.tolist()

    def _assemble(
        self,
        df: pd.DataFrame,
        person_embeddings: Dict[int, List[float]],
        feature_name: str,
    ) -> np.ndarray:
        rows = []
        for position in range(len(df)):
            embedding = person_embeddings.get(position)
            if embedding is None:
                logger.warning(
                    f"Feature {feature_name}: no embedding for person at position "
                    f"{position}, using a zero vector"
                )
                embedding = [0.0] * self.embedding_dim
            rows.append(embedding)
        return np.array(rows)
