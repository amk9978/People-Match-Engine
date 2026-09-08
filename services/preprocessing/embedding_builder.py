import asyncio
import logging
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from services.cache.app_cache_service import app_cache_service
from services.preprocessing.embedding_interface import EmbeddingServiceProtocol
from services.preprocessing.fast_embedding_service import FastEmbeddingService
from services.preprocessing.tag_extractor import tag_extractor

logger = logging.getLogger(__name__)

BUSINESS_TAG_COLUMNS = {
    "industry": "Company Identity - Industry Classification",
    "market": "Company Market - Market Traction",
    "offering": "Company Offering - Value Proposition",
}


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

    def extract_business_tags_for_person(self, row: pd.Series) -> Dict[str, List[str]]:
        """Extract business tags for a person for causal analysis"""
        return {
            feature: tag_extractor.extract_tags(row[column], feature)
            for feature, column in BUSINESS_TAG_COLUMNS.items()
        }

    async def embed_features(
        self, df: pd.DataFrame, feature_columns: Dict[str, str]
    ) -> Dict[str, np.ndarray]:
        assert df.index.equals(
            pd.RangeIndex(len(df))
        ), "people are addressed by position; the frame must be indexed 0..n-1"

        feature_embeddings = {}
        for feature_name, column_name in feature_columns.items():
            feature_embeddings[feature_name] = await self._embed_one_feature(
                df, feature_name, column_name
            )

        logger.info(f"Cache after embedding: {self.cache.get_cache_stats()}")
        return feature_embeddings

    async def _embed_one_feature(
        self, df: pd.DataFrame, feature_name: str, column_name: str
    ) -> np.ndarray:
        cache_status = self.cache.get_dataset_embedding_cache_status(df, feature_name)
        person_embeddings = dict(cache_status["cached_embeddings"])
        uncached_positions = cache_status["uncached_indices"]

        if uncached_positions:
            values = self._collect_values(
                df, uncached_positions, column_name, feature_name
            )
            value_embeddings = await self._embed_values(values)

            computed = {
                position: self._person_vector(
                    tag_extractor.extract_tags(
                        df.iloc[position][column_name], feature_name
                    ),
                    value_embeddings,
                )
                for position in uncached_positions
            }
            person_embeddings.update(computed)
            self.cache.cache_dataset_embeddings(df, feature_name, computed)

        matrix = self._assemble(df, person_embeddings, feature_name)
        logger.info(
            f"Feature {feature_name}: {matrix.shape[0]} people, {matrix.shape[1]} dimensions"
        )
        return matrix

    def _collect_values(
        self,
        df: pd.DataFrame,
        positions: List[int],
        column_name: str,
        feature_name: str,
    ) -> Set[str]:
        values = set()
        for position in positions:
            values.update(
                tag_extractor.extract_tags(df.iloc[position][column_name], feature_name)
            )
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
