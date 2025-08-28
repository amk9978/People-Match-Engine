import asyncio
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from services.preprocessing.embedding_service import embedding_service
from services.preprocessing.semantic_person_deduplicator import (
    SemanticPersonDeduplicator,
)
from services.preprocessing.tag_extractor import tag_extractor
from services.redis.redis_cache import RedisEmbeddingCache

logger = logging.getLogger(__name__)


class EmbeddingBuilder:
    """Handles feature embedding generation with caching and deduplication"""

    def __init__(self):
        self.cache = RedisEmbeddingCache()
        self.person_deduplicator = SemanticPersonDeduplicator()

    async def extract_and_deduplicate_tags(self, text: str, category: str) -> List[str]:
        """Extract tags and apply semantic deduplication for any feature category"""
        raw_tags = tag_extractor.extract_tags(text, category)
        return raw_tags

        return await self.person_deduplicator.apply_semantic_deduplication(
            raw_tags, category
        )

    async def get_cached_embedding(self, tag: str) -> List[float]:
        """Get embedding using shared embedding service"""
        return await embedding_service.get_embedding(tag)

    async def extract_business_tags_for_person(self, row) -> Dict[str, List[str]]:
        """Extract deduplicated business tags for a person for causal analysis"""
        business_tags = {}

        industry_text = row["Company Identity - Industry Classification"]
        business_tags["industry"] = await self.extract_and_deduplicate_tags(
            industry_text, "industry"
        )

        market_text = row["Company Market - Market Traction"]
        business_tags["market"] = await self.extract_and_deduplicate_tags(
            market_text, "market"
        )

        offering_text = row["Company Offering - Value Proposition"]
        business_tags["offering"] = await self.extract_and_deduplicate_tags(
            offering_text, "offering"
        )

        return business_tags

    async def embed_features(
        self, df: pd.DataFrame, feature_columns: Dict[str, str]
    ) -> Dict[str, np.ndarray]:
        """Create OpenAI embeddings for multiple feature categories with async batch processing"""
        logger.info("Creating feature embeddings...")

        feature_embeddings = {}

        for feature_name, column_name in feature_columns.items():
            logger.info(f"\nProcessing {feature_name} ({column_name})...")

            all_unique_values = set()
            for _, row in df.iterrows():

                values = await self.extract_and_deduplicate_tags(
                    row[column_name], feature_name
                )
                all_unique_values.update(values)

            logger.info(
                f"Found {len(all_unique_values)} unique values for {feature_name}"
            )

            cached_values = {}
            uncached_values = []
            cache_hits = 0

            for value in all_unique_values:
                if value.strip():
                    if self.cache.exists(value):
                        cached_values[value] = self.cache.get(value)
                        cache_hits += 1
                    else:
                        uncached_values.append(value)
                else:
                    cached_values[value] = [0.0] * 1536

            logger.info(
                f"  Cache hits: {cache_hits}, API calls needed: {len(uncached_values)}"
            )

            value_embeddings = cached_values.copy()

            if uncached_values:
                logger.info(
                    f"  Processing {len(uncached_values)} embeddings in async batches..."
                )

                tasks = []
                for value in uncached_values:
                    tasks.append(self.get_cached_embedding(value))

                embeddings_results = await asyncio.gather(
                    *tasks, return_exceptions=True
                )

                for value, embedding in zip(uncached_values, embeddings_results):
                    if isinstance(embedding, Exception):
                        logger.info(f"  Error processing {value}: {embedding}")
                        value_embeddings[value] = [0.0] * 1536
                    else:
                        value_embeddings[value] = embedding

            logger.info(
                f"  {feature_name} summary: {cache_hits} from cache, {len(uncached_values)} async API calls"
            )

            person_feature_embeddings = []
            for idx, row in df.iterrows():

                values = await self.extract_and_deduplicate_tags(
                    row[column_name], feature_name
                )

                if values:

                    person_value_embeddings = [
                        value_embeddings[val]
                        for val in values
                        if val in value_embeddings
                    ]

                    if person_value_embeddings:

                        person_value_embeddings = np.array(person_value_embeddings)
                        person_embedding = np.sum(person_value_embeddings, axis=0)
                        norm = np.linalg.norm(person_embedding)
                        if norm > 0:
                            person_embedding = person_embedding / norm
                    else:
                        person_embedding = [0.0] * 1536
                else:
                    person_embedding = [0.0] * 1536

                person_feature_embeddings.append(person_embedding)

            feature_embeddings[feature_name] = np.array(person_feature_embeddings)
            logger.info(
                f"Created {feature_embeddings[feature_name].shape[1]}D embeddings for {len(person_feature_embeddings)} people"
            )

        cache_info = self.cache.get_cache_info()
        logger.info(f"Redis cache status: {cache_info}")

        return feature_embeddings

    async def preprocess_tags(
        self,
        csv_path: str,
        similarity_threshold: float = 0.7,
        fuzzy_threshold: float = 0.90,
        force_rebuild: bool = False,
    ) -> Dict[str, any]:
        """Run tag deduplication preprocessing"""
        logger.info("Running tag deduplication preprocessing...")

        existing_stats = self.person_deduplicator.get_stats()
        if not force_rebuild and "error" not in existing_stats:
            logger.info(f"✓ Found existing person-level deduplication results")
            return existing_stats

        logger.info("Building semantic person-level tag deduplication mappings...")
        dedup_results = await self.person_deduplicator.process_dataset_semantic(
            csv_path, similarity_threshold
        )

        logger.info(f"✓ Person-level tag deduplication complete")

        return dedup_results

    def extract_tags(self, persona_titles: str) -> List[str]:
        """Extract and clean tags from the persona titles column"""
        return tag_extractor.extract_persona_tags(persona_titles)
