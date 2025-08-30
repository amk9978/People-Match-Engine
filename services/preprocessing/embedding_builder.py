import asyncio
import logging
import sys
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd

from services.preprocessing.embedding_interface import EmbeddingServiceProtocol
from services.preprocessing.fast_embedding_service import FastEmbeddingService
from services.preprocessing.semantic_person_deduplicator import (
    SemanticPersonDeduplicator,
)
from services.preprocessing.tag_extractor import tag_extractor
from services.redis.app_cache_service import app_cache_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class EmbeddingBuilder:
    """Handles feature embedding generation with caching and deduplication"""

    def __init__(
        self,
        cache=None,
        person_deduplicator: SemanticPersonDeduplicator = None,
        embedding_service: EmbeddingServiceProtocol = None,
    ):
        self.cache = cache or app_cache_service
        self.person_deduplicator = person_deduplicator or SemanticPersonDeduplicator()
        self.embedding_service = embedding_service or FastEmbeddingService()

    async def extract_and_deduplicate_tags(self, text: str, category: str) -> List[str]:
        """Extract tags and apply semantic deduplication for any feature category"""
        raw_tags = tag_extractor.extract_tags(text, category)
        return raw_tags

        return await self.person_deduplicator.apply_semantic_deduplication(
            raw_tags, category
        )

    async def get_cached_embedding(self, tag: str) -> List[float]:
        """Get embedding using shared embedding service"""
        return await self.embedding_service.get_embedding(tag)

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
        """Create OpenAI embeddings for multiple feature categories with row-based caching"""
        logger.info("Creating feature embeddings with row-based caching...")

        feature_embeddings = {}

        for feature_name, column_name in feature_columns.items():
            logger.info(
                f"\nProcessing {feature_name} ({column_name}) with row-based caching..."
            )

            try:
                cache_status = self.cache.get_dataset_embedding_cache_status(
                    df, feature_name
                )

                if cache_status is None:
                    logger.error(f"❌ Cache status returned None for {feature_name}")
                    logger.error(f"❌ Cache object type: {type(self.cache)}")
                    logger.error(f"❌ Feature name: {feature_name}")
                    logger.error(f"❌ DataFrame shape: {df.shape}")
                    logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                    # Use fallback
                    cached_embeddings = {}
                    uncached_indices = list(range(len(df)))
                else:
                    cached_embeddings = cache_status.get("cached_embeddings", {})
                    uncached_indices = cache_status.get(
                        "uncached_indices", list(range(len(df)))
                    )

                logger.info(
                    f"Found {len(cached_embeddings)} cached embeddings, {len(uncached_indices)} need computing"
                )

                person_feature_embeddings = {}
                for idx, embedding in cached_embeddings.items():
                    person_feature_embeddings[idx] = embedding

                if uncached_indices:
                    logger.info(
                        f"Computing embeddings for {len(uncached_indices)} uncached people. {feature_name}"
                    )
                    valid_uncached_indices = [
                        idx for idx in uncached_indices if idx < len(df)
                    ]
                    all_unique_values = set()
                    for idx in valid_uncached_indices:
                        row = df.iloc[idx]
                        values = await self.extract_and_deduplicate_tags(
                            row[column_name], feature_name
                        )
                        all_unique_values.update(values)

                    logger.info(
                        f"Found {len(all_unique_values)} unique values from uncached people"
                    )

                    cached_values = {}
                    uncached_values = []
                    cache_hits = 0

                    for value in all_unique_values:
                        if value.strip():
                            cached_embedding = self.cache.get_text_embedding(value)
                            if cached_embedding:
                                cached_values[value] = cached_embedding
                                cache_hits += 1
                            else:
                                uncached_values.append(value)
                        else:
                            cached_values[value] = [0.0] * 384

                    logger.info(
                        f"Text cache hits: {cache_hits}, requests needed: {len(uncached_values)}"
                    )

                    value_embeddings = cached_values.copy()

                    if uncached_values:
                        tasks = []
                        for value in uncached_values:
                            tasks.append(self.get_cached_embedding(value))

                        embeddings_results = await asyncio.gather(
                            *tasks, return_exceptions=True
                        )

                        for value, embedding in zip(
                            uncached_values, embeddings_results
                        ):
                            if isinstance(embedding, Exception):
                                logger.info(f"Error processing {value}: {embedding}")
                                value_embeddings[value] = [0.0] * 384
                            else:
                                value_embeddings[value] = embedding

                    new_person_embeddings = {}
                    for idx in valid_uncached_indices:
                        row = df.iloc[idx]
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
                                person_value_embeddings = np.array(
                                    person_value_embeddings
                                )
                                person_embedding = np.sum(
                                    person_value_embeddings, axis=0
                                )
                                norm = np.linalg.norm(person_embedding)
                                if norm > 0:
                                    person_embedding = person_embedding / norm
                            else:
                                person_embedding = [0.0] * 384
                        else:
                            person_embedding = [0.0] * 384

                        person_feature_embeddings[idx] = person_embedding.tolist()
                        new_person_embeddings[idx] = person_embedding.tolist()

                    self.cache.cache_dataset_embeddings(
                        df, feature_name, new_person_embeddings
                    )
                    logger.info(
                        f"Cached {len(new_person_embeddings)} new person embeddings"
                    )

                # Convert dictionary back to array aligned with DataFrame
                embeddings_array = []
                for i in df.index:
                    if i in person_feature_embeddings:
                        embeddings_array.append(person_feature_embeddings[i])
                    else:
                        embeddings_array.append([0.0] * 384)
                        logger.warning(
                            f"Warning: No embedding for person at index {i}, using fallback"
                        )

                    feature_embeddings[feature_name] = np.array(embeddings_array)
                    logger.info(
                        f"Created {feature_embeddings[feature_name].shape[1]}D embeddings for {len(person_feature_embeddings)} people"
                    )

            except Exception as e:
                logger.error(f"❌ Analysis failed for {feature_name}: {e}")
                logger.error(f"❌ Exception type: {type(e).__name__}")
                logger.error(f"❌ Exception details: {str(e)}")
                logger.error(f"❌ Full traceback: {traceback.format_exc()}")

                # Create fallback embeddings (zero vectors)
                fallback_embeddings = np.zeros((len(df), 384))
                feature_embeddings[feature_name] = fallback_embeddings
                logger.info(f"⚠️  Created fallback zero embeddings for {feature_name}")

        cache_info = self.cache.get_cache_stats()
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
