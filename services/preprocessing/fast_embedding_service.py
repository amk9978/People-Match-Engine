import asyncio
import logging
from typing import List

import numpy as np
import sentry_sdk
from fastembed import TextEmbedding

import settings
from services.preprocessing.embedding_interface import EmbeddingServiceProtocol
from services.redis.redis_cache import RedisEmbeddingCache

logger = logging.getLogger(__name__)


class FastEmbeddingService(EmbeddingServiceProtocol):
    """Shared service for FastEmbed embedding retrieval with Redis caching"""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)
        self.cache = RedisEmbeddingCache()
        self.batch_delay = settings.EMBEDDING_BATCH_DELAY
        self.embedding_dim = 384

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching, returns List[float] for compatibility"""
        cached_embedding = self.cache.get(text)
        if cached_embedding:
            return cached_embedding

        try:
            with sentry_sdk.start_transaction(
                op="ai.embed", name="fastembed_embedding", sampled=False
            ):
                embeddings = list(self.model.embed([text]))
                embedding = embeddings[0].tolist()

            self.cache.set(text, embedding)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for '{text}': {e}")
            return [0.0] * self.embedding_dim

    async def get_embedding_array(self, text: str) -> np.ndarray:
        """Get embedding as numpy array"""
        embedding = await self.get_embedding(text)
        return np.array(embedding)

    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with true batching and validation"""
        if not texts:
            return []

        # Separate cached vs uncached texts
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)

        # Check cache first and collect uncached texts
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text)
            if cached_embedding:
                results[i] = cached_embedding
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if not uncached_texts:
            return results

        batch_size = 2048

        for batch_start in range(0, len(uncached_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]
            batch_indices = uncached_indices[batch_start:batch_end]

            try:
                logger.info(
                    f"Requesting embeddings for batch of {len(batch_texts)} texts..."
                )

                with sentry_sdk.start_transaction(
                    op="ai.embed", name="fastembed_batch_embedding", sampled=False
                ):
                    embeddings = list(self.model.embed(batch_texts))

                if len(embeddings) != len(batch_texts):
                    raise ValueError(
                        f"Expected {len(batch_texts)} embeddings, got {len(embeddings)}"
                    )

                # Process each embedding in the response
                for i, embedding_array in enumerate(embeddings):
                    original_index = batch_indices[i]
                    original_text = batch_texts[i]
                    embedding = embedding_array.tolist()

                    # VALIDATION: Check embedding dimensions
                    if len(embedding) != self.embedding_dim:
                        raise ValueError(
                            f"Expected {self.embedding_dim} dimensions, got {len(embedding)} for text: '{original_text}'"
                        )

                    # Store in results and cache
                    results[original_index] = embedding
                    self.cache.set(original_text, embedding)

                logger.info(
                    f"Successfully processed batch of {len(batch_texts)} embeddings"
                )

                if batch_end < len(uncached_texts):
                    await asyncio.sleep(self.batch_delay)

            except Exception as e:
                logger.info(f"Error in batch embedding request: {e}")
                logger.info("Falling back to individual requests for this batch...")

                # FALLBACK: Individual requests for failed batch
                for i, text in enumerate(batch_texts):
                    original_index = batch_indices[i]
                    try:
                        with sentry_sdk.start_transaction(
                            op="ai.embed",
                            name="fastembed_fallback_embedding",
                            sampled=False,
                        ):
                            embeddings = list(self.model.embed([text]))
                            embedding = embeddings[0].tolist()
                        results[original_index] = embedding
                        self.cache.set(text, embedding)

                    except Exception as individual_error:
                        logger.error(
                            f"Error getting embedding for '{text}': {individual_error}"
                        )
                        results[original_index] = [0.0] * self.embedding_dim

        # FINAL VALIDATION: Ensure no None values remain
        for i, result in enumerate(results):
            if result is None:
                logger.info(
                    f"Warning: No embedding obtained for text at index {i}: '{texts[i]}'. Using fallback."
                )
                results[i] = [0.0] * self.embedding_dim

        logger.info(
            f"Batch embedding complete: {len(results)} total embeddings returned"
        )
        return results
