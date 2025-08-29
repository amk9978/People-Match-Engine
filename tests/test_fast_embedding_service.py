import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from services.preprocessing.fast_embedding_service import FastEmbeddingService


class TestFastEmbeddingService:

    @pytest.fixture
    def mock_cache(self):
        return MagicMock()

    @pytest.fixture
    def mock_text_embedding_model(self):
        return MagicMock()

    @pytest.fixture
    def fast_embedding_service(self, mock_cache, mock_text_embedding_model):
        with patch(
            "services.preprocessing.fast_embedding_service.RedisEmbeddingCache",
            return_value=mock_cache,
        ), patch(
            "services.preprocessing.fast_embedding_service.TextEmbedding",
            return_value=mock_text_embedding_model,
        ):
            service = FastEmbeddingService()
            service.cache = mock_cache
            service.model = mock_text_embedding_model
            return service

    @pytest.mark.asyncio
    async def test_get_embedding_cache_hit(self, fast_embedding_service, mock_cache):
        test_text = "test text"
        expected_embedding = [0.1, 0.2, 0.3] * 128
        mock_cache.get.return_value = expected_embedding

        result = await fast_embedding_service.get_embedding(test_text)

        assert result == expected_embedding
        mock_cache.get.assert_called_once_with(test_text)
        fast_embedding_service.model.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_embedding_cache_miss_success(
        self, fast_embedding_service, mock_cache, mock_text_embedding_model
    ):
        test_text = "test text"
        expected_embedding = [0.1, 0.2] * 192

        mock_cache.get.return_value = None
        mock_embedding_array = MagicMock()
        mock_embedding_array.tolist.return_value = expected_embedding
        mock_text_embedding_model.embed.return_value = [mock_embedding_array]

        result = await fast_embedding_service.get_embedding(test_text)

        assert result == expected_embedding
        mock_cache.get.assert_called_once_with(test_text)
        mock_cache.set.assert_called_once_with(test_text, expected_embedding)
        mock_text_embedding_model.embed.assert_called_once_with([test_text])

    @pytest.mark.asyncio
    async def test_get_embedding_model_error_fallback(
        self, fast_embedding_service, mock_cache, mock_text_embedding_model
    ):
        test_text = "test text"
        mock_cache.get.return_value = None
        mock_text_embedding_model.embed.side_effect = Exception("Model Error")

        result = await fast_embedding_service.get_embedding(test_text)

        assert result == [0.0] * 384
        mock_cache.get.assert_called_once_with(test_text)
        mock_cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_embedding_array(self, fast_embedding_service, mock_cache):
        test_text = "test text"
        expected_embedding = [0.1, 0.2, 0.3] * 128
        mock_cache.get.return_value = expected_embedding

        result = await fast_embedding_service.get_embedding_array(test_text)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(expected_embedding))

    @pytest.mark.asyncio
    async def test_get_batch_embeddings_empty_input(self, fast_embedding_service):
        result = await fast_embedding_service.get_batch_embeddings([])
        assert result == []

    @pytest.mark.asyncio
    async def test_get_batch_embeddings_all_cached(
        self, fast_embedding_service, mock_cache
    ):
        texts = ["text1", "text2", "text3"]
        cached_embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

        mock_cache.get.side_effect = cached_embeddings

        result = await fast_embedding_service.get_batch_embeddings(texts)

        assert result == cached_embeddings
        assert mock_cache.get.call_count == 3
        fast_embedding_service.model.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_batch_embeddings_mixed_cache(
        self, fast_embedding_service, mock_cache, mock_text_embedding_model
    ):
        texts = ["cached_text", "uncached_text1", "uncached_text2"]
        cached_embedding = [0.1] * 384
        uncached_embeddings = [[0.2] * 384, [0.3] * 384]

        def cache_side_effect(text):
            return cached_embedding if text == "cached_text" else None

        mock_cache.get.side_effect = cache_side_effect

        mock_embedding_arrays = []
        for emb in uncached_embeddings:
            mock_array = MagicMock()
            mock_array.tolist.return_value = emb
            mock_embedding_arrays.append(mock_array)

        mock_text_embedding_model.embed.return_value = mock_embedding_arrays

        result = await fast_embedding_service.get_batch_embeddings(texts)

        expected_result = [
            cached_embedding,
            uncached_embeddings[0],
            uncached_embeddings[1],
        ]
        assert result == expected_result
        assert mock_cache.get.call_count == 3
        mock_text_embedding_model.embed.assert_called_once_with(
            ["uncached_text1", "uncached_text2"]
        )
        assert mock_cache.set.call_count == 2

    @pytest.mark.asyncio
    async def test_get_batch_embeddings_validation_error(
        self, fast_embedding_service, mock_cache, mock_text_embedding_model
    ):
        texts = ["text1", "text2"]
        mock_cache.get.return_value = None

        # First call: batch request with wrong dimensions triggers validation error
        mock_embedding_arrays = []
        for _ in range(2):
            mock_array = MagicMock()
            mock_array.tolist.return_value = [0.1] * 100  # Wrong dimension
            mock_embedding_arrays.append(mock_array)

        # Individual fallback calls - both succeed with correct dimensions
        correct_embedding = [0.5] * 384
        mock_array_fallback = MagicMock()
        mock_array_fallback.tolist.return_value = correct_embedding

        mock_text_embedding_model.embed.side_effect = [
            mock_embedding_arrays,  # Batch fails validation
            [mock_array_fallback],  # text1 individual success
            [mock_array_fallback],  # text2 individual success
        ]

        result = await fast_embedding_service.get_batch_embeddings(texts)

        # Fallback to individual requests should succeed with correct dimensions
        assert len(result) == 2
        assert all(len(emb) == 384 for emb in result)
        assert all(emb == correct_embedding for emb in result)

    @pytest.mark.asyncio
    async def test_get_batch_embeddings_batch_error_fallback(
        self, fast_embedding_service, mock_cache, mock_text_embedding_model
    ):
        texts = ["text1", "text2"]
        mock_cache.get.return_value = None

        fallback_embedding = [0.5] * 384
        fallback_array = MagicMock()
        fallback_array.tolist.return_value = fallback_embedding

        mock_text_embedding_model.embed.side_effect = [
            Exception("Batch failed"),
            [fallback_array],
            [fallback_array],
        ]

        result = await fast_embedding_service.get_batch_embeddings(texts)

        assert result == [fallback_embedding, fallback_embedding]
        assert mock_text_embedding_model.embed.call_count == 3

    def test_service_initialization(self, mock_cache, mock_text_embedding_model):
        with patch(
            "services.preprocessing.fast_embedding_service.RedisEmbeddingCache",
            return_value=mock_cache,
        ), patch(
            "services.preprocessing.fast_embedding_service.TextEmbedding",
            return_value=mock_text_embedding_model,
        ), patch(
            "services.preprocessing.fast_embedding_service.settings.EMBEDDING_BATCH_DELAY",
            2.5,
        ):

            service = FastEmbeddingService(model_name="test-model")

            assert service.cache == mock_cache
            assert service.model == mock_text_embedding_model
            assert service.batch_delay == 2.5
            assert service.embedding_dim == 384
