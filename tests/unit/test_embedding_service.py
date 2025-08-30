import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from services.preprocessing.embedding_service import EmbeddingService


class TestEmbeddingService:

    @pytest.fixture
    def mock_cache(self):
        return MagicMock()

    @pytest.fixture
    def mock_openai_client(self):
        return AsyncMock()

    @pytest.fixture
    def embedding_service(self, mock_cache, mock_openai_client):
        with patch(
            "services.preprocessing.embedding_service.RedisEmbeddingCache",
            return_value=mock_cache,
        ), patch(
            "services.preprocessing.embedding_service.AsyncOpenAI",
            return_value=mock_openai_client,
        ):
            service = EmbeddingService()
            service.cache = mock_cache
            service.openai_client = mock_openai_client
            return service

    @pytest.mark.asyncio
    async def test_get_embedding_cache_hit(self, embedding_service, mock_cache):
        test_text = "test text"
        expected_embedding = [0.1, 0.2, 0.3]
        mock_cache.get.return_value = expected_embedding

        result = await embedding_service.get_embedding(test_text)

        assert result == expected_embedding
        mock_cache.get.assert_called_once_with(test_text)
        embedding_service.openai_client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_embedding_cache_miss_success(
        self, embedding_service, mock_cache, mock_openai_client
    ):
        test_text = "test text"
        expected_embedding = [0.1, 0.2, 0.3]

        mock_cache.get.return_value = None

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=expected_embedding)]
        mock_openai_client.embeddings.create.return_value = mock_response

        result = await embedding_service.get_embedding(test_text)

        assert result == expected_embedding
        mock_cache.get.assert_called_once_with(test_text)
        mock_cache.set.assert_called_once_with(test_text, expected_embedding)
        mock_openai_client.embeddings.create.assert_called_once_with(
            input=test_text, model="text-embedding-3-small"
        )

    @pytest.mark.asyncio
    async def test_get_embedding_api_error_fallback(
        self, embedding_service, mock_cache, mock_openai_client
    ):
        test_text = "test text"
        mock_cache.get.return_value = None
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")

        result = await embedding_service.get_embedding(test_text)

        assert result == [0.0] * 1536
        mock_cache.get.assert_called_once_with(test_text)
        mock_cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_embedding_array(self, embedding_service, mock_cache):
        test_text = "test text"
        expected_embedding = [0.1, 0.2, 0.3]
        mock_cache.get.return_value = expected_embedding

        result = await embedding_service.get_embedding_array(test_text)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(expected_embedding))

    @pytest.mark.asyncio
    async def test_get_embedding_empty_text(
        self, embedding_service, mock_cache, mock_openai_client
    ):
        test_text = ""
        expected_embedding = [0.0, 0.0, 0.0]

        mock_cache.get.return_value = None
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=expected_embedding)]
        mock_openai_client.embeddings.create.return_value = mock_response

        result = await embedding_service.get_embedding(test_text)

        assert result == expected_embedding
        mock_openai_client.embeddings.create.assert_called_once_with(
            input="", model="text-embedding-3-small"
        )

    def test_service_initialization(self, mock_cache, mock_openai_client):
        with patch(
            "services.preprocessing.embedding_service.RedisEmbeddingCache",
            return_value=mock_cache,
        ), patch(
            "services.preprocessing.embedding_service.AsyncOpenAI",
            return_value=mock_openai_client,
        ), patch(
            "services.preprocessing.embedding_service.settings.OPENAI_API_KEY",
            "test-key",
        ), patch(
            "services.preprocessing.embedding_service.settings.EMBEDDING_BATCH_DELAY",
            1.5,
        ):

            service = EmbeddingService()

            assert service.cache == mock_cache
            assert service.openai_client == mock_openai_client
            assert service.batch_delay == 1.5
