import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

pytest_plugins = ('pytest_asyncio',)


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings to avoid dependency issues during testing"""
    with patch('settings.OPENAI_API_KEY', 'test-api-key'), \
         patch('settings.REDIS_URL', 'redis://localhost:6379/0'), \
         patch('settings.EMBEDDING_BATCH_DELAY', 1.0), \
         patch('settings.MIN_DENSITY', 0.1):
        yield


@pytest.fixture
def mock_redis_cache():
    """Provides a mock Redis cache for tests"""
    cache = MagicMock()
    cache.get.return_value = None
    cache.set.return_value = True
    cache.delete.return_value = 1
    return cache


@pytest.fixture
def sample_embedding():
    """Standard embedding for testing"""
    return [0.1, 0.2, 0.3, 0.4, 0.5] * 77 + [0.1] * 5  # 384 dimensions total