import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["SENTRY_DSN"] = ""
os.environ["REDIS_URL"] = ""

from match_engine.services.cache.factory import set_cache_backend
from match_engine.services.cache.memory import InMemoryBackend

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(autouse=True)
def in_memory_cache():
    """Give every test an empty process-wide cache that never opens a socket."""
    set_cache_backend(InMemoryBackend())
    yield
    set_cache_backend(InMemoryBackend())


@pytest.fixture(autouse=True)
def mock_settings():
    with (
        patch("match_engine.settings.OPENAI_API_KEY", "test-api-key"),
        patch("match_engine.settings.EMBEDDING_BATCH_DELAY", 1.0),
        patch("match_engine.settings.MIN_DENSITY", 0.1),
    ):
        yield


@pytest.fixture
def mock_cache():
    cache = MagicMock()
    cache.get.return_value = None
    cache.set.return_value = True
    cache.delete.return_value = 1
    return cache


@pytest.fixture
def sample_embedding():
    return [0.1, 0.2, 0.3, 0.4, 0.5] * 77 + [0.1] * 5
