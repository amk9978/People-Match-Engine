from typing import List, Protocol

import numpy as np


class EmbeddingServiceProtocol(Protocol):
    """Protocol for embedding services"""

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching, returns List[float] for compatibility"""
        ...

    async def get_embedding_array(self, text: str) -> np.ndarray:
        """Get embedding as numpy array"""
        ...

    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with true batching and validation"""
        ...
