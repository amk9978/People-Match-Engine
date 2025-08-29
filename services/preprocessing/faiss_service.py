import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss
from services.preprocessing.embedding_interface import EmbeddingServiceProtocol

logger = logging.getLogger(__name__)


class FAISSService:
    """FAISS-powered similarity search service for efficient nearest neighbor queries"""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Dict[int, Any] = {}
        self._is_trained = False

    def build_index(
        self, 
        embeddings: List[List[float]], 
        metadata: Optional[List[Any]] = None,
        index_type: str = "flat"
    ) -> None:
        """Build FAISS index from embeddings"""
        if not embeddings:
            raise ValueError("Cannot build index with empty embeddings")
        
        self.embeddings = np.array(embeddings, dtype=np.float32)
        
        if self.embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {self.embeddings.shape[1]}")
        
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            n_lists = min(100, max(1, len(embeddings) // 50))
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_lists)
            self.index.train(self.embeddings)
            self._is_trained = True
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.index.add(self.embeddings)
        
        if metadata:
            self.metadata = {i: meta for i, meta in enumerate(metadata)}
        
        logger.info(f"Built FAISS {index_type} index with {len(embeddings)} vectors")

    def search(
        self, 
        query_embedding: List[float], 
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[int, float, Any]]:
        """Search for k most similar embeddings"""
        if not self.index:
            raise ValueError("Index not built. Call build_index() first")
        
        query_vector = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if threshold is None or score >= threshold:
                metadata = self.metadata.get(idx)
                results.append((int(idx), float(score), metadata))
        
        return results

    def batch_search(
        self, 
        query_embeddings: List[List[float]], 
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[List[Tuple[int, float, Any]]]:
        """Batch search for multiple query embeddings"""
        if not self.index:
            raise ValueError("Index not built. Call build_index() first")
        
        query_vectors = np.array(query_embeddings, dtype=np.float32)
        scores, indices = self.index.search(query_vectors, k)
        
        results = []
        for i in range(len(query_embeddings)):
            query_results = []
            for idx, score in zip(indices[i], scores[i]):
                if threshold is None or score >= threshold:
                    metadata = self.metadata.get(idx)
                    query_results.append((int(idx), float(score), metadata))
            results.append(query_results)
        
        return results

    def get_similarity_matrix(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """Get similarity matrix for specified indices or all embeddings"""
        if not self.index or self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first")
        
        if indices:
            subset_embeddings = self.embeddings[indices]
        else:
            subset_embeddings = self.embeddings
        
        return np.dot(subset_embeddings, subset_embeddings.T)

    def add_embeddings(
        self, 
        new_embeddings: List[List[float]], 
        new_metadata: Optional[List[Any]] = None
    ) -> None:
        """Add new embeddings to existing index"""
        if not self.index:
            raise ValueError("Index not built. Call build_index() first")
        
        new_vectors = np.array(new_embeddings, dtype=np.float32)
        current_size = self.index.ntotal
        
        self.index.add(new_vectors)
        
        if new_metadata:
            for i, meta in enumerate(new_metadata):
                self.metadata[current_size + i] = meta
        
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, new_vectors])
        
        logger.info(f"Added {len(new_embeddings)} embeddings to index")

    def save_index(self, filepath: str) -> None:
        """Save FAISS index to disk"""
        if not self.index:
            raise ValueError("No index to save")
        
        faiss.write_index(self.index, filepath)
        logger.info(f"Saved FAISS index to {filepath}")

    def load_index(self, filepath: str) -> None:
        """Load FAISS index from disk"""
        self.index = faiss.read_index(filepath)
        self.embedding_dim = self.index.d
        logger.info(f"Loaded FAISS index from {filepath}")

    @property
    def size(self) -> int:
        """Get number of vectors in index"""
        return self.index.ntotal if self.index else 0

    def clear(self) -> None:
        """Clear the index and reset state"""
        self.index = None
        self.embeddings = None
        self.metadata.clear()
        self._is_trained = False


class FAISSEmbeddingAdapter:
    """Adapter to integrate FAISS service with embedding services"""
    
    def __init__(self, embedding_service: EmbeddingServiceProtocol, embedding_dim: int = 384):
        self.embedding_service = embedding_service
        self.faiss_service = FAISSService(embedding_dim)
    
    async def build_index_from_texts(
        self, 
        texts: List[str], 
        metadata: Optional[List[Any]] = None,
        index_type: str = "flat"
    ) -> None:
        """Build FAISS index from text using embedding service"""
        embeddings = await self.embedding_service.get_batch_embeddings(texts)
        self.faiss_service.build_index(embeddings, metadata, index_type)
    
    async def search_similar_texts(
        self, 
        query_text: str, 
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[int, float, Any]]:
        """Search for similar texts using query text"""
        query_embedding = await self.embedding_service.get_embedding(query_text)
        return self.faiss_service.search(query_embedding, k, threshold)