import logging
import os
from typing import Dict, Set, Tuple

import numpy as np

from services.faiss_similarity_engine import FAISSGraphMatcher, FAISSSimilarityEngine

logger = logging.getLogger(__name__)


class FaissOptimizer:
    """Handles FAISS-based optimization for similarity search"""

    def __init__(self):
        self.faiss_engine = None
        self.faiss_matcher = None
        self.top_k = int(os.getenv("FAISS_TOP_K", "50"))

    def initialize_faiss(self, feature_embeddings: Dict[str, np.ndarray]) -> None:
        """Initialize FAISS similarity engine with feature embeddings"""
        logger.info(
            f"🚀 Initializing FAISS optimization (top-{self.top_k} per person)..."
        )

        self.faiss_engine = FAISSSimilarityEngine(embedding_dim=1536)
        self.faiss_engine.build_indices(feature_embeddings)
        self.faiss_matcher = FAISSGraphMatcher(self.faiss_engine)

    def get_candidate_pairs(self, min_similarity: float = 0.05) -> Set[Tuple[int, int]]:
        """Get all candidate pairs from FAISS similarity search"""
        if not self.faiss_matcher:
            raise ValueError("FAISS not initialized. Call initialize_faiss() first.")

        # Get similarity pairs for each feature
        feature_similarities = {}

        feature_names = [
            "role_spec",
            "experience",
            "personas",
            "industry",
            "market",
            "offering",
        ]

        for feature_name in feature_names:
            try:
                similarities = self.faiss_matcher.get_similarity_pairs(
                    feature_name, top_k=self.top_k, min_similarity=min_similarity
                )
                feature_similarities[feature_name] = similarities
                logger.info(f"   ✓ {feature_name}: {len(similarities)} candidate pairs")
            except Exception as e:
                logger.info(f"   ⚠️ {feature_name}: skipped ({e})")
                feature_similarities[feature_name] = {}

        # Combine all candidate pairs
        all_candidate_pairs = set()
        for similarities in feature_similarities.values():
            all_candidate_pairs.update(similarities.keys())

        logger.info(f"⚡ FAISS found {len(all_candidate_pairs)} total candidate pairs")

        return all_candidate_pairs, feature_similarities

    def get_similarity_scores(
            self, person_i: int, person_j: int, feature_similarities: Dict
    ) -> Dict[str, float]:
        """Get similarity scores for a specific pair from FAISS results"""
        pair = (person_i, person_j) if person_i < person_j else (person_j, person_i)

        return {
            "role": feature_similarities.get("role_spec", {}).get(pair, 0.0),
            "experience": feature_similarities.get("experience", {}).get(pair, 0.0),
            "persona": feature_similarities.get("personas", {}).get(pair, 0.0),
            "industry": feature_similarities.get("industry", {}).get(pair, 0.0),
            "market": feature_similarities.get("market", {}).get(pair, 0.0),
            "offering": feature_similarities.get("offering", {}).get(pair, 0.0),
        }

    def is_enabled(self) -> bool:
        """Check if FAISS optimization is enabled"""
        return os.getenv("USE_FAISS_OPTIMIZATION", "true").lower() == "true"
