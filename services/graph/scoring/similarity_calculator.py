import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Handles all similarity calculations for features"""

    def __init__(self):
        self._cached_matrices = {}

    def precompute_similarity_matrices(
        self, feature_embeddings: Dict[str, np.ndarray]
    ) -> None:
        """Precompute similarity matrices once for all features"""
        logger.info("⚡ Precomputing similarity matrices...")

        feature_names = [
            "role_spec",
            "experience",
            "personas",
            "industry",
            "market",
            "offering",
        ]

        for feature_name in feature_names:
            if feature_name in feature_embeddings:
                self._cached_matrices[feature_name] = cosine_similarity(
                    feature_embeddings[feature_name]
                )

    def get_similarity_score(
        self, feature_name: str, person_i: int, person_j: int
    ) -> float:
        """Get similarity score between two people for a specific feature"""
        if feature_name in self._cached_matrices:
            return self._cached_matrices[feature_name][person_i][person_j]
        return 0.0

    def get_all_similarities(self, person_i: int, person_j: int) -> Dict[str, float]:
        """Get all similarity scores between two people"""
        return {
            "role": self.get_similarity_score("role_spec", person_i, person_j),
            "experience": self.get_similarity_score("experience", person_i, person_j),
            "industry": self.get_similarity_score("industry", person_i, person_j),
            "market": self.get_similarity_score("market", person_i, person_j),
            "offering": self.get_similarity_score("offering", person_i, person_j),
            "persona": self.get_similarity_score("personas", person_i, person_j),
        }

    def clear_cache(self) -> None:
        """Clear all cached similarity matrices"""
        self._cached_matrices.clear()
