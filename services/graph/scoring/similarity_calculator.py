import logging
from typing import Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from services.analysis.dataset_insights import DatasetInsightsAnalyzer
from shared.shared import FEATURES

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Handles all similarity calculations for features"""

    def __init__(self, insight_analyzer: DatasetInsightsAnalyzer = None):
        self._cached_matrices = {}
        self.insight_analyzer = insight_analyzer or DatasetInsightsAnalyzer()

    def precompute_similarity_matrices(
        self, feature_embeddings: Dict[str, np.ndarray]
    ) -> None:
        """Precompute similarity matrices once for all features"""
        logger.info("⚡ Precomputing similarity matrices...")

        feature_names = FEATURES + ["personas"]

        for feature_name in feature_names:
            if feature_name in feature_embeddings:
                raw_matrix = cosine_similarity(feature_embeddings[feature_name])
                self._cached_matrices[feature_name] = (
                    self.insight_analyzer.normalize_matrix(
                        raw_matrix, preserve_diagonal=True
                    )
                )

    def get_similarity_score(
        self, feature_name: str, person_i: int, person_j: int
    ) -> float:
        """Get similarity score between two people for a specific feature"""
        if feature_name in self._cached_matrices:
            return self._cached_matrices[feature_name][person_i][person_j]
        return 0.0

    def get_similarity_matrices(self) -> Dict[str, np.ndarray]:
        """Get all precomputed similarity matrices"""
        return {
            feature: matrix
            for feature, matrix in self._cached_matrices.items()
            if feature in FEATURES
        }

    def get_all_similarities(self, person_i: int, person_j: int) -> Dict[str, float]:
        """Get all similarity scores between two people"""
        return {
            "role": self.get_similarity_score("role", person_i, person_j),
            "experience": self.get_similarity_score("experience", person_i, person_j),
            "industry": self.get_similarity_score("industry", person_i, person_j),
            "market": self.get_similarity_score("market", person_i, person_j),
            "offering": self.get_similarity_score("offering", person_i, person_j),
            "persona": self.get_similarity_score("personas", person_i, person_j),
        }

    def clear_cache(self) -> None:
        """Clear all cached similarity matrices"""
        self._cached_matrices.clear()
