import logging
from typing import Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Holds one person-by-person cosine similarity matrix per feature.

    Matrices start raw so informativeness can be measured from their spread, then
    are replaced by calibrated ones before any edge is scored.
    """

    def __init__(self):
        self._matrices: Dict[str, np.ndarray] = {}

    def precompute(self, feature_embeddings: Dict[str, np.ndarray]) -> None:
        self._matrices = {
            name: cosine_similarity(embeddings)
            for name, embeddings in feature_embeddings.items()
        }
        logger.info(f"Computed similarity for {len(self._matrices)} features")

    def raw_matrices(self) -> Dict[str, np.ndarray]:
        return dict(self._matrices)

    def apply_calibrated(self, matrices: Dict[str, np.ndarray]) -> None:
        assert set(matrices) == set(
            self._matrices
        ), "calibrated matrices must cover exactly the computed features"
        self._matrices = dict(matrices)

    def get_similarity_score(
        self, feature_name: str, person_i: int, person_j: int
    ) -> float:
        matrix = self._matrices.get(feature_name)
        if matrix is None:
            return 0.0
        return float(matrix[person_i][person_j])

    def get_all_similarities(self, person_i: int, person_j: int) -> Dict[str, float]:
        return {
            name: float(matrix[person_i][person_j])
            for name, matrix in self._matrices.items()
        }

    def clear_cache(self) -> None:
        self._matrices.clear()
