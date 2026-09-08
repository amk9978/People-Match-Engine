import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from match_engine.services.preprocessing.embedding_interface import (
    EmbeddingServiceProtocol,
)
from match_engine.services.preprocessing.fast_embedding_service import (
    FastEmbeddingService,
)
from match_engine.services.scoring.report import ScoringReport

logger = logging.getLogger(__name__)

BAND_WIDTH = 0.15
NEUTRAL_MEDIAN = 0.5


def band(similarity: np.ndarray, median: float) -> np.ndarray:
    """Score how complementary a pair is, given how alike their profiles read.

    Two people complement each other when their profiles are related without
    being the same, so the curve peaks at moderate similarity and falls away on
    both sides. The falling factor reaches zero at similarity one, which agrees
    with the zero that identical profiles already take on the diagonal, so a
    one-character difference in a job title produces no jump.

    The peak follows the feature's median, which puts the curve where the data
    is. The width stays a constant on purpose. A width taken from the feature's
    own spread would stretch a degenerate feature across the full range and make
    it look informative, and informativeness is measured from exactly that
    spread."""
    clipped = np.clip(similarity, 0.0, 1.0)
    offset = clipped - median
    return (1.0 - clipped) * np.exp(-(offset**2) / (2.0 * BAND_WIDTH**2))


@dataclass(frozen=True)
class EmbeddingScores:
    scores: Dict[str, Dict[str, float]]
    report: ScoringReport


class EmbeddingComplementarityScorer:
    """Scores complementarity from embedding distance, without a model call.

    A run with no API key still needs real numbers, and the alternative is a
    matrix of neutral fallbacks that reports an opinion nobody gave. The signal
    is weaker than a model's, because it comes from the same vectors similarity
    comes from, and a run says which scorer produced it."""

    def __init__(self, embedding_service: EmbeddingServiceProtocol = None):
        self._embedding_service = embedding_service

    @property
    def embedding_service(self) -> EmbeddingServiceProtocol:
        """Built on first use, because the default downloads a model."""
        if self._embedding_service is None:
            self._embedding_service = FastEmbeddingService()
        return self._embedding_service

    async def get_profile_complementarity(
        self,
        target_profiles: List[str],
        comparison_profiles: List[str],
        category: str,
    ) -> EmbeddingScores:
        vocabulary = list(dict.fromkeys(target_profiles + comparison_profiles))
        if not vocabulary:
            return EmbeddingScores(scores={}, report=ScoringReport(scorer="embedding"))

        vectors = await self._unit_vectors(vocabulary)
        position = {profile: index for index, profile in enumerate(vocabulary)}

        rows = np.array([position[profile] for profile in target_profiles])
        columns = np.array([position[profile] for profile in comparison_profiles])
        similarity = vectors[rows] @ vectors[columns].T

        scores = band(similarity, self._median(similarity))
        logger.info(
            f"Feature {category}: scored {scores.size} pairs from "
            f"{len(vocabulary)} distinct profiles without a model call"
        )

        return EmbeddingScores(
            scores={
                target: {
                    comparison: float(scores[i, j])
                    for j, comparison in enumerate(comparison_profiles)
                }
                for i, target in enumerate(target_profiles)
            },
            report=ScoringReport(scorer="embedding", scored_pairs=int(scores.size)),
        )

    async def _unit_vectors(self, vocabulary: List[str]) -> np.ndarray:
        embeddings = await self.embedding_service.get_batch_embeddings(vocabulary)
        vectors = np.array(embeddings, dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms > 0.0, norms, 1.0)

    def _median(self, similarity: np.ndarray) -> float:
        """The median of the pairs that are not a profile against itself.

        A one-profile feature has no such pair, and a neutral centre keeps the
        curve defined rather than making the caller special-case it."""
        rows, columns = similarity.shape
        if rows == columns:
            off_diagonal = similarity[~np.eye(rows, dtype=bool)]
        else:
            off_diagonal = similarity.ravel()

        if off_diagonal.size == 0:
            return NEUTRAL_MEDIAN
        return float(np.median(off_diagonal))
