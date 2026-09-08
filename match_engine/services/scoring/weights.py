from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from match_engine.services.scoring.intent import Intent
from match_engine.services.scoring.weight_resolver import normalize_weights


@dataclass(frozen=True)
class ExplicitWeights:
    """Per-feature weights a caller chose instead of having them measured."""

    similarity: Dict[str, float]
    complementarity: Dict[str, float]


class ExplicitWeightResolver:
    """Returns the caller's weights in place of measuring informativeness.

    A caller who has already decided how to weigh their features skips the
    three-factor resolution entirely. The weights still normalize to one on each
    vector, so an edge means the same thing whichever resolver produced it."""

    def __init__(self, weights: ExplicitWeights):
        self._weights = weights

    def resolve(
        self,
        raw_similarity: Dict[str, np.ndarray],
        raw_complementarity: Dict[str, np.ndarray],
        intent: Optional[Intent],
    ):
        features = set(raw_similarity) | set(raw_complementarity)
        assert (
            set(self._weights.similarity) == features
        ), f"similarity weights must name exactly {sorted(features)}"
        assert (
            set(self._weights.complementarity) == features
        ), f"complementarity weights must name exactly {sorted(features)}"

        return (
            normalize_weights(dict(self._weights.similarity)),
            normalize_weights(dict(self._weights.complementarity)),
        )
