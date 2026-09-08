import json
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from match_engine.services.scoring.intent import Intent
from match_engine.services.scoring.weight_resolver import normalize_weights


class WeightsError(ValueError):
    """The supplied weights cannot be read as a pair of per-feature vectors."""


@dataclass(frozen=True)
class ExplicitWeights:
    """Per-feature weights a caller chose instead of having them measured."""

    similarity: Dict[str, float]
    complementarity: Dict[str, float]

    @classmethod
    def parse(cls, document: str) -> "ExplicitWeights":
        """Read weights from a JSON object with a similarity and a
        complementarity map, each keyed by feature name."""
        try:
            payload = json.loads(document)
        except json.JSONDecodeError as broken:
            raise WeightsError(f"weights must be JSON: {broken}") from broken

        if not isinstance(payload, dict):
            raise WeightsError("weights must be a JSON object")

        vectors = {}
        for key in ("similarity", "complementarity"):
            vector = payload.get(key)
            if not isinstance(vector, dict) or not vector:
                raise WeightsError(f"weights need a non-empty {key} object")
            for name, value in vector.items():
                if not isinstance(value, (int, float)) or value < 0:
                    raise WeightsError(f"{key}.{name} must be zero or more")
            vectors[key] = {name: float(value) for name, value in vector.items()}

        return cls(
            similarity=vectors["similarity"],
            complementarity=vectors["complementarity"],
        )


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
