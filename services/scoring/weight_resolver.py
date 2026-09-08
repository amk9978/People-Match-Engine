import logging
from typing import Dict, Tuple

import numpy as np

from services.scoring.informativeness import measure_all
from services.scoring.intent import Intent

logger = logging.getLogger(__name__)


class WeightResolver:
    """Resolves the per-feature similarity and complementarity weights for one run.

    Three factors multiply. Intent says whether the user cares about a feature and
    which way. Informativeness says whether the feature can separate anyone at all.
    Calibration, which happens separately, makes the two comparable across
    features so a weight means the same thing everywhere.

        w_s[f] proportional to importance[f] * direction[f]       * info_s[f]
        w_c[f] proportional to importance[f] * (1 - direction[f]) * info_c[f]

    Both vectors are normalized to sum to one over whatever features the dataset
    happens to have.
    """

    def resolve(
        self,
        raw_similarity: Dict[str, np.ndarray],
        raw_complementarity: Dict[str, np.ndarray],
        intent: Intent,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Take the RAW matrices. Informativeness is measured here, before any
        calibration flattens the spread it reads."""
        info_s = measure_all(raw_similarity)
        info_c = measure_all(raw_complementarity)

        names = [name for name in intent.importance if name in info_s or name in info_c]
        assert names, "no feature has a matrix to weigh"

        w_s = {
            name: intent.importance[name]
            * intent.direction[name]
            * info_s.get(name, 0.0)
            for name in names
        }
        w_c = {
            name: intent.importance[name]
            * (1.0 - intent.direction[name])
            * info_c.get(name, 0.0)
            for name in names
        }

        logger.info(
            f"Informativeness: similarity {_rounded(info_s)}, "
            f"complementarity {_rounded(info_c)}"
        )
        return _normalize(w_s), _normalize(w_c)


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    """Scale to sum one, falling back to uniform when every weight is zero."""
    total = sum(weights.values())
    if total <= 0:
        share = 1.0 / len(weights)
        return {name: share for name in weights}
    return {name: value / total for name, value in weights.items()}


def _rounded(values: Dict[str, float]) -> Dict[str, float]:
    return {name: round(value, 3) for name, value in values.items()}
