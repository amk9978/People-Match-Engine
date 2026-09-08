from typing import Dict

import numpy as np

FULL_SPREAD_IQR = 0.5


def informativeness(matrix: np.ndarray) -> float:
    """How well a feature separates two people, from the spread of its raw scores.

    The interquartile range of the off-diagonal values, divided by the range a
    uniformly spread feature would show. A feature scoring every pair alike gets
    0.0 and contributes nothing to the edge weight. The IQR is used rather than
    the full range because one unusual pair should not set the scale.

    Must be measured before calibration. Rank normalization makes every feature
    uniform by construction, so measuring afterwards returns the same number for
    every feature and destroys the signal.
    """
    if matrix.size == 0 or matrix.shape[0] < 2:
        return 0.0

    values = matrix[np.triu_indices_from(matrix, k=1)]
    if values.size == 0:
        return 0.0

    spread = float(np.percentile(values, 75) - np.percentile(values, 25))
    return float(np.clip(spread / FULL_SPREAD_IQR, 0.0, 1.0))


def measure_all(matrices: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {name: informativeness(matrix) for name, matrix in matrices.items()}
