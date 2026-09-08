from typing import Dict

import numpy as np
from scipy.stats import rankdata


def rank_normalize(matrix: np.ndarray, preserve_diagonal: bool = True) -> np.ndarray:
    """Replace each off-diagonal score with its percentile among that feature's pairs.

    Every feature ends up uniform on [0, 1], so 0.8 means the 80th percentile pair
    in this dataset whichever feature produced it. The transform is monotone, so
    it cannot reorder pairs within a feature, and it hands no part of the scale to
    the one unusually alike pair that min-max normalization would.
    """
    if matrix.size == 0 or matrix.shape[0] < 2:
        return matrix.copy()

    upper = np.triu_indices_from(matrix, k=1)
    values = matrix[upper]
    if values.size == 0:
        return matrix.copy()

    percentiles = (rankdata(values, method="average") - 0.5) / values.size

    normalized = np.zeros_like(matrix, dtype=float)
    normalized[upper] = percentiles
    normalized = normalized + normalized.T

    if preserve_diagonal:
        np.fill_diagonal(normalized, np.diag(matrix))

    return normalized


def calibrate_all(
    matrices: Dict[str, np.ndarray], preserve_diagonal: bool = True
) -> Dict[str, np.ndarray]:
    return {
        name: rank_normalize(matrix, preserve_diagonal)
        for name, matrix in matrices.items()
    }
