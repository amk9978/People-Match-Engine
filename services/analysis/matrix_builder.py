import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from services.features.feature_set import FeatureSet
from services.scoring.complementarity_scorer import ComplementarityScorer
from services.scoring.llm_scorer import LLMComplementarityScorer
from services.scoring.report import ScoringReport

logger = logging.getLogger(__name__)

SELF_COMPLEMENTARITY = 0.0
UNKNOWN_PAIR_SCORE = 0.5


class MatrixBuilder:
    """Holds one profile-by-profile complementarity matrix per feature.

    A profile is the raw cell value for that feature, so two people with the same
    cell share a score and the matrices are sized by distinct profiles rather than
    by people. Matrices start raw so informativeness can be measured from their
    spread, then are replaced by calibrated ones before any edge is scored.
    """

    def __init__(self, scorer: ComplementarityScorer = None):
        self.scorer = scorer or LLMComplementarityScorer()
        self.scoring_report = ScoringReport()
        self._profiles: Dict[str, List[str]] = {}
        self._positions: Dict[str, Dict[str, int]] = {}
        self._matrices: Dict[str, np.ndarray] = {}
        self._person_profiles: Dict[int, Dict[str, str]] = {}

    async def build(self, df: pd.DataFrame, feature_set: FeatureSet) -> None:
        """Score every distinct profile pair for every feature."""
        report = ScoringReport()

        for feature in feature_set:
            profiles = self._distinct_profiles(df, feature.column)
            self._profiles[feature.name] = profiles
            self._positions[feature.name] = {
                profile: index for index, profile in enumerate(profiles)
            }

            logger.info(f"Feature {feature.name}: {len(profiles)} distinct profiles")
            result = await self.scorer.get_profile_complementarity(
                profiles, profiles, feature.name
            )
            self._matrices[feature.name] = self._as_matrix(profiles, result.scores)
            report = report.merge(result.report)

        self.scoring_report = report
        logger.info(f"Complementarity scoring complete: {report.to_dict()}")

    def _distinct_profiles(self, df: pd.DataFrame, column: str) -> List[str]:
        values = df[column].dropna().astype(str).str.strip()
        return sorted({value for value in values if value})

    def _as_matrix(
        self, profiles: List[str], scores: Dict[str, Dict[str, float]]
    ) -> np.ndarray:
        matrix = np.zeros((len(profiles), len(profiles)))
        positions = {profile: index for index, profile in enumerate(profiles)}

        for source, row in scores.items():
            i = positions.get(source)
            if i is None:
                continue
            for target, score in row.items():
                j = positions.get(target)
                if j is not None:
                    matrix[i, j] = score

        np.fill_diagonal(matrix, SELF_COMPLEMENTARITY)
        return matrix

    def raw_matrices(self) -> Dict[str, np.ndarray]:
        return dict(self._matrices)

    def apply_calibrated(self, matrices: Dict[str, np.ndarray]) -> None:
        assert set(matrices) == set(
            self._matrices
        ), "calibrated matrices must cover exactly the scored features"
        self._matrices = dict(matrices)

    def index_people(self, df: pd.DataFrame, feature_set: FeatureSet) -> None:
        """Record each person's profile string per feature, keyed by row position."""
        assert df.index.equals(
            pd.RangeIndex(len(df))
        ), "people are addressed by position; the frame must be indexed 0..n-1"

        self._person_profiles = {
            position: {
                feature.name: str(df.iloc[position].get(feature.column, "")).strip()
                for feature in feature_set
            }
            for position in range(len(df))
        }

    def _get_complementarity_score(
        self, person_i: int, person_j: int, feature_name: str
    ) -> float:
        source = self._person_profiles.get(person_i, {}).get(feature_name, "")
        target = self._person_profiles.get(person_j, {}).get(feature_name, "")
        positions = self._positions.get(feature_name, {})

        i = positions.get(source)
        j = positions.get(target)
        if i is None or j is None:
            return UNKNOWN_PAIR_SCORE

        return float(self._matrices[feature_name][i, j])

    def get_all_complementarities(
        self, person_i: int, person_j: int
    ) -> Dict[str, float]:
        return {
            feature_name: self._get_complementarity_score(
                person_i, person_j, feature_name
            )
            for feature_name in self._matrices
        }

    def clear_cache(self) -> None:
        self._person_profiles.clear()
        self._matrices.clear()
        self._profiles.clear()
        self._positions.clear()
