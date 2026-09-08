import logging
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from services.analysis.business_analyzer import BusinessAnalyzer
from services.analysis.dataset_insights import DatasetInsightsAnalyzer
from services.analysis.scoring_report import ScoringReport
from shared.shared import FEATURE_COLUMN_MAPPING, FEATURES

logger = logging.getLogger(__name__)

SELF_COMPLEMENTARITY = 0.0
UNKNOWN_PAIR_SCORE = 0.5


class MatrixBuilder:
    """Builds one complementarity matrix per feature from the profiles in a dataset.

    A profile is the raw cell value for that feature, so two people with the same
    cell share a score and the matrices are sized by distinct profiles rather
    than by people."""

    def __init__(
        self,
        business_analyzer: BusinessAnalyzer = None,
        insight_analyzer: DatasetInsightsAnalyzer = None,
    ):
        self.business_analyzer = business_analyzer or BusinessAnalyzer()
        self.insight_analyzer = insight_analyzer or DatasetInsightsAnalyzer()
        self.scoring_report = ScoringReport()
        self._matrices = {}
        self._person_profiles = {}

    async def build_all_complementarity_matrices(
        self, csv_path: str
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        profiles_by_feature = self._extract_profiles(csv_path)

        matrices = {}
        report = ScoringReport()

        for feature, profiles in profiles_by_feature.items():
            profile_list = sorted(profiles)
            logger.info(f"Feature {feature}: {len(profile_list)} distinct profiles")

            result = await self.business_analyzer.get_profile_complementarity(
                profile_list, profile_list, feature
            )
            matrices[feature] = self._as_matrix(profile_list, result.scores)
            report = report.merge(result.report)

        self.scoring_report = report
        logger.info(f"Complementarity scoring complete: {report.to_dict()}")
        return matrices

    def _extract_profiles(self, csv_path: str) -> Dict[str, Set[str]]:
        df = pd.read_csv(csv_path)
        profiles = {}

        for feature in FEATURES:
            column = FEATURE_COLUMN_MAPPING[feature]
            if column not in df.columns:
                logger.warning(f"Feature {feature}: column {column} is absent")
                profiles[feature] = set()
                continue

            values = df[column].dropna().astype(str).str.strip()
            profiles[feature] = {value for value in values if value}

        return profiles

    def _as_matrix(
        self, profile_list: List[str], scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        matrix = {}
        for source in profile_list:
            row = dict(scores.get(source, {}))
            row[source] = SELF_COMPLEMENTARITY
            matrix[source] = row
        return matrix

    def load_matrices_into_memory(
        self, matrices: Dict[str, Dict[str, Dict[str, float]]]
    ) -> None:
        self._matrices = dict(matrices)

    def get_complementarity_matrices(self) -> Dict[str, np.ndarray]:
        """Render each feature's matrix as a numpy array for the insight analyzer."""
        numpy_matrices = {}

        for feature in FEATURES:
            if feature not in self._matrices:
                continue

            matrix_dict = self._matrices[feature]
            profiles = list(matrix_dict.keys())
            numpy_matrix = np.zeros((len(profiles), len(profiles)))

            for i, source in enumerate(profiles):
                row = matrix_dict[source]
                for j, target in enumerate(profiles):
                    if target in row:
                        numpy_matrix[i, j] = row[target]

            numpy_matrices[feature] = self.insight_analyzer.normalize_matrix(
                numpy_matrix, preserve_diagonal=False
            )

        return numpy_matrices

    async def precompute_person_tags(self, df: pd.DataFrame, embedding_builder) -> None:
        """Record each person's profile string per feature, keyed by row position."""
        assert df.index.equals(
            pd.RangeIndex(len(df))
        ), "people are addressed by position; the frame must be indexed 0..n-1"

        self._person_profiles = {}
        for position in range(len(df)):
            row = df.iloc[position]
            self._person_profiles[position] = {
                feature: str(row.get(column, "")).strip()
                for feature, column in FEATURE_COLUMN_MAPPING.items()
            }

    def _get_complementarity_score(
        self, person_i: int, person_j: int, category: str
    ) -> float:
        source = self._person_profiles.get(person_i, {}).get(category, "")
        target = self._person_profiles.get(person_j, {}).get(category, "")

        if not source or not target:
            return UNKNOWN_PAIR_SCORE

        matrix = self._matrices.get(category, {})

        if source in matrix and target in matrix[source]:
            return matrix[source][target]
        if target in matrix and source in matrix[target]:
            return matrix[target][source]

        return UNKNOWN_PAIR_SCORE

    def get_all_complementarities(
        self, person_i: int, person_j: int
    ) -> Dict[str, float]:
        return {
            feature: self._get_complementarity_score(person_i, person_j, feature)
            for feature in FEATURES
        }

    def clear_cache(self) -> None:
        self._person_profiles.clear()
        self._matrices.clear()
