from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shared.shared import FEATURES


@dataclass
class FeatureInsight:
    feature_name: str
    similarity_mean: float
    similarity_variance: float
    complementarity_mean: float
    complementarity_variance: float
    diversity_level: str
    complementarity_potential: str
    homogeneity_score: float


class DatasetInsightsAnalyzer:
    def __init__(self):
        self.insights = {}

    @staticmethod
    def normalize_matrix(
        matrix: np.ndarray, preserve_diagonal: bool = True
    ) -> np.ndarray:
        """
        Min-max normalize matrix to [0,1] range while preserving relationships.

        Args:
            matrix: Input matrix to normalize
            preserve_diagonal: Whether to preserve diagonal values (for similarity/complementarity semantics)

        Returns:
            Normalized matrix in [0,1] range
        """
        if matrix.size == 0:
            return matrix

        # Store original diagonal if preserving
        original_diagonal = np.diag(matrix).copy() if preserve_diagonal else None

        # Get upper triangle values (excluding diagonal) for min/max calculation
        upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]

        if len(upper_triangle) == 0:
            return matrix

        min_val = np.min(upper_triangle)
        max_val = np.max(upper_triangle)

        # Handle case where all values are identical
        if max_val == min_val:
            normalized = np.ones_like(matrix) * 0.5
        else:
            # Normalize entire matrix
            normalized = (matrix - min_val) / (max_val - min_val)

        # Restore original diagonal if preserving
        if preserve_diagonal and original_diagonal is not None:
            np.fill_diagonal(normalized, original_diagonal)

        return normalized

    def analyze_feature_matrices(
        self,
        similarity_matrices: Dict[str, np.ndarray],
        complementarity_matrices: Dict[str, np.ndarray],
    ) -> Dict[str, FeatureInsight]:
        """
        Analyze similarity and complementarity matrices to extract dataset characteristics.

        Args:
            similarity_matrices: Dict mapping feature names to similarity matrices
            complementarity_matrices: Dict mapping feature names to complementarity matrices

        Returns:
            Dict mapping feature names to FeatureInsight objects
        """
        insights = {}

        for feature in FEATURES:
            if feature in similarity_matrices and feature in complementarity_matrices:
                insight = self._analyze_single_feature(
                    feature,
                    similarity_matrices[feature],
                    complementarity_matrices[feature],
                )
                insights[feature] = insight

        return insights

    def analyze_dataset_directly(self, df: pd.DataFrame) -> Dict[str, FeatureInsight]:
        """
        Analyze dataset directly without pre-computed matrices (for testing).
        Computes quick similarity/complementarity estimates from raw data.

        Args:
            df: DataFrame with columns matching FEATURES

        Returns:
            Dict mapping feature names to FeatureInsight objects
        """
        insights = {}

        for feature in FEATURES:
            if feature in df.columns:
                column_data = df[feature].fillna("").astype(str)

                sim_matrix, comp_matrix = self._compute_quick_matrices(column_data)

                insight = self._analyze_single_feature(feature, sim_matrix, comp_matrix)
                insights[feature] = insight

        return insights

    def _compute_quick_matrices(
        self, column_data: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute quick similarity and complementarity matrices from text data."""
        if len(column_data) < 2:
            size = len(column_data)
            return np.ones((size, size)), np.zeros((size, size))

        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(column_data)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            complementarity_matrix = 1.0 - similarity_matrix
            np.fill_diagonal(complementarity_matrix, 0)

            return similarity_matrix, complementarity_matrix

        except ValueError:
            size = len(column_data)
            return np.ones((size, size)) * 0.5, np.ones((size, size)) * 0.5

    def _analyze_single_feature(
        self,
        feature_name: str,
        similarity_matrix: np.ndarray,
        complementarity_matrix: np.ndarray,
    ) -> FeatureInsight:
        """Analyze a single feature's matrices to extract insights."""

        sim_upper_triangle = self._get_upper_triangle_values(similarity_matrix)
        comp_upper_triangle = self._get_upper_triangle_values(complementarity_matrix)

        sim_mean = float(np.mean(sim_upper_triangle))
        sim_variance = float(np.var(sim_upper_triangle))
        comp_mean = float(np.mean(comp_upper_triangle))
        comp_variance = float(np.var(comp_upper_triangle))

        diversity_level = self._classify_diversity_level(sim_mean, sim_variance)
        complementarity_potential = self._classify_complementarity_potential(
            comp_mean, comp_variance
        )
        homogeneity_score = self._calculate_homogeneity_score(sim_mean, sim_variance)

        return FeatureInsight(
            feature_name=feature_name,
            similarity_mean=sim_mean,
            similarity_variance=sim_variance,
            complementarity_mean=comp_mean,
            complementarity_variance=comp_variance,
            diversity_level=diversity_level,
            complementarity_potential=complementarity_potential,
            homogeneity_score=homogeneity_score,
        )

    def _get_upper_triangle_values(self, matrix: np.ndarray) -> np.ndarray:
        """Extract upper triangle values from symmetric matrix (excluding diagonal)."""
        return matrix[np.triu_indices_from(matrix, k=1)]

    def _classify_diversity_level(self, sim_mean: float, sim_variance: float) -> str:
        """Classify the diversity level based on similarity statistics."""
        if sim_mean > 0.8:
            return "very_low"
        elif sim_mean > 0.6:
            return "low"
        elif sim_mean > 0.4:
            return "moderate"
        elif sim_mean > 0.2:
            return "high"
        else:
            return "very_high"

    def _classify_complementarity_potential(
        self, comp_mean: float, comp_variance: float
    ) -> str:
        """Classify complementarity potential based on complementarity statistics."""
        if comp_mean > 0.7:
            return "excellent"
        elif comp_mean > 0.5:
            return "good"
        elif comp_mean > 0.3:
            return "moderate"
        else:
            return "limited"

    def _calculate_homogeneity_score(
        self, sim_mean: float, sim_variance: float
    ) -> float:
        """Calculate a homogeneity score combining mean and variance."""
        return float(sim_mean * (1.0 - sim_variance))

    def generate_context_summary(self, insights: Dict[str, FeatureInsight]) -> str:
        """Generate a human-readable context summary for ChatGPT."""
        summary_parts = []

        for feature, insight in insights.items():
            feature_desc = self._generate_feature_description(insight)
            summary_parts.append(f"{feature}: {feature_desc}")

        return "Dataset characteristics: " + "; ".join(summary_parts)

    def _generate_feature_description(self, insight: FeatureInsight) -> str:
        """Generate a business-readable description of a single feature's characteristics."""

        # Interpret similarity levels
        if insight.similarity_mean > 0.7:
            sim_desc = "very homogeneous"
        elif insight.similarity_mean > 0.5:
            sim_desc = "moderately homogeneous"
        elif insight.similarity_mean > 0.3:
            sim_desc = "diverse"
        else:
            sim_desc = "extremely diverse"

        # Interpret complementarity potential
        if insight.complementarity_mean > 0.7:
            comp_desc = "excellent matching opportunities"
        elif insight.complementarity_mean > 0.5:
            comp_desc = "good matching potential"
        elif insight.complementarity_mean > 0.3:
            comp_desc = "limited matching opportunities"
        else:
            comp_desc = "very limited matching opportunities"

        # Provide actionable business context
        business_insight = f"{sim_desc}, {comp_desc}"

        # Add specific guidance based on the pattern
        if insight.similarity_mean < 0.3 and insight.complementarity_mean > 0.7:
            business_insight += (
                " - lower both weights since diversity is already maximized"
            )
        elif insight.similarity_mean > 0.7 and insight.complementarity_mean < 0.3:
            business_insight += " - increase complementarity weights to add diversity"

        return business_insight

    def get_feature_recommendations(
        self, insights: Dict[str, FeatureInsight]
    ) -> Dict[str, str]:
        """Generate weight adjustment recommendations based on insights."""
        recommendations = {}

        for feature, insight in insights.items():
            if insight.diversity_level in ["very_low", "low"]:
                if insight.complementarity_potential in ["limited", "moderate"]:
                    recommendations[feature] = "reduce_both_weights"
                else:
                    recommendations[feature] = "emphasize_complementarity"
            elif insight.diversity_level in ["high", "very_high"]:
                if insight.complementarity_potential in ["excellent", "good"]:
                    recommendations[feature] = "balanced_or_emphasize_complementarity"
                else:
                    recommendations[feature] = "emphasize_similarity"
            else:
                recommendations[feature] = "balanced_weights"

        return recommendations
