#!/usr/bin/env python3

import warnings
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class FeatureAnalyzer:
    """Statistical analysis for feature importance and data engineering"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.feature_columns = {
            "role_spec": "Professional Identity - Role Specification",
            "experience": "Professional Identity - Experience Level",
            "industry": "Company Identity - Industry Classification",
            "market": "Company Market - Market Traction",
            "offering": "Company Offering - Value Proposition",
            "personas": "All Persona Titles",
        }

    def extract_tags(self, text: str, is_persona: bool = False) -> List[str]:
        """Extract tags from text"""
        if pd.isna(text):
            return []

        separator = ";" if is_persona else "|"
        tags = [tag.strip() for tag in str(text).split(separator)]
        return [tag for tag in tags if tag]

    def calculate_feature_variance(
        self, feature_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """Calculate variance metrics for each feature"""
        variance_analysis = {}

        for feature_name, embeddings in feature_embeddings.items():
            # Calculate variance across all dimensions
            dimension_variances = np.var(embeddings, axis=0)

            # Calculate pairwise distances to measure spread
            distances = pdist(embeddings, metric="cosine")

            variance_analysis[feature_name] = {
                "mean_embedding_variance": float(np.mean(dimension_variances)),
                "total_embedding_variance": float(np.sum(dimension_variances)),
                "max_dimension_variance": float(np.max(dimension_variances)),
                "mean_pairwise_distance": float(np.mean(distances)),
                "std_pairwise_distance": float(np.std(distances)),
                "distance_range": float(np.max(distances) - np.min(distances)),
            }

        return variance_analysis

    def calculate_feature_correlations(
        self, feature_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate correlations between feature embeddings"""
        correlations = {}
        feature_names = list(feature_embeddings.keys())

        for i, feature1 in enumerate(feature_names):
            for j, feature2 in enumerate(feature_names[i + 1 :], i + 1):
                # Calculate correlation between feature centroids
                centroid1 = np.mean(feature_embeddings[feature1], axis=0)
                centroid2 = np.mean(feature_embeddings[feature2], axis=0)

                correlation = np.corrcoef(centroid1, centroid2)[0, 1]
                correlations[f"{feature1}_vs_{feature2}"] = float(correlation)

                # Also calculate average pairwise correlation
                similarities = []
                for k in range(
                    min(100, len(feature_embeddings[feature1]))
                ):  # Sample for efficiency
                    sim = cosine_similarity(
                        feature_embeddings[feature1][k : k + 1],
                        feature_embeddings[feature2][k : k + 1],
                    )[0, 0]
                    similarities.append(sim)

                correlations[f"{feature1}_vs_{feature2}_pairwise_avg"] = float(
                    np.mean(similarities)
                )

        return correlations

    def analyze_tag_frequency_distribution(self) -> Dict[str, Dict]:
        """Analyze tag frequency and dominance patterns"""
        tag_analysis = {}

        for feature_name, column_name in self.feature_columns.items():
            tag_counts = {}
            total_tags = 0

            # Count all tags for this feature
            for _, row in self.df.iterrows():
                is_persona = feature_name == "personas"
                tags = self.extract_tags(row[column_name], is_persona)

                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    total_tags += 1

            if total_tags > 0:
                # Calculate distribution metrics
                frequencies = list(tag_counts.values())
                unique_tags = len(tag_counts)

                # Gini coefficient for inequality
                sorted_freq = sorted(frequencies)
                n = len(sorted_freq)
                gini = (
                    2 * sum(i * freq for i, freq in enumerate(sorted_freq, 1))
                    - (n + 1) * sum(sorted_freq)
                ) / (n * sum(sorted_freq))

                # Dominance analysis
                top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
                top_5_coverage = (
                    sum(count for _, count in top_tags[:5]) / total_tags
                    if total_tags > 0
                    else 0
                )
                top_10_coverage = (
                    sum(count for _, count in top_tags[:10]) / total_tags
                    if total_tags > 0
                    else 0
                )

                tag_analysis[feature_name] = {
                    "unique_tag_count": unique_tags,
                    "total_tag_instances": total_tags,
                    "avg_tags_per_person": total_tags / len(self.df),
                    "gini_coefficient": float(gini),
                    "top_5_tag_coverage": float(top_5_coverage),
                    "top_10_tag_coverage": float(top_10_coverage),
                    "most_common_tags": top_tags[:10],
                    "singleton_tag_count": sum(
                        1 for count in frequencies if count == 1
                    ),
                    "singleton_tag_ratio": (
                        sum(1 for count in frequencies if count == 1) / unique_tags
                        if unique_tags > 0
                        else 0
                    ),
                }

        return tag_analysis

    def calculate_mutual_information(
        self, feature_embeddings: Dict[str, np.ndarray], cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate mutual information between features and cluster assignments"""
        mi_scores = {}

        for feature_name, embeddings in feature_embeddings.items():
            # Use PCA to reduce dimensionality for MI calculation
            from sklearn.decomposition import PCA

            pca = PCA(n_components=min(10, embeddings.shape[1]))
            reduced_embeddings = pca.fit_transform(embeddings)

            # Calculate MI for each principal component
            mi_values = []
            for i in range(reduced_embeddings.shape[1]):
                mi = mutual_info_classif(
                    reduced_embeddings[:, i].reshape(-1, 1),
                    cluster_labels,
                    random_state=42,
                )[0]
                mi_values.append(mi)

            mi_scores[feature_name] = {
                "mean_mutual_info": float(np.mean(mi_values)),
                "max_mutual_info": float(np.max(mi_values)),
                "total_mutual_info": float(np.sum(mi_values)),
            }

        return mi_scores

    def feature_ablation_analysis(
        self,
        feature_embeddings: Dict[str, np.ndarray],
        feature_weights: Dict[str, float],
    ) -> Dict[str, Dict]:
        """Analyze impact of removing each feature on clustering quality"""
        ablation_results = {}

        # Calculate baseline combined similarity matrix
        baseline_similarity = self._calculate_weighted_similarity(
            feature_embeddings, feature_weights
        )
        baseline_modularity = self._calculate_modularity(baseline_similarity)

        for feature_name in feature_embeddings.keys():
            # Create feature set without this feature
            reduced_features = {
                k: v for k, v in feature_embeddings.items() if k != feature_name
            }
            reduced_weights = {
                k: v for k, v in feature_weights.items() if k != feature_name
            }

            # Renormalize weights
            total_weight = sum(reduced_weights.values())
            if total_weight > 0:
                reduced_weights = {
                    k: v / total_weight for k, v in reduced_weights.items()
                }

            # Calculate similarity without this feature
            reduced_similarity = self._calculate_weighted_similarity(
                reduced_features, reduced_weights
            )
            reduced_modularity = self._calculate_modularity(reduced_similarity)

            # Calculate impact
            ablation_results[feature_name] = {
                "modularity_drop": float(baseline_modularity - reduced_modularity),
                "similarity_correlation": float(
                    np.corrcoef(
                        baseline_similarity.flatten(), reduced_similarity.flatten()
                    )[0, 1]
                ),
                "importance_score": float(
                    abs(baseline_modularity - reduced_modularity)
                ),
            }

        return ablation_results

    def _calculate_weighted_similarity(
        self, feature_embeddings: Dict[str, np.ndarray], weights: Dict[str, float]
    ) -> np.ndarray:
        """Calculate weighted similarity matrix"""
        n_people = len(list(feature_embeddings.values())[0])
        combined_similarity = np.zeros((n_people, n_people))

        for feature_name, weight in weights.items():
            if feature_name in feature_embeddings:
                similarity_matrix = cosine_similarity(feature_embeddings[feature_name])
                combined_similarity += weight * similarity_matrix

        return combined_similarity

    def _calculate_modularity(
        self, similarity_matrix: np.ndarray, threshold: float = 0.5
    ) -> float:
        """Calculate modularity for a similarity matrix"""
        # Convert to adjacency matrix
        adjacency = (similarity_matrix > threshold).astype(int)
        np.fill_diagonal(adjacency, 0)

        # Simple modularity calculation
        m = np.sum(adjacency) / 2
        if m == 0:
            return 0.0

        degrees = np.sum(adjacency, axis=1)
        modularity = 0.0

        for i in range(len(adjacency)):
            for j in range(len(adjacency)):
                expected = (degrees[i] * degrees[j]) / (2 * m)
                modularity += adjacency[i][j] - expected

        return modularity / (2 * m)


class SubgraphAnalyzer:
    """Analyze subgraph quality using statistical metrics"""

    def __init__(self, graph, feature_embeddings: Dict[str, np.ndarray]):
        self.graph = graph
        self.feature_embeddings = feature_embeddings

    def calculate_inter_vs_intra_cluster_distances(
        self, subgraph_nodes: Set[int]
    ) -> Dict[str, Dict]:
        """Calculate inter-cluster vs outside cluster distance analysis"""
        results = {}

        subgraph_indices = list(subgraph_nodes)
        outside_indices = [
            i
            for i in range(len(list(self.feature_embeddings.values())[0]))
            if i not in subgraph_nodes
        ]

        for feature_name, embeddings in self.feature_embeddings.items():
            if len(subgraph_indices) < 2 or len(outside_indices) < 1:
                continue

            # Intra-cluster distances (within subgraph)
            subgraph_embeddings = embeddings[subgraph_indices]
            intra_distances = pdist(subgraph_embeddings, metric="cosine")

            # Inter-cluster distances (subgraph to outside)
            inter_distances = []
            for i in subgraph_indices:
                for j in outside_indices:
                    dist = (
                        1
                        - cosine_similarity(
                            embeddings[i : i + 1], embeddings[j : j + 1]
                        )[0, 0]
                    )
                    inter_distances.append(dist)

            # Calculate metrics
            results[feature_name] = {
                "mean_intra_distance": float(np.mean(intra_distances)),
                "std_intra_distance": float(np.std(intra_distances)),
                "mean_inter_distance": float(np.mean(inter_distances)),
                "std_inter_distance": float(np.std(inter_distances)),
                "separation_ratio": (
                    float(np.mean(inter_distances) / np.mean(intra_distances))
                    if np.mean(intra_distances) > 0
                    else float("inf")
                ),
                "cohesion_score": float(
                    1 / (1 + np.mean(intra_distances))
                ),  # Higher is better
                "separation_score": float(np.mean(inter_distances)),  # Higher is better
                "silhouette_approximation": float(
                    (np.mean(inter_distances) - np.mean(intra_distances))
                    / max(np.mean(inter_distances), np.mean(intra_distances))
                ),
            }

        return results

    def calculate_silhouette_scores(self, subgraph_nodes: Set[int]) -> Dict[str, Dict]:
        """Calculate detailed silhouette analysis"""
        results = {}

        # Create cluster labels (1 for subgraph, 0 for outside)
        n_total = len(list(self.feature_embeddings.values())[0])
        cluster_labels = np.array(
            [1 if i in subgraph_nodes else 0 for i in range(n_total)]
        )

        for feature_name, embeddings in self.feature_embeddings.items():
            if len(np.unique(cluster_labels)) < 2:
                continue

            # Calculate silhouette scores
            try:
                silhouette_avg = silhouette_score(
                    embeddings, cluster_labels, metric="cosine"
                )
                silhouette_samples_scores = silhouette_samples(
                    embeddings, cluster_labels, metric="cosine"
                )

                # Separate scores for subgraph and outside
                subgraph_scores = silhouette_samples_scores[cluster_labels == 1]
                outside_scores = silhouette_samples_scores[cluster_labels == 0]

                results[feature_name] = {
                    "overall_silhouette": float(silhouette_avg),
                    "subgraph_mean_silhouette": float(np.mean(subgraph_scores)),
                    "subgraph_std_silhouette": float(np.std(subgraph_scores)),
                    "subgraph_min_silhouette": float(np.min(subgraph_scores)),
                    "outside_mean_silhouette": float(np.mean(outside_scores)),
                    "subgraph_quality_ratio": (
                        float(np.mean(subgraph_scores) / np.mean(outside_scores))
                        if np.mean(outside_scores) != 0
                        else float("inf")
                    ),
                    "poorly_clustered_count": int(
                        np.sum(subgraph_scores < 0)
                    ),  # Negative silhouette
                    "well_clustered_ratio": (
                        float(np.sum(subgraph_scores > 0.5) / len(subgraph_scores))
                        if len(subgraph_scores) > 0
                        else 0.0
                    ),
                }

            except Exception as e:
                results[feature_name] = {"error": str(e)}

        return results

    def calculate_density_gradients(self, subgraph_nodes: Set[int]) -> Dict[str, float]:
        """Calculate density gradients at subgraph boundaries"""
        if not hasattr(self.graph, "edges"):
            return {}

        # Calculate internal density
        subgraph = self.graph.subgraph(subgraph_nodes)
        internal_edges = list(subgraph.edges(data=True))
        internal_weight = sum(data["weight"] for _, _, data in internal_edges)
        max_internal_edges = len(subgraph_nodes) * (len(subgraph_nodes) - 1) / 2
        internal_density = (
            internal_weight / max_internal_edges if max_internal_edges > 0 else 0
        )

        # Calculate boundary density (edges crossing the boundary)
        boundary_weight = 0
        boundary_count = 0

        for node in subgraph_nodes:
            for neighbor in self.graph.neighbors(node):
                if neighbor not in subgraph_nodes:
                    boundary_weight += self.graph[node][neighbor]["weight"]
                    boundary_count += 1

        boundary_density = boundary_weight / boundary_count if boundary_count > 0 else 0

        return {
            "internal_density": float(internal_density),
            "boundary_density": float(boundary_density),
            "density_gradient": float(internal_density - boundary_density),
            "density_ratio": (
                float(internal_density / boundary_density)
                if boundary_density > 0
                else float("inf")
            ),
        }

    def comprehensive_subgraph_analysis(
        self, subgraph_nodes: Set[int]
    ) -> Dict[str, any]:
        """Run comprehensive statistical analysis on subgraph quality"""
        return {
            "size_metrics": {
                "subgraph_size": len(subgraph_nodes),
                "total_size": len(list(self.feature_embeddings.values())[0]),
                "coverage_ratio": len(subgraph_nodes)
                / len(list(self.feature_embeddings.values())[0]),
            },
            "distance_analysis": self.calculate_inter_vs_intra_cluster_distances(
                subgraph_nodes
            ),
            "silhouette_analysis": self.calculate_silhouette_scores(subgraph_nodes),
            "density_analysis": self.calculate_density_gradients(subgraph_nodes),
        }
