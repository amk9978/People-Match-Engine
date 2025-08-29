import itertools
import logging
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class SubgraphAnalyzer:
    """Handles subgraph analysis and insights"""

    def __init__(self):
        pass

    def get_subgraph_info(
        self,
        nodes: Set[int],
        feature_embeddings: Dict[str, np.ndarray],
        df: pd.DataFrame,
        graph: nx.Graph,
        matrix_builder=None,
        tuned_w_s: Optional[Dict[str, float]] = None,
        tuned_w_c: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Get detailed information about a subgraph"""
        if not nodes:
            return {"size": 0, "density": 0.0, "members": []}

        # Create mapping from node IDs to DataFrame positions for embedding indexing
        node_to_pos = {node_id: pos for pos, node_id in enumerate(df.index)}

        valid_nodes = {node for node in nodes if node in df.index}
        invalid_nodes = [node for node in nodes if node not in df.index]

        if invalid_nodes:
            logger.warning(
                f"Filtering out invalid nodes not in DataFrame: {invalid_nodes}"
            )

        members = []
        for node in valid_nodes:
            person_data = df.loc[node]
            members.append(
                {
                    "name": person_data["Person Name"],
                    "linkedin": person_data.get("LinkedIn URL", ""),
                }
            )

        subgraph = graph.subgraph(valid_nodes)
        density = self.calculate_subgraph_density(valid_nodes, graph)

        edges_with_weights = subgraph.edges(data=True)
        if edges_with_weights:
            total_weight = sum(
                data.get("weight", 0.0) for _, _, data in edges_with_weights
            )
            avg_weight = total_weight / len(edges_with_weights)
        else:
            avg_weight = 0.0

        edges_data = []
        for u, v, data in edges_with_weights:
            if u in df.index and v in df.index:
                source_name = df.loc[u]["Person Name"]
                target_name = df.loc[v]["Person Name"]
                edges_data.append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "weight": data.get("weight", 0.0),
                        "score": data.get("weight", 0.0),
                    }
                )
            else:
                logger.warning(f"Skipping edge ({u}, {v}) - nodes not in DataFrame")

        hybrid_insights = {}
        complementarity_insights = {}
        feature_importance_analysis = {}
        dataset_values_analysis = {}

        if matrix_builder and tuned_w_s and tuned_w_c:
            complementarity_insights = self.analyze_complementarity_centroids(
                valid_nodes, matrix_builder, tuned_w_s, tuned_w_c
            )

            hybrid_insights = self.analyze_hybrid_centroids(
                valid_nodes,
                feature_embeddings,
                matrix_builder,
                tuned_w_s,
                tuned_w_c,
                df,
            )

            feature_importance_analysis = self.analyze_feature_importance(
                valid_nodes, graph, matrix_builder, tuned_w_s, tuned_w_c
            )

            dataset_values_analysis = self.analyze_dataset_values(
                valid_nodes, df, matrix_builder
            )

        weighted_communities = self.detect_weighted_communities(valid_nodes, graph)

        max_cycle = self.find_maximum_weight_cycle(valid_nodes, graph)

        layout_coords = self.compute_stress_layout(valid_nodes, graph)

        replication_recommendations = {}
        if matrix_builder and tuned_w_s and tuned_w_c and feature_embeddings:
            replication_recommendations = self.get_optimal_nodes_for_replication(
                df, matrix_builder, feature_embeddings, tuned_w_s, tuned_w_c, top_k=5
            )

        result = {
            "nodes": valid_nodes,
            "size": len(valid_nodes),
            "density": density,
            "avg_edge_weight": avg_weight,
            "members": members,
            "edges": len(edges_with_weights),
            "edges_data": edges_data,
            "centroid_insights": self.analyze_subgraph_centroids(
                valid_nodes, feature_embeddings, df
            ),
            "complementarity_insights": complementarity_insights,
            "hybrid_insights": hybrid_insights,
            "feature_importance": feature_importance_analysis,
            "dataset_values": dataset_values_analysis,
            "weighted_communities": weighted_communities,
            "maximum_weight_cycle": max_cycle,
            "stress_layout": layout_coords,
            "replication_recommendations": replication_recommendations,
        }

        legacy_subgroups = self.analyze_subgroups(valid_nodes, graph, df)
        result.update(legacy_subgroups)
        return result

    def calculate_subgraph_density(self, nodes: Set[int], graph: nx.Graph) -> float:
        """Calculate WEIGHTED density of a subgraph given a set of nodes"""
        if len(nodes) < 2:
            return 0.0

        subgraph = graph.subgraph(nodes)

        total_weight = sum(
            data.get("weight", 0.0) for _, _, data in subgraph.edges(data=True)
        )

        num_nodes = len(nodes)
        max_possible_edges = num_nodes * (num_nodes - 1) / 2

        return total_weight / max_possible_edges if max_possible_edges > 0 else 0.0

    def analyze_subgraph_centroids(
        self,
        nodes: Set[int],
        feature_embeddings: Dict[str, np.ndarray],
        df: pd.DataFrame,
    ) -> Dict:
        """Analyze centroids to identify which feature values make the subgraph dense"""
        # Create mapping from node IDs to DataFrame positions for embedding indexing
        node_to_pos = {node_id: pos for pos, node_id in enumerate(df.index)}

        centroids = {}

        for feature_name, embeddings in feature_embeddings.items():
            valid_positions = [
                node_to_pos[node] for node in nodes if node in node_to_pos
            ]
            subgraph_embeddings = embeddings[valid_positions]
            centroid = np.mean(subgraph_embeddings, axis=0)

            closest_to_centroid = self._find_nodes_closest_to_centroid(
                nodes, embeddings, centroid, feature_name, node_to_pos
            )

            centroids[feature_name] = {
                "centroid": centroid,
                "closest_tags": [],
                "closest_nodes_to_centroid": closest_to_centroid,
            }

        return centroids

    def _find_nodes_closest_to_centroid(
        self,
        nodes: Set[int],
        embeddings: np.ndarray,
        centroid: np.ndarray,
        feature_name: str,
        node_to_pos: Dict[int, int],
        top_k: int = 5,
    ) -> List[Dict]:
        """Find nodes within subgraph that are closest to centroid - these drive density"""
        if not nodes:
            return []

        node_list = list(nodes)
        valid_positions = [
            node_to_pos[node] for node in node_list if node in node_to_pos
        ]
        subgraph_embeddings = embeddings[valid_positions]

        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            return []

        similarities = []
        for i, node_embedding in enumerate(subgraph_embeddings):
            embedding_norm = np.linalg.norm(node_embedding)
            if embedding_norm > 0:
                similarity = np.dot(node_embedding, centroid) / (
                    embedding_norm * centroid_norm
                )
                similarities.append(
                    {
                        "node_id": node_list[i],
                        "similarity_to_centroid": float(similarity),
                        "feature": feature_name,
                    }
                )

        similarities.sort(key=lambda x: x["similarity_to_centroid"], reverse=True)
        return similarities[:top_k]

    def analyze_complementarity_centroids(
        self,
        nodes: Set[int],
        matrix_builder,
        tuned_w_s: Dict[str, float],
        tuned_w_c: Dict[str, float],
    ) -> Dict:
        """Analyze complementarity patterns within the subgraph using tuned weights"""
        if not nodes or len(nodes) < 2:
            return {
                "complementarity_patterns": {},
                "summary": "Insufficient nodes for complementarity analysis",
            }

        node_list = list(nodes)
        feature_categories = [
            "role",
            "experience",
            "persona",
            "industry",
            "market",
            "offering",
        ]

        complementarity_patterns = {}

        for category in feature_categories:

            scores = []
            relationships = []

            for i, node_i in enumerate(node_list):
                for j, node_j in enumerate(node_list[i + 1 :], start=i + 1):
                    comp_scores = matrix_builder.get_all_complementarities(
                        node_i, node_j
                    )
                    if category in comp_scores:
                        score = comp_scores[category]
                        scores.append(score)
                        relationships.append(
                            {
                                "node_i": node_i,
                                "node_j": node_j,
                                "score": score,
                                "tuned_weight": tuned_w_c.get(category, 1.0),
                            }
                        )

            if scores:

                relationships.sort(key=lambda x: x["score"], reverse=True)

                complementarity_patterns[category] = {
                    "avg_complementarity": float(np.mean(scores)),
                    "max_complementarity": float(max(scores)),
                    "min_complementarity": float(min(scores)),
                    "tuned_weight": tuned_w_c.get(category, 1.0),
                    "weighted_avg": float(
                        np.mean(scores) * tuned_w_c.get(category, 1.0)
                    ),
                    "top_complementary_pairs": relationships[:5],
                    "total_pairs": len(relationships),
                }
            else:
                complementarity_patterns[category] = {
                    "avg_complementarity": 0.0,
                    "max_complementarity": 0.0,
                    "min_complementarity": 0.0,
                    "tuned_weight": tuned_w_c.get(category, 1.0),
                    "weighted_avg": 0.0,
                    "top_complementary_pairs": [],
                    "total_pairs": 0,
                }

        weighted_scores = []
        for category, pattern in complementarity_patterns.items():
            if pattern["total_pairs"] > 0:
                weighted_scores.append(pattern["weighted_avg"])

        overall_complementarity = (
            float(np.mean(weighted_scores)) if weighted_scores else 0.0
        )

        strongest_category = max(
            complementarity_patterns.keys(),
            key=lambda cat: complementarity_patterns[cat]["weighted_avg"],
        )

        return {
            "complementarity_patterns": complementarity_patterns,
            "overall_complementarity_strength": overall_complementarity,
            "strongest_complementarity_category": strongest_category,
            "tuning_summary": {
                "similarity_weights_used": tuned_w_s,
                "complementarity_weights_used": tuned_w_c,
            },
        }

    def analyze_subgroups(
        self, nodes: Set[int], graph: nx.Graph, df: pd.DataFrame
    ) -> Dict:
        """Analyze cohesive subgroups within the dense subgraph"""
        if len(nodes) < 4:
            return {
                "subgroups": [],
                "subgroup_summary": {
                    "total_subgroups": 0,
                    "strongest_subgroup_strength": 0.0,
                    "avg_subgroup_density": 0.0,
                },
            }

        subgraph = graph.subgraph(nodes)

        try:
            import networkx.algorithms.community as nx_comm

            communities = list(
                nx_comm.greedy_modularity_communities(subgraph, weight="weight")
            )
        except:
            communities = list(nx.connected_components(subgraph))

        subgroups = []
        total_density = 0.0
        max_strength = 0.0

        for i, community in enumerate(communities):
            if len(community) >= 3:
                subgroup_density = self.calculate_subgraph_density(community, graph)
                total_density += subgroup_density

                community_subgraph = subgraph.subgraph(community)
                edges_with_weights = community_subgraph.edges(data=True)
                if edges_with_weights:
                    avg_weight = sum(
                        data.get("weight", 0.0) for _, _, data in edges_with_weights
                    ) / len(edges_with_weights)
                    max_strength = max(max_strength, avg_weight)
                else:
                    avg_weight = 0.0

                members = []
                for node in community:
                    person_data = df.loc[node]
                    members.append(
                        {
                            "name": person_data["Person Name"],
                            "company": person_data["Person Company"],
                            "role": person_data.get(
                                "Professional Identity - Role Specification", ""
                            ),
                        }
                    )

                subgroups.append(
                    {
                        "nodes": community,
                        "size": len(community),
                        "density": subgroup_density,
                        "connection_strength": avg_weight,
                        "members": members,
                    }
                )

        subgroups.sort(key=lambda x: x["connection_strength"], reverse=True)

        avg_density = total_density / len(subgroups) if subgroups else 0.0

        return {
            "subgroups": subgroups,
            "subgroup_summary": {
                "total_subgroups": len(subgroups),
                "strongest_subgroup_strength": max_strength,
                "avg_subgroup_density": avg_density,
            },
        }

    def detect_weighted_communities(self, nodes: Set[int], graph: nx.Graph) -> Dict:
        """Detect communities in weighted subgraph using algorithms optimized for dense/complete graphs"""
        if len(nodes) < 3:
            return {
                "communities": [],
                "summary": "Insufficient nodes for community detection",
            }

        subgraph = graph.subgraph(nodes)

        # Calculate graph density to determine algorithm selection
        density = self.calculate_subgraph_density(nodes, graph)
        is_dense_graph = (
            density > 0.3
        )  # Dense if >30% of possible edges exist with significant weight

        communities_results = {}

        # For dense/complete graphs, use specialized algorithms
        if is_dense_graph:
            logger.info(
                f"Dense graph detected (density: {density:.3f}). Using specialized algorithms."
            )

            # 1. Spectral Clustering - Best for complete weighted graphs
            try:
                communities_results["spectral"] = self._spectral_clustering_communities(
                    nodes, subgraph
                )
            except Exception as e:
                logger.info(f"Spectral clustering failed: {e}")
                communities_results["spectral"] = {"communities": [], "error": str(e)}

            # 2. Hierarchical Clustering on weight matrix
            try:
                communities_results["hierarchical"] = (
                    self._hierarchical_clustering_communities(nodes, subgraph)
                )
            except Exception as e:
                logger.info(f"Hierarchical clustering failed: {e}")
                communities_results["hierarchical"] = {
                    "communities": [],
                    "error": str(e),
                }

            # 3. Threshold-based community detection
            try:
                communities_results["threshold_based"] = (
                    self._threshold_based_communities(nodes, subgraph)
                )
            except Exception as e:
                logger.info(f"Threshold-based failed: {e}")
                communities_results["threshold_based"] = {
                    "communities": [],
                    "error": str(e),
                }

            # 4. Multi-resolution Leiden (better than Louvain for dense graphs)
            try:
                communities_results["multi_resolution"] = (
                    self._multi_resolution_communities(nodes, subgraph)
                )
            except Exception as e:
                logger.info(f"Multi-resolution failed: {e}")
                communities_results["multi_resolution"] = {
                    "communities": [],
                    "error": str(e),
                }

        else:
            # For sparse graphs, use traditional methods
            logger.info(
                f"Sparse graph detected (density: {density:.3f}). Using traditional algorithms."
            )

            try:
                louvain_communities = list(
                    nx_comm.louvain_communities(subgraph, weight="weight", seed=42)
                )
                communities_results["louvain"] = self._analyze_communities(
                    louvain_communities, subgraph, "Louvain (weighted)"
                )
            except Exception as e:
                logger.info(f"Louvain failed: {e}")
                communities_results["louvain"] = {"communities": [], "error": str(e)}

            try:
                greedy_communities = list(
                    nx_comm.greedy_modularity_communities(subgraph, weight="weight")
                )
                communities_results["greedy_modularity"] = self._analyze_communities(
                    greedy_communities, subgraph, "Greedy Modularity (weighted)"
                )
            except Exception as e:
                logger.info(f"Greedy modularity failed: {e}")
                communities_results["greedy_modularity"] = {
                    "communities": [],
                    "error": str(e),
                }

        best_method = self._select_best_communities_for_dense_graphs(
            communities_results, is_dense_graph
        )

        return {
            "graph_density": density,
            "is_dense_graph": is_dense_graph,
            "all_methods": communities_results,
            "best_method": best_method,
            "communities": communities_results.get(best_method, {}).get(
                "communities", []
            ),
            "summary": self._summarize_communities(
                communities_results.get(best_method, {})
            ),
        }

    def _analyze_communities(
        self, communities, subgraph: nx.Graph, method_name: str
    ) -> Dict:
        """Analyze detected communities for quality metrics"""
        analyzed_communities = []

        for i, community in enumerate(communities):
            if len(community) < 2:
                continue

            community_subgraph = subgraph.subgraph(community)

            internal_weight = sum(
                data.get("weight", 0.0)
                for _, _, data in community_subgraph.edges(data=True)
            )

            external_weight = 0.0
            for node in community:
                for neighbor in subgraph.neighbors(node):
                    if neighbor not in community:
                        edge_data = subgraph.get_edge_data(node, neighbor)
                        external_weight += (
                            edge_data.get("weight", 0.0) if edge_data else 0.0
                        )

            density = self.calculate_subgraph_density(community, subgraph)

            analyzed_communities.append(
                {
                    "id": i,
                    "nodes": list(community),
                    "size": len(community),
                    "internal_weight": internal_weight,
                    "external_weight": external_weight,
                    "density": density,
                    "cohesion_ratio": (
                        internal_weight / (internal_weight + external_weight)
                        if (internal_weight + external_weight) > 0
                        else 0.0
                    ),
                }
            )

        total_modularity = 0.0
        try:
            import networkx.algorithms.community as nx_comm

            if len(communities) > 1:
                total_modularity = nx_comm.modularity(
                    subgraph, communities, weight="weight"
                )
        except:
            total_modularity = 0.0

        return {
            "method": method_name,
            "communities": analyzed_communities,
            "num_communities": len(analyzed_communities),
            "modularity": total_modularity,
            "avg_community_size": (
                np.mean([c["size"] for c in analyzed_communities])
                if analyzed_communities
                else 0.0
            ),
        }

    def _spectral_clustering_communities(
        self, nodes: Set[int], subgraph: nx.Graph
    ) -> Dict:
        """Use spectral clustering optimized for complete weighted graphs"""
        try:
            import numpy as np
            from sklearn.cluster import SpectralClustering

            node_list = list(nodes)
            n_nodes = len(node_list)

            # Build adjacency matrix with weights
            adj_matrix = np.zeros((n_nodes, n_nodes))
            edge_weights = []

            for i, node_i in enumerate(node_list):
                for j, node_j in enumerate(node_list):
                    if i != j and subgraph.has_edge(node_i, node_j):
                        weight = subgraph.get_edge_data(node_i, node_j).get(
                            "weight", 0.0
                        )
                        adj_matrix[i, j] = weight
                        edge_weights.append(weight)

            if not edge_weights:
                return {"communities": [], "error": "No weighted edges found"}

            # Determine optimal number of clusters based on weight distribution
            n_clusters_range = range(2, min(8, max(3, n_nodes // 4)))
            best_communities = []
            best_score = -1
            best_n_clusters = 2

            for n_clusters in n_clusters_range:
                try:
                    # Use spectral clustering with different parameters
                    spectral = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity="precomputed",
                        random_state=42,
                        assign_labels="kmeans",
                    )

                    cluster_labels = spectral.fit_predict(adj_matrix)

                    # Convert labels to communities
                    communities = []
                    for cluster_id in range(n_clusters):
                        community = [
                            node_list[i]
                            for i, label in enumerate(cluster_labels)
                            if label == cluster_id
                        ]
                        if (
                            len(community) >= 2
                        ):  # Only include communities with at least 2 nodes
                            communities.append(set(community))

                    if len(communities) > 1:
                        # Calculate silhouette-like score for this clustering
                        score = self._evaluate_dense_graph_communities(
                            communities, subgraph, adj_matrix, node_list
                        )

                        if score > best_score:
                            best_score = score
                            best_communities = communities
                            best_n_clusters = n_clusters

                except Exception as e:
                    logger.info(
                        f"Spectral clustering failed for n_clusters={n_clusters}: {e}"
                    )
                    continue

            return self._analyze_communities(
                best_communities, subgraph, f"Spectral Clustering (k={best_n_clusters})"
            )

        except ImportError:
            return {"communities": [], "error": "sklearn not available"}
        except Exception as e:
            return {"communities": [], "error": str(e)}

    def _hierarchical_clustering_communities(
        self, nodes: Set[int], subgraph: nx.Graph
    ) -> Dict:
        """Use hierarchical clustering on the distance matrix derived from edge weights"""
        try:
            import numpy as np
            from scipy.spatial.distance import pdist, squareform
            from sklearn.cluster import AgglomerativeClustering

            node_list = list(nodes)
            n_nodes = len(node_list)

            # Build distance matrix (1 - normalized_weight)
            distance_matrix = np.ones((n_nodes, n_nodes))
            edge_weights = []

            for i, node_i in enumerate(node_list):
                for j, node_j in enumerate(node_list):
                    if i != j and subgraph.has_edge(node_i, node_j):
                        weight = subgraph.get_edge_data(node_i, node_j).get(
                            "weight", 0.0
                        )
                        edge_weights.append(weight)

            if not edge_weights:
                return {"communities": [], "error": "No weighted edges found"}

            # Normalize weights to [0, 1] and convert to distances
            min_weight = min(edge_weights)
            max_weight = max(edge_weights)
            weight_range = max_weight - min_weight

            for i, node_i in enumerate(node_list):
                distance_matrix[i, i] = 0  # Distance to self is 0
                for j, node_j in enumerate(node_list):
                    if i != j and subgraph.has_edge(node_i, node_j):
                        weight = subgraph.get_edge_data(node_i, node_j).get(
                            "weight", 0.0
                        )
                        if weight_range > 0:
                            normalized_weight = (weight - min_weight) / weight_range
                            distance_matrix[i, j] = 1.0 - normalized_weight
                        else:
                            distance_matrix[i, j] = 0.5

            # Try different numbers of clusters
            best_communities = []
            best_score = -1
            best_n_clusters = 2

            for n_clusters in range(2, min(8, max(3, n_nodes // 3))):
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=n_clusters, metric="precomputed", linkage="average"
                    )

                    cluster_labels = clustering.fit_predict(distance_matrix)

                    # Convert to communities
                    communities = []
                    for cluster_id in range(n_clusters):
                        community = [
                            node_list[i]
                            for i, label in enumerate(cluster_labels)
                            if label == cluster_id
                        ]
                        if len(community) >= 2:
                            communities.append(set(community))

                    if len(communities) > 1:
                        score = self._evaluate_dense_graph_communities(
                            communities, subgraph, distance_matrix, node_list
                        )

                        if score > best_score:
                            best_score = score
                            best_communities = communities
                            best_n_clusters = n_clusters

                except Exception as e:
                    logger.info(
                        f"Hierarchical clustering failed for n_clusters={n_clusters}: {e}"
                    )
                    continue

            return self._analyze_communities(
                best_communities,
                subgraph,
                f"Hierarchical Clustering (k={best_n_clusters})",
            )

        except ImportError:
            return {"communities": [], "error": "sklearn not available"}
        except Exception as e:
            return {"communities": [], "error": str(e)}

    def _threshold_based_communities(self, nodes: Set[int], subgraph: nx.Graph) -> Dict:
        """Filter edges by weight percentiles and apply community detection"""
        try:
            import networkx.algorithms.community as nx_comm

            # Get all edge weights
            edge_weights = [
                data.get("weight", 0.0) for _, _, data in subgraph.edges(data=True)
            ]

            if not edge_weights:
                return {"communities": [], "error": "No weighted edges found"}

            # Try different weight thresholds
            import numpy as np

            thresholds = [np.percentile(edge_weights, p) for p in [50, 60, 70, 80, 90]]
            best_communities = []
            best_score = -1
            best_threshold = thresholds[0]

            for threshold in thresholds:
                try:
                    # Create filtered subgraph with only high-weight edges
                    filtered_edges = [
                        (u, v, d)
                        for u, v, d in subgraph.edges(data=True)
                        if d.get("weight", 0.0) >= threshold
                    ]

                    if len(filtered_edges) < len(
                        nodes
                    ):  # Not enough edges to be meaningful
                        continue

                    filtered_graph = nx.Graph()
                    filtered_graph.add_nodes_from(subgraph.nodes())
                    filtered_graph.add_edges_from(filtered_edges)

                    # Apply Leiden algorithm (better than Louvain)
                    try:
                        import igraph as ig
                        import leidenalg

                        # Convert to igraph for Leiden
                        g_ig = ig.Graph.TupleList(
                            [(str(u), str(v)) for u, v, _ in filtered_edges],
                            weights=True,
                        )
                        partition = leidenalg.find_partition(
                            g_ig, leidenalg.ModularityVertexPartition
                        )

                        communities = []
                        for community_nodes in partition:
                            community = set(
                                int(g_ig.vs[v]["name"]) for v in community_nodes
                            )
                            if len(community) >= 2:
                                communities.append(community)

                    except ImportError:
                        # Fallback to Louvain if Leiden not available
                        communities = list(
                            nx_comm.louvain_communities(filtered_graph, weight="weight")
                        )
                        communities = [c for c in communities if len(c) >= 2]

                    if len(communities) > 1:
                        # Create dummy adjacency matrix for scoring
                        node_list = list(nodes)
                        adj_matrix = np.zeros((len(node_list), len(node_list)))
                        for i, node_i in enumerate(node_list):
                            for j, node_j in enumerate(node_list):
                                if filtered_graph.has_edge(node_i, node_j):
                                    adj_matrix[i, j] = filtered_graph.get_edge_data(
                                        node_i, node_j
                                    ).get("weight", 0.0)

                        score = self._evaluate_dense_graph_communities(
                            communities, subgraph, adj_matrix, node_list
                        )

                        if score > best_score:
                            best_score = score
                            best_communities = communities
                            best_threshold = threshold

                except Exception as e:
                    logger.info(
                        f"Threshold-based clustering failed for threshold={threshold}: {e}"
                    )
                    continue

            return self._analyze_communities(
                best_communities, subgraph, f"Threshold-based (t={best_threshold:.3f})"
            )

        except Exception as e:
            return {"communities": [], "error": str(e)}

    def _multi_resolution_communities(
        self, nodes: Set[int], subgraph: nx.Graph
    ) -> Dict:
        """Apply Louvain with multiple resolution parameters"""
        try:
            import networkx.algorithms.community as nx_comm

            # Try multiple resolution parameters
            resolutions = [0.5, 1.0, 1.5, 2.0, 3.0]
            best_communities = []
            best_score = -1
            best_resolution = 1.0

            for resolution in resolutions:
                try:
                    communities = list(
                        nx_comm.louvain_communities(
                            subgraph, weight="weight", resolution=resolution, seed=42
                        )
                    )

                    # Filter out single-node communities
                    communities = [c for c in communities if len(c) >= 2]

                    if len(communities) > 1:
                        # Create adjacency matrix for scoring
                        import numpy as np

                        node_list = list(nodes)
                        adj_matrix = np.zeros((len(node_list), len(node_list)))
                        for i, node_i in enumerate(node_list):
                            for j, node_j in enumerate(node_list):
                                if subgraph.has_edge(node_i, node_j):
                                    adj_matrix[i, j] = subgraph.get_edge_data(
                                        node_i, node_j
                                    ).get("weight", 0.0)

                        score = self._evaluate_dense_graph_communities(
                            communities, subgraph, adj_matrix, node_list
                        )

                        if score > best_score:
                            best_score = score
                            best_communities = communities
                            best_resolution = resolution

                except Exception as e:
                    logger.info(
                        f"Multi-resolution failed for resolution={resolution}: {e}"
                    )
                    continue

            return self._analyze_communities(
                best_communities,
                subgraph,
                f"Multi-resolution Louvain (r={best_resolution})",
            )

        except Exception as e:
            return {"communities": [], "error": str(e)}

    def _evaluate_dense_graph_communities(
        self, communities, subgraph: nx.Graph, adj_matrix: np.ndarray, node_list: list
    ) -> float:
        """Evaluate community quality for dense graphs using intra vs inter-community weights"""
        if not communities or len(communities) <= 1:
            return -1.0

        try:
            # Create node to community mapping
            node_to_community = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_to_community[node] = i

            intra_weights = []
            inter_weights = []

            # Calculate intra and inter-community weights
            for i, node_i in enumerate(node_list):
                for j, node_j in enumerate(node_list[i + 1 :], start=i + 1):
                    weight = adj_matrix[i, j]

                    if node_i in node_to_community and node_j in node_to_community:
                        if node_to_community[node_i] == node_to_community[node_j]:
                            intra_weights.append(weight)
                        else:
                            inter_weights.append(weight)

            if not intra_weights or not inter_weights:
                return -1.0

            # Calculate modularity-like score: higher intra-community weights vs lower inter-community weights
            avg_intra = np.mean(intra_weights)
            avg_inter = np.mean(inter_weights)

            # Penalize extreme imbalances in community sizes
            sizes = [len(c) for c in communities]
            size_variance = np.var(sizes)
            size_penalty = min(1.0, size_variance / (np.mean(sizes) ** 2))

            # Score combines weight separation and size balance
            separation_score = (avg_intra - avg_inter) / max(
                avg_intra + avg_inter, 0.001
            )
            community_count_bonus = min(
                1.0, len(communities) / 5.0
            )  # Bonus for having multiple communities

            final_score = (
                separation_score
                * (1.0 - size_penalty * 0.3)
                * (1.0 + community_count_bonus * 0.2)
            )

            return float(final_score)

        except Exception as e:
            logger.info(f"Community evaluation failed: {e}")
            return -1.0

    def _select_best_communities_for_dense_graphs(
        self, communities_results: Dict, is_dense_graph: bool
    ) -> str:
        """Select best community detection method optimized for dense graphs"""
        if not communities_results:
            return "spectral"  # Default fallback

        best_method = None
        best_score = -1.0

        # Priority order for dense graphs
        if is_dense_graph:
            method_priorities = [
                "spectral",
                "hierarchical",
                "threshold_based",
                "multi_resolution",
            ]
        else:
            method_priorities = [
                "louvain",
                "greedy_modularity",
                "multi_resolution",
                "spectral",
            ]

        for method in method_priorities:
            if method in communities_results:
                result = communities_results[method]

                if "error" in result or not result.get("communities"):
                    continue

                communities = result.get("communities", [])
                num_communities = len(communities)

                if num_communities <= 1:
                    continue

                # For dense graphs, prioritize methods that find multiple meaningful communities
                if is_dense_graph:
                    # Custom scoring for dense graphs
                    avg_community_size = (
                        np.mean([c["size"] for c in communities]) if communities else 0
                    )
                    size_balance = 1.0 - (
                        np.var([c["size"] for c in communities])
                        / max(avg_community_size**2, 1)
                    )

                    # Prefer methods that find 2-6 communities with good size balance
                    community_count_score = (
                        min(1.0, num_communities / 6.0) if num_communities <= 6 else 0.5
                    )

                    avg_density = (
                        np.mean([c["density"] for c in communities])
                        if communities
                        else 0
                    )
                    avg_cohesion = (
                        np.mean([c.get("cohesion_ratio", 0) for c in communities])
                        if communities
                        else 0
                    )

                    score = (
                        community_count_score * 0.4
                        + size_balance * 0.3
                        + avg_density * 0.2
                        + avg_cohesion * 0.1
                    )
                else:
                    # Traditional scoring for sparse graphs
                    modularity = result.get("modularity", 0.0)
                    score = (
                        modularity * 0.7
                        + (min(num_communities, 5) / 5.0) * 0.2
                        + (min(avg_community_size, 10) / 10.0) * 0.1
                    )

                if score > best_score:
                    best_score = score
                    best_method = method

        return best_method if best_method else list(communities_results.keys())[0]

    def _select_best_communities(self, communities_results: Dict) -> str:
        """Select best community detection method based on modularity and other metrics"""
        best_method = "louvain"
        best_score = -1.0

        for method, result in communities_results.items():
            if "error" in result or not result.get("communities"):
                continue

            modularity = result.get("modularity", 0.0)
            num_communities = result.get("num_communities", 0)
            avg_size = result.get("avg_community_size", 0.0)

            score = (
                modularity * 0.7
                + (min(num_communities, 5) / 5.0) * 0.2
                + (min(avg_size, 10) / 10.0) * 0.1
            )

            if score > best_score:
                best_score = score
                best_method = method

        return best_method

    def _summarize_communities(self, community_result: Dict) -> str:
        """Generate summary of community detection results"""
        if not community_result or "error" in community_result:
            return "Community detection failed"

        communities = community_result.get("communities", [])
        if not communities:
            return "No communities detected"

        method = community_result.get("method", "Unknown")
        modularity = community_result.get("modularity", 0.0)

        sizes = [c["size"] for c in communities]
        avg_cohesion = np.mean([c["cohesion_ratio"] for c in communities])

        return f"{method}: {len(communities)} communities (sizes: {sizes}), modularity: {modularity:.3f}, avg cohesion: {avg_cohesion:.3f}"

    def find_maximum_weight_cycle(self, nodes: Set[int], graph: nx.Graph) -> Dict:
        """Find maximum weight Hamiltonian cycle covering all nodes"""
        if len(nodes) < 3:
            return {
                "cycle": [],
                "weight": 0.0,
                "summary": "Insufficient nodes for cycle detection",
            }

        subgraph = graph.subgraph(nodes)
        node_list = list(nodes)
        n = len(node_list)

        if n <= 10:
            return self._tsp_hamiltonian_cycle(node_list, subgraph)

        return self._greedy_hamiltonian_cycle(node_list, subgraph)

    def _tsp_hamiltonian_cycle(self, nodes: List[int], subgraph: nx.Graph) -> Dict:
        """Solve TSP exactly for small graphs"""
        n = len(nodes)
        max_weight = 0.0
        best_cycle = []

        for perm in itertools.permutations(nodes[1:]):
            cycle = [nodes[0]] + list(perm)
            weight = 0.0
            valid = True

            for i in range(n):
                u, v = cycle[i], cycle[(i + 1) % n]
                if subgraph.has_edge(u, v):
                    weight += subgraph.get_edge_data(u, v).get("weight", 0.0)
                else:
                    valid = False
                    break

            if valid and weight > max_weight:
                max_weight = weight
                best_cycle = cycle

        return {
            "maximum_cycle": {
                "nodes": best_cycle,
                "weight": max_weight,
                "length": len(best_cycle),
            },
            "summary": (
                f"Hamiltonian cycle: {len(best_cycle)} nodes, weight: {max_weight:.3f}"
                if best_cycle
                else "No Hamiltonian cycle found"
            ),
        }

    def _greedy_hamiltonian_cycle(self, nodes: List[int], subgraph: nx.Graph) -> Dict:
        """Greedy approximation for larger graphs"""
        if not nodes:
            return {
                "maximum_cycle": {"nodes": [], "weight": 0.0, "length": 0},
                "summary": "No nodes",
            }

        current = nodes[0]
        cycle = [current]
        remaining = set(nodes[1:])
        total_weight = 0.0

        while remaining:
            best_next = None
            best_weight = -1

            for neighbor in remaining:
                if subgraph.has_edge(current, neighbor):
                    weight = subgraph.get_edge_data(current, neighbor).get(
                        "weight", 0.0
                    )
                    if weight > best_weight:
                        best_weight = weight
                        best_next = neighbor

            if best_next is None:
                best_next = next(iter(remaining))
                best_weight = 0.0

            cycle.append(best_next)
            remaining.remove(best_next)
            total_weight += best_weight
            current = best_next

        if subgraph.has_edge(cycle[-1], cycle[0]):
            total_weight += subgraph.get_edge_data(cycle[-1], cycle[0]).get(
                "weight", 0.0
            )

        return {
            "maximum_cycle": {
                "nodes": cycle,
                "weight": total_weight,
                "length": len(cycle),
            },
            "summary": f"Greedy Hamiltonian cycle: {len(cycle)} nodes, weight: {total_weight:.3f}",
        }

    def compute_stress_layout(self, nodes: Set[int], graph: nx.Graph) -> Dict:
        """Compute stress/MDS layout for subgraph visualization"""
        if len(nodes) < 2:
            return {"coordinates": {}, "summary": "Insufficient nodes for layout"}

        node_list = list(nodes)
        subgraph = graph.subgraph(nodes)

        try:

            n = len(node_list)
            distance_matrix = np.full((n, n), np.inf)

            np.fill_diagonal(distance_matrix, 0)

            all_weights = []
            for i, node_i in enumerate(node_list):
                for j, node_j in enumerate(node_list):
                    if i != j and subgraph.has_edge(node_i, node_j):
                        edge_weight = subgraph.get_edge_data(node_i, node_j).get(
                            "weight", 0.01
                        )
                        all_weights.append(edge_weight)

            if all_weights:
                min_weight = min(all_weights)
                max_weight = max(all_weights)
                weight_range = max_weight - min_weight

                for i, node_i in enumerate(node_list):
                    for j, node_j in enumerate(node_list):
                        if i != j and subgraph.has_edge(node_i, node_j):
                            edge_weight = subgraph.get_edge_data(node_i, node_j).get(
                                "weight", 0.01
                            )

                            if weight_range > 0:
                                normalized_weight = (
                                    edge_weight - min_weight
                                ) / weight_range

                                distance_matrix[i, j] = 2.0 - (1.9 * normalized_weight)
                            else:

                                distance_matrix[i, j] = 1.0

            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        distance_matrix[i, j] = min(
                            distance_matrix[i, j],
                            distance_matrix[i, k] + distance_matrix[k, j],
                        )

            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
            coordinates = mds.fit_transform(distance_matrix)

            if coordinates.shape[0] > 1:
                coords_min = coordinates.min(axis=0)
                coords_max = coordinates.max(axis=0)
                coords_range = coords_max - coords_min
                coords_range[coords_range == 0] = 1
                coordinates = (coordinates - coords_min) / coords_range

            layout_coords = {}
            for i, node in enumerate(node_list):
                layout_coords[int(node)] = {
                    "x": float(coordinates[i, 0]),
                    "y": float(coordinates[i, 1]),
                }

            return {
                "coordinates": layout_coords,
                "stress": float(mds.stress_) if hasattr(mds, "stress_") else 0.0,
                "summary": (
                    f"MDS layout computed for {len(node_list)} nodes, stress: {mds.stress_:.3f}"
                    if hasattr(mds, "stress_")
                    else f"Layout computed for {len(node_list)} nodes"
                ),
            }

        except Exception as e:
            logger.info(f"MDS layout failed: {e}")

            fallback_coords = {}
            for i, node in enumerate(node_list):
                angle = 2 * np.pi * i / len(node_list)
                fallback_coords[int(node)] = {
                    "x": float(0.5 + 0.3 * np.cos(angle)),
                    "y": float(0.5 + 0.3 * np.sin(angle)),
                }

            return {
                "coordinates": fallback_coords,
                "error": str(e),
                "summary": f"Fallback circular layout for {len(node_list)} nodes",
            }

    def analyze_hybrid_centroids(
        self,
        nodes: Set[int],
        feature_embeddings: Dict[str, np.ndarray],
        matrix_builder,
        tuned_w_s: Dict[str, float],
        tuned_w_c: Dict[str, float],
        df: pd.DataFrame,
    ) -> Dict:
        """Combine embedding similarity and complementarity analysis"""
        if not nodes or len(nodes) < 2:
            return {
                "hybrid_patterns": {},
                "summary": "Insufficient nodes for hybrid analysis",
            }

        # Create mapping from node IDs to DataFrame positions for embedding indexing
        node_to_pos = {node_id: pos for pos, node_id in enumerate(df.index)}
        node_list = list(nodes)
        feature_categories = [
            "role",
            "experience",
            "persona",
            "industry",
            "market",
            "offering",
        ]
        hybrid_patterns = {}

        for category in feature_categories:
            if category not in feature_embeddings:
                continue

            valid_positions = [
                node_to_pos[node] for node in node_list if node in node_to_pos
            ]
            category_embeddings = feature_embeddings[category][valid_positions]

            embedding_similarities = []
            complementarity_scores = []
            hybrid_scores = []

            for i, node_i in enumerate(node_list):
                for j, node_j in enumerate(node_list[i + 1 :], start=i + 1):

                    emb_i = category_embeddings[i]
                    emb_j = category_embeddings[j]

                    norm_i = np.linalg.norm(emb_i)
                    norm_j = np.linalg.norm(emb_j)
                    if norm_i > 0 and norm_j > 0:
                        sim_score = np.dot(emb_i, emb_j) / (norm_i * norm_j)
                    else:
                        sim_score = 0.0

                    comp_scores = matrix_builder.get_all_complementarities(
                        node_i, node_j
                    )
                    comp_score = comp_scores.get(category, 0.5)

                    w_s = tuned_w_s.get(category, 1.0)
                    w_c = tuned_w_c.get(category, 1.0)

                    hybrid_score = (w_s * sim_score + w_c * comp_score) / (w_s + w_c)

                    embedding_similarities.append(sim_score)
                    complementarity_scores.append(comp_score)
                    hybrid_scores.append(hybrid_score)

            if hybrid_scores:
                hybrid_patterns[category] = {
                    "avg_embedding_similarity": float(np.mean(embedding_similarities)),
                    "avg_complementarity": float(np.mean(complementarity_scores)),
                    "avg_hybrid_score": float(np.mean(hybrid_scores)),
                    "similarity_weight": tuned_w_s.get(category, 1.0),
                    "complementarity_weight": tuned_w_c.get(category, 1.0),
                    "balance_ratio": float(np.mean(embedding_similarities))
                    / max(float(np.mean(complementarity_scores)), 0.01),
                    "total_pairs": len(hybrid_scores),
                }

        overall_scores = [
            pattern["avg_hybrid_score"] for pattern in hybrid_patterns.values()
        ]
        overall_hybrid_strength = (
            float(np.mean(overall_scores)) if overall_scores else 0.0
        )

        most_balanced_category = None
        best_balance_score = float("inf")

        for category, pattern in hybrid_patterns.items():
            balance_deviation = abs(1.0 - pattern["balance_ratio"])
            if balance_deviation < best_balance_score:
                best_balance_score = balance_deviation
                most_balanced_category = category

        return {
            "hybrid_patterns": hybrid_patterns,
            "overall_hybrid_strength": overall_hybrid_strength,
            "most_balanced_category": most_balanced_category,
            "summary": f"Hybrid analysis: {len(hybrid_patterns)} features, overall strength: {overall_hybrid_strength:.3f}, most balanced: {most_balanced_category}",
        }

    def analyze_feature_importance(
        self,
        nodes: Set[int],
        graph: nx.Graph,
        matrix_builder,
        tuned_w_s: Dict[str, float],
        tuned_w_c: Dict[str, float],
    ) -> Dict:
        """Analyze which features contribute most to subgraph density based on tuned parameters"""
        if not nodes or len(nodes) < 2:
            return {
                "feature_contributions": {},
                "summary": "Insufficient nodes for feature importance",
            }

        node_list = list(nodes)
        subgraph = graph.subgraph(nodes)
        feature_categories = [
            "role",
            "experience",
            "persona",
            "industry",
            "market",
            "offering",
        ]

        feature_contributions = {}
        total_subgraph_weight = sum(
            data.get("weight", 0.0) for _, _, data in subgraph.edges(data=True)
        )

        for category in feature_categories:
            similarity_weight = tuned_w_s.get(category, 1.0)
            complementarity_weight = tuned_w_c.get(category, 1.0)
            total_weight = similarity_weight + complementarity_weight

            category_sim_scores = []
            category_comp_scores = []
            edge_contributions = []

            for i, node_i in enumerate(node_list):
                for j, node_j in enumerate(node_list[i + 1 :], start=i + 1):
                    if subgraph.has_edge(node_i, node_j):
                        comp_scores = matrix_builder.get_all_complementarities(
                            node_i, node_j
                        )
                        comp_score = comp_scores.get(category, 0.5)
                        category_comp_scores.append(comp_score)

                        sim_score = 1.0 - comp_score + 0.1
                        category_sim_scores.append(sim_score)

                        feature_contribution = (
                            similarity_weight * sim_score
                            + complementarity_weight * comp_score
                        ) / total_weight
                        edge_data = subgraph.get_edge_data(node_i, node_j)
                        edge_weight = edge_data.get("weight", 0.0)
                        edge_contributions.append(feature_contribution * edge_weight)

            if edge_contributions:
                total_feature_contribution = sum(edge_contributions)
                avg_similarity = np.mean(category_sim_scores)
                avg_complementarity = np.mean(category_comp_scores)

                feature_contributions[category] = {
                    "total_contribution": total_feature_contribution,
                    "contribution_percentage": (
                        total_feature_contribution / max(total_subgraph_weight, 0.001)
                    )
                    * 100,
                    "avg_similarity": avg_similarity,
                    "avg_complementarity": avg_complementarity,
                    "similarity_weight": similarity_weight,
                    "complementarity_weight": complementarity_weight,
                    "similarity_dominance": similarity_weight
                    / max(total_weight, 0.001),
                    "complementarity_dominance": complementarity_weight
                    / max(total_weight, 0.001),
                    "edges_analyzed": len(edge_contributions),
                }

        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: x[1]["contribution_percentage"],
            reverse=True,
        )

        sim_dominant_features = []
        comp_dominant_features = []

        for feature, contrib in feature_contributions.items():
            if contrib["similarity_dominance"] > contrib["complementarity_dominance"]:
                sim_dominant_features.append(
                    {
                        "feature": feature,
                        "dominance_ratio": contrib["similarity_dominance"]
                        / max(contrib["complementarity_dominance"], 0.001),
                        "contribution": contrib["contribution_percentage"],
                    }
                )
            else:
                comp_dominant_features.append(
                    {
                        "feature": feature,
                        "dominance_ratio": contrib["complementarity_dominance"]
                        / max(contrib["similarity_dominance"], 0.001),
                        "contribution": contrib["contribution_percentage"],
                    }
                )

        return {
            "feature_contributions": feature_contributions,
            "ranked_features": [f[0] for f in sorted_features],
            "top_contributor": sorted_features[0][0] if sorted_features else None,
            "similarity_dominant_features": sim_dominant_features,
            "complementarity_dominant_features": comp_dominant_features,
            "density_drivers": {
                "similarity_driven": len(sim_dominant_features)
                > len(comp_dominant_features),
                "primary_mechanism": (
                    "similarity"
                    if len(sim_dominant_features) > len(comp_dominant_features)
                    else "complementarity"
                ),
                "balance_score": abs(
                    len(sim_dominant_features) - len(comp_dominant_features)
                )
                / max(len(feature_categories), 1),
            },
            "summary": f"Top contributor: {sorted_features[0][0] if sorted_features else 'None'} ({sorted_features[0][1]['contribution_percentage']:.1f}% of density), {len(sim_dominant_features)} sim-driven, {len(comp_dominant_features)} comp-driven features",
        }

    def analyze_dataset_values(
        self, nodes: Set[int], df: pd.DataFrame, matrix_builder
    ) -> Dict:
        """Extract and analyze actual dataset values for subgraph members"""
        if not nodes:
            return {"member_profiles": [], "summary": "No nodes to analyze"}

        node_list = list(nodes)
        member_profiles = []

        feature_columns = {
            "role": "Professional Identity - Role Specification",
            "experience": "Professional Identity - Experience Level",
            "persona": "All Persona Titles",
            "industry": "Company Identity - Industry Classification",
            "market": "Company Market - Market Traction",
            "offering": "Company Offering - Value Proposition",
        }

        for node_idx in node_list:
            person_data = df.loc[node_idx]
            profile = {
                "node_id": int(node_idx),
                "name": person_data["Person Name"],
                "company": person_data.get("Person Company", ""),
                "linkedin": person_data.get("LinkedIn URL", ""),
                "features": {},
            }

            for feature_key, column_name in feature_columns.items():
                raw_value = person_data.get(column_name, "")
                profile["features"][feature_key] = {
                    "raw_value": str(raw_value) if pd.notna(raw_value) else "",
                    "tags": self._extract_feature_tags(
                        str(raw_value) if pd.notna(raw_value) else ""
                    ),
                    "length": len(str(raw_value)) if pd.notna(raw_value) else 0,
                }

            member_profiles.append(profile)

        common_patterns = self._find_common_patterns(member_profiles)
        diversity_analysis = self._analyze_diversity(member_profiles)

        return {
            "member_profiles": member_profiles,
            "group_size": len(member_profiles),
            "common_patterns": common_patterns,
            "diversity_analysis": diversity_analysis,
            "summary": f"Analyzed {len(member_profiles)} members, {len(common_patterns.get('common_keywords', []))} common keywords, diversity score: {diversity_analysis.get('overall_diversity', 0.0):.2f}",
        }

    def _extract_feature_tags(self, value: str) -> List[str]:
        """Extract meaningful tags from feature values"""
        if not value or value.strip() == "":
            return []

        delimiters = ["|", ",", ";", "\n", " and ", " & "]
        tags = [value]

        for delimiter in delimiters:
            new_tags = []
            for tag in tags:
                new_tags.extend([t.strip() for t in tag.split(delimiter) if t.strip()])
            tags = new_tags

        clean_tags = []
        for tag in tags:
            tag = tag.strip().lower()
            if len(tag) > 2 and tag not in [
                "the",
                "and",
                "or",
                "in",
                "at",
                "of",
                "to",
                "for",
            ]:
                clean_tags.append(tag)

        return clean_tags[:10]

    def _find_common_patterns(self, profiles: List[Dict]) -> Dict:
        """Find common patterns across member profiles"""
        if not profiles:
            return {"common_keywords": [], "shared_features": {}}

        keyword_counts = defaultdict(int)
        feature_value_counts = {}

        for profile in profiles:
            profile_keywords = set()

            for feature_key, feature_data in profile["features"].items():
                if feature_key not in feature_value_counts:
                    feature_value_counts[feature_key] = defaultdict(int)

                for tag in feature_data["tags"]:
                    profile_keywords.add(tag)
                    feature_value_counts[feature_key][tag] += 1

            for keyword in profile_keywords:
                keyword_counts[keyword] += 1

        min_threshold = max(1, len(profiles) * 0.5)
        common_keywords = [
            keyword
            for keyword, count in keyword_counts.items()
            if count >= min_threshold
        ]
        common_keywords.sort(key=lambda x: keyword_counts[x], reverse=True)

        shared_features = {}
        for feature_key, value_counts in feature_value_counts.items():
            top_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            shared_features[feature_key] = [
                {
                    "value": value,
                    "count": count,
                    "percentage": (count / len(profiles)) * 100,
                }
                for value, count in top_values
                if count > 1
            ]

        return {
            "common_keywords": common_keywords[:20],
            "shared_features": shared_features,
            "keyword_frequencies": dict(
                sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            ),
        }

    def _analyze_diversity(self, profiles: List[Dict]) -> Dict:
        """Analyze diversity within the subgroup"""
        if len(profiles) < 2:
            return {"overall_diversity": 0.0, "feature_diversity": {}}

        feature_diversity = {}

        for feature_key in [
            "role",
            "experience",
            "persona",
            "industry",
            "market",
            "offering",
        ]:

            all_tags = []
            profile_tag_sets = []

            for profile in profiles:
                feature_data = profile["features"].get(feature_key, {})
                tags = set(feature_data.get("tags", []))
                all_tags.extend(tags)
                profile_tag_sets.append(tags)

            if all_tags:
                unique_tags = len(set(all_tags))
                total_tags = len(all_tags)

                jaccard_distances = []
                for i in range(len(profile_tag_sets)):
                    for j in range(i + 1, len(profile_tag_sets)):
                        set_i, set_j = profile_tag_sets[i], profile_tag_sets[j]
                        if len(set_i.union(set_j)) > 0:
                            jaccard_sim = len(set_i.intersection(set_j)) / len(
                                set_i.union(set_j)
                            )
                            jaccard_distances.append(1.0 - jaccard_sim)
                        else:
                            jaccard_distances.append(1.0)

                avg_diversity = np.mean(jaccard_distances) if jaccard_distances else 0.0

                feature_diversity[feature_key] = {
                    "unique_tags": unique_tags,
                    "total_mentions": total_tags,
                    "uniqueness_ratio": unique_tags / max(total_tags, 1),
                    "avg_pairwise_diversity": avg_diversity,
                    "diversity_score": (unique_tags / max(total_tags, 1))
                    * avg_diversity,
                }

        diversity_scores = [fd["diversity_score"] for fd in feature_diversity.values()]
        overall_diversity = np.mean(diversity_scores) if diversity_scores else 0.0

        return {
            "overall_diversity": overall_diversity,
            "feature_diversity": feature_diversity,
            "most_diverse_feature": (
                max(
                    feature_diversity.keys(),
                    key=lambda k: feature_diversity[k]["diversity_score"],
                )
                if feature_diversity
                else None
            ),
            "least_diverse_feature": (
                min(
                    feature_diversity.keys(),
                    key=lambda k: feature_diversity[k]["diversity_score"],
                )
                if feature_diversity
                else None
            ),
        }

    def get_optimal_nodes_for_replication(
        self,
        df: pd.DataFrame,
        matrix_builder,
        feature_embeddings: Dict[str, np.ndarray],
        tuned_w_s: Dict[str, float],
        tuned_w_c: Dict[str, float],
        top_k: int = 5,
    ) -> Dict:
        """
        Greedy approach: Find the top node in each of the 6 matrices (sim/comp for each feature).
        Returns optimal feature profile composed of best values from each matrix.
        """
        if not tuned_w_s or not tuned_w_c:
            return {
                "error": "No tuned weights provided",
                "optimal_feature_profile": {},
                "matrix_champions": [],
            }

        feature_categories = [
            "role",
            "experience",
            "persona",
            "industry",
            "market",
            "offering",
        ]

        matrix_selections = self._determine_optimal_matrix_selection(
            tuned_w_s, tuned_w_c
        )

        matrix_champions = []
        optimal_feature_values = {}

        for feature in feature_categories:
            matrix_type = matrix_selections[feature]["selected_matrix"]
            weight_value = matrix_selections[feature]["weight_value"]

            champion_node = self._find_matrix_champion(
                df,
                matrix_builder,
                feature_embeddings,
                feature,
                matrix_type,
                weight_value,
            )

            if champion_node:
                matrix_champions.append(champion_node)
                optimal_feature_values[feature] = champion_node["feature_value"]

        optimal_profile_summary = self._create_optimal_profile_summary(
            optimal_feature_values, matrix_champions, matrix_selections
        )

        return {
            "matrix_selections": matrix_selections,
            "matrix_champions": matrix_champions,
            "optimal_feature_profile": optimal_feature_values,
            "profile_summary": optimal_profile_summary,
            "strategy_explanation": self._explain_greedy_strategy(
                matrix_selections, matrix_champions
            ),
        }

    def _analyze_feature_priorities(
        self, tuned_w_s: Dict[str, float], tuned_w_c: Dict[str, float]
    ) -> Dict:
        """Analyze which features have highest impact based on weight magnitudes"""
        feature_priorities = []

        for feature in [
            "role",
            "experience",
            "persona",
            "industry",
            "market",
            "offering",
        ]:
            w_s = tuned_w_s.get(feature, 1.0)
            w_c = tuned_w_c.get(feature, 1.0)

            # Determine which is dominant and by how much
            total_weight = w_s + w_c
            similarity_dominance = w_s / total_weight
            complementarity_dominance = w_c / total_weight

            # Overall impact is total magnitude of weights
            impact_magnitude = total_weight

            # Determine strategy: similarity or complementarity focused
            if w_c > w_s:
                dominant_type = "complementarity"
                dominance_ratio = w_c / max(w_s, 0.1)
            else:
                dominant_type = "similarity"
                dominance_ratio = w_s / max(w_c, 0.1)

            feature_priorities.append(
                {
                    "feature": feature,
                    "impact_magnitude": impact_magnitude,
                    "dominant_type": dominant_type,
                    "dominance_ratio": dominance_ratio,
                    "similarity_weight": w_s,
                    "complementarity_weight": w_c,
                    "similarity_dominance": similarity_dominance,
                    "complementarity_dominance": complementarity_dominance,
                }
            )

        # Sort by impact magnitude (total weight importance)
        feature_priorities.sort(key=lambda x: x["impact_magnitude"], reverse=True)

        return {
            "priority_features": feature_priorities,
            "weight_summary": {
                "highest_impact_feature": feature_priorities[0]["feature"],
                "complementarity_focused_features": [
                    f["feature"]
                    for f in feature_priorities
                    if f["dominant_type"] == "complementarity"
                ],
                "similarity_focused_features": [
                    f["feature"]
                    for f in feature_priorities
                    if f["dominant_type"] == "similarity"
                ],
            },
        }

    def _find_highest_weighted_nodes_in_feature(
        self,
        df: pd.DataFrame,
        matrix_builder,
        feature_embeddings: Dict[str, np.ndarray],
        feature: str,
        dominant_type: str,  # "similarity" or "complementarity"
        tuned_w_s: Dict[str, float],
        tuned_w_c: Dict[str, float],
        top_k: int = 5,
    ) -> List[Dict]:
        """Find nodes with highest average weighted connections in a specific feature"""

        node_scores = []

        for node_idx in df.index:
            # Calculate average weighted connection strength for this node
            total_weighted_score = 0.0
            connection_count = 0

            for other_idx in df.index:
                if node_idx == other_idx:
                    continue

                if dominant_type == "complementarity":
                    # Use complementarity matrix
                    comp_scores = matrix_builder.get_all_complementarities(
                        node_idx, other_idx
                    )
                    feature_score = comp_scores.get(feature, 0.5)
                    weight = tuned_w_c.get(feature, 1.0)
                else:
                    # Use similarity from embeddings
                    if feature in feature_embeddings:
                        embeddings = feature_embeddings[feature]
                        node_to_pos = {
                            node_id: pos for pos, node_id in enumerate(df.index)
                        }
                        emb_i = (
                            embeddings[node_to_pos[node_idx]]
                            if node_idx in node_to_pos
                            else None
                        )
                        emb_j = (
                            embeddings[node_to_pos[other_idx]]
                            if other_idx in node_to_pos
                            else None
                        )

                        if emb_i is None or emb_j is None:
                            continue

                        norm_i = np.linalg.norm(emb_i)
                        norm_j = np.linalg.norm(emb_j)
                        if norm_i > 0 and norm_j > 0:
                            feature_score = np.dot(emb_i, emb_j) / (norm_i * norm_j)
                        else:
                            feature_score = 0.0
                    else:
                        feature_score = 0.0
                    weight = tuned_w_s.get(feature, 1.0)

                total_weighted_score += feature_score * weight
                connection_count += 1

            avg_weighted_score = total_weighted_score / max(connection_count, 1)

            # Get person details
            person_data = df.loc[node_idx]

            node_scores.append(
                {
                    "node_idx": int(node_idx),
                    "name": person_data.get("Person Name", "Unknown"),
                    "company": person_data.get("Person Company", ""),
                    "feature": feature,
                    "dominant_type": dominant_type,
                    "avg_weighted_score": float(avg_weighted_score),
                    "feature_weight": weight,
                    "feature_profile": self._get_feature_profile(person_data, feature),
                }
            )

        # Sort by average weighted score
        node_scores.sort(key=lambda x: x["avg_weighted_score"], reverse=True)
        return node_scores[:top_k]

    def _get_feature_profile(self, person_data: pd.Series, feature: str) -> str:
        """Extract the feature profile for a person"""
        if feature == "role":
            return str(
                person_data.get("Professional Identity - Role Specification", "")
            ).strip()
        elif feature == "experience":
            return str(
                person_data.get("Professional Identity - Experience Level", "")
            ).strip()
        elif feature == "persona":
            return str(person_data.get("All Persona Titles", "")).strip()
        elif feature == "industry":
            return str(
                person_data.get("Company Identity - Industry Classification", "")
            ).strip()
        elif feature == "market":
            return str(person_data.get("Company Market - Market Traction", "")).strip()
        elif feature == "offering":
            return str(
                person_data.get("Company Offering - Value Proposition", "")
            ).strip()
        return ""

    def _rank_replication_candidates(
        self,
        candidates: List[Dict],
        tuned_w_s: Dict[str, float],
        tuned_w_c: Dict[str, float],
    ) -> List[Dict]:
        """Rank replication candidates by overall impact potential"""

        # Calculate composite impact score for each candidate
        for candidate in candidates:
            feature = candidate["feature"]
            dominant_type = candidate["dominant_type"]
            avg_score = candidate["avg_weighted_score"]

            # Weight by the feature's overall importance (sum of w_s + w_c)
            total_feature_weight = tuned_w_s.get(feature, 1.0) + tuned_w_c.get(
                feature, 1.0
            )

            # Higher scores for dominant strategy alignment
            strategy_bonus = 1.2 if dominant_type == "complementarity" else 1.0

            composite_score = avg_score * total_feature_weight * strategy_bonus
            candidate["composite_impact_score"] = float(composite_score)

        # Sort by composite impact score
        candidates.sort(key=lambda x: x["composite_impact_score"], reverse=True)
        return candidates

    def _generate_replication_strategy(
        self, priority_analysis: Dict, top_candidates: List[Dict]
    ) -> str:
        """Generate human-readable strategy summary"""

        if not top_candidates:
            return "No suitable replication candidates found"

        priority_features = priority_analysis["priority_features"][:3]
        top_candidate = top_candidates[0]

        strategy_lines = [
            f"REPLICATION STRATEGY BASED ON RUNTIME WEIGHTS:",
            f"",
            f"Top Priority Features:",
        ]

        for i, feature_info in enumerate(priority_features, 1):
            strategy_lines.append(
                f"{i}. {feature_info['feature'].upper()}: {feature_info['dominant_type']} focus "
                f"(impact: {feature_info['impact_magnitude']:.2f})"
            )

        strategy_lines.extend(
            [
                f"",
                f"RECOMMENDED NODE TO REPLICATE:",
                f"• {top_candidate['name']} ({top_candidate['company']})",
                f"• Strongest in: {top_candidate['feature']} ({top_candidate['dominant_type']})",
                f"• Impact Score: {top_candidate['composite_impact_score']:.3f}",
                f"• Profile: {top_candidate['feature_profile'][:100]}...",
                f"",
                f"REPLICATION IMPACT:",
                f"• Replicating this node will {'increase complementarity' if top_candidate['dominant_type'] == 'complementarity' else 'increase similarity'}",
                f"• Focus on the {top_candidate['feature']} dimension",
                f"• Expected density improvement in weighted subgraphs",
            ]
        )

        return "\n".join(strategy_lines)

    def _determine_optimal_matrix_selection(
        self, tuned_w_s: Dict[str, float], tuned_w_c: Dict[str, float]
    ) -> Dict[str, Dict]:
        """
        For each feature, determine whether to use similarity or complementarity matrix
        based on which weight is higher.
        """
        matrix_selections = {}

        for feature in [
            "role",
            "experience",
            "persona",
            "industry",
            "market",
            "offering",
        ]:
            w_s = tuned_w_s.get(feature, 1.0)
            w_c = tuned_w_c.get(feature, 1.0)

            if w_c > w_s:
                selected_matrix = "complementarity"
                weight_value = w_c
                selection_reason = f"Complementarity weight ({w_c:.2f}) > Similarity weight ({w_s:.2f})"
            else:
                selected_matrix = "similarity"
                weight_value = w_s
                selection_reason = f"Similarity weight ({w_s:.2f}) >= Complementarity weight ({w_c:.2f})"

            matrix_selections[feature] = {
                "selected_matrix": selected_matrix,
                "weight_value": weight_value,
                "similarity_weight": w_s,
                "complementarity_weight": w_c,
                "selection_reason": selection_reason,
            }

        return matrix_selections

    def _find_matrix_champion(
        self,
        df: pd.DataFrame,
        matrix_builder,
        feature_embeddings: Dict[str, np.ndarray],
        feature: str,
        matrix_type: str,  # "similarity" or "complementarity"
        weight_value: float,
    ) -> Dict:
        """
        Find the node with the highest weighted average connections in a specific matrix.
        """
        node_scores = []

        for node_idx in df.index:
            total_weighted_score = 0.0
            connection_count = 0

            for other_idx in df.index:
                if node_idx == other_idx:
                    continue

                if matrix_type == "complementarity":
                    # Use complementarity matrix
                    comp_scores = matrix_builder.get_all_complementarities(
                        node_idx, other_idx
                    )
                    feature_score = comp_scores.get(feature, 0.5)
                else:
                    # Use similarity from embeddings
                    if feature in feature_embeddings:
                        embeddings = feature_embeddings[feature]
                        node_to_pos = {
                            node_id: pos for pos, node_id in enumerate(df.index)
                        }
                        emb_i = (
                            embeddings[node_to_pos[node_idx]]
                            if node_idx in node_to_pos
                            else None
                        )
                        emb_j = (
                            embeddings[node_to_pos[other_idx]]
                            if other_idx in node_to_pos
                            else None
                        )

                        if emb_i is None or emb_j is None:
                            continue

                        norm_i = np.linalg.norm(emb_i)
                        norm_j = np.linalg.norm(emb_j)
                        if norm_i > 0 and norm_j > 0:
                            feature_score = np.dot(emb_i, emb_j) / (norm_i * norm_j)
                        else:
                            feature_score = 0.0
                    else:
                        feature_score = 0.0

                total_weighted_score += feature_score * weight_value
                connection_count += 1

            avg_weighted_score = total_weighted_score / max(connection_count, 1)

            # Get person details and feature value
            person_data = df.loc[node_idx]
            feature_value = self._get_feature_profile(person_data, feature)

            node_scores.append(
                {
                    "node_idx": int(node_idx),
                    "name": person_data.get("Person Name", "Unknown"),
                    "company": person_data.get("Person Company", ""),
                    "feature": feature,
                    "matrix_type": matrix_type,
                    "avg_weighted_score": float(avg_weighted_score),
                    "weight_used": weight_value,
                    "feature_value": feature_value,
                }
            )

        # Return the champion (highest scoring node)
        if node_scores:
            champion = max(node_scores, key=lambda x: x["avg_weighted_score"])
            return champion
        return None

    def _create_optimal_profile_summary(
        self,
        optimal_feature_values: Dict[str, str],
        matrix_champions: List[Dict],
        matrix_selections: Dict[str, Dict],
    ) -> str:
        """
        Create a human-readable summary of the optimal feature profile.
        """
        if not optimal_feature_values:
            return "No optimal profile could be generated"

        summary_lines = [
            "OPTIMAL FEATURE PROFILE (Greedy Matrix Selection):",
            "=" * 55,
        ]

        for feature in [
            "role",
            "experience",
            "persona",
            "industry",
            "market",
            "offering",
        ]:
            if feature in optimal_feature_values:
                feature_value = optimal_feature_values[feature]
                matrix_info = matrix_selections[feature]

                # Find the champion for this feature
                champion = next(
                    (c for c in matrix_champions if c["feature"] == feature), None
                )

                if champion and feature_value:
                    summary_lines.extend(
                        [
                            f"",
                            f"{feature.upper()}:",
                            f"  Value: {feature_value[:80]}{'...' if len(feature_value) > 80 else ''}",
                            f"  Source: {champion['name']} ({champion['company']})",
                            f"  Matrix: {matrix_info['selected_matrix']} (weight: {matrix_info['weight_value']:.2f})",
                            f"  Score: {champion['avg_weighted_score']:.3f}",
                        ]
                    )

        summary_lines.extend(
            [
                "",
                "INTERPRETATION:",
                "This profile combines the strongest feature values from each matrix,",
                "creating an optimal 'super-node' that would maximize edge weights",
                "across all feature dimensions based on your current weight configuration.",
            ]
        )

        return "\n".join(summary_lines)

    def _explain_greedy_strategy(
        self, matrix_selections: Dict[str, Dict], matrix_champions: List[Dict]
    ) -> str:
        """
        Explain the greedy strategy used for matrix selection.
        """
        if not matrix_selections:
            return "No strategy explanation available"

        explanation_lines = [
            "GREEDY MATRIX SELECTION STRATEGY:",
            "=" * 40,
        ]

        similarity_features = []
        complementarity_features = []

        for feature, selection in matrix_selections.items():
            if selection["selected_matrix"] == "similarity":
                similarity_features.append(
                    f"{feature} (w_s={selection['similarity_weight']:.2f})"
                )
            else:
                complementarity_features.append(
                    f"{feature} (w_c={selection['complementarity_weight']:.2f})"
                )

        explanation_lines.extend(
            [
                "",
                "MATRIX SELECTION SUMMARY:",
                f"Similarity-focused features: {', '.join(similarity_features) if similarity_features else 'None'}",
                f"Complementarity-focused features: {', '.join(complementarity_features) if complementarity_features else 'None'}",
                "",
                "LOGIC:",
                "For each feature, selected the matrix (similarity vs complementarity)",
                "with the higher weight value. This ensures we're optimizing for the",
                "user's expressed preferences in each dimension.",
                "",
                f"RESULT: Found {len(matrix_champions)} champion nodes across 6 matrices.",
                "Combined their feature values into an optimal composite profile.",
            ]
        )

        return "\n".join(explanation_lines)
