import itertools
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

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

        members = []
        for node in nodes:
            person_data = df.iloc[node]
            members.append(
                {
                    "name": person_data["Person Name"],
                    "linkedin": person_data.get("LinkedIn URL", ""),
                }
            )

        subgraph = graph.subgraph(nodes)
        density = self.calculate_subgraph_density(nodes, graph)

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
            source_name = df.iloc[u]["Person Name"]
            target_name = df.iloc[v]["Person Name"]
            edges_data.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "weight": data.get("weight", 0.0),
                    "score": data.get("weight", 0.0),
                }
            )

        hybrid_insights = {}
        complementarity_insights = {}
        feature_importance_analysis = {}
        dataset_values_analysis = {}

        if matrix_builder and tuned_w_s and tuned_w_c:
            complementarity_insights = self.analyze_complementarity_centroids(
                nodes, matrix_builder, tuned_w_s, tuned_w_c
            )

            hybrid_insights = self.analyze_hybrid_centroids(
                nodes, feature_embeddings, matrix_builder, tuned_w_s, tuned_w_c
            )

            feature_importance_analysis = self.analyze_feature_importance(
                nodes, graph, matrix_builder, tuned_w_s, tuned_w_c
            )

            dataset_values_analysis = self.analyze_dataset_values(
                nodes, df, matrix_builder
            )

        weighted_communities = self.detect_weighted_communities(nodes, graph)

        max_cycle = self.find_maximum_weight_cycle(nodes, graph)

        layout_coords = self.compute_stress_layout(nodes, graph)

        result = {
            "nodes": nodes,
            "size": len(nodes),
            "density": density,
            "avg_edge_weight": avg_weight,
            "members": members,
            "edges": len(edges_with_weights),
            "edges_data": edges_data,
            "centroid_insights": self.analyze_subgraph_centroids(
                nodes, feature_embeddings
            ),
            "complementarity_insights": complementarity_insights,
            "hybrid_insights": hybrid_insights,
            "feature_importance": feature_importance_analysis,
            "dataset_values": dataset_values_analysis,
            "weighted_communities": weighted_communities,
            "maximum_weight_cycle": max_cycle,
            "stress_layout": layout_coords,
        }

        legacy_subgroups = self.analyze_subgroups(nodes, graph, df)
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
        self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray]
    ) -> Dict:
        """Analyze centroids to identify which feature values make the subgraph dense"""
        centroids = {}

        for feature_name, embeddings in feature_embeddings.items():
            subgraph_embeddings = embeddings[list(nodes)]
            centroid = np.mean(subgraph_embeddings, axis=0)

            closest_to_centroid = self._find_nodes_closest_to_centroid(
                nodes, embeddings, centroid, feature_name
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
        top_k: int = 5,
    ) -> List[Dict]:
        """Find nodes within subgraph that are closest to centroid - these drive density"""
        if not nodes:
            return []

        node_list = list(nodes)
        subgraph_embeddings = embeddings[node_list]

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
                    person_data = df.iloc[node]
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
        """Detect communities in weighted subgraph using multiple algorithms"""
        if len(nodes) < 3:
            return {
                "communities": [],
                "summary": "Insufficient nodes for community detection",
            }

        subgraph = graph.subgraph(nodes)
        communities_results = {}

        try:

            import networkx.algorithms.community as nx_comm

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

        best_method = self._select_best_communities(communities_results)

        return {
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
    ) -> Dict:
        """Combine embedding similarity and complementarity analysis"""
        if not nodes or len(nodes) < 2:
            return {
                "hybrid_patterns": {},
                "summary": "Insufficient nodes for hybrid analysis",
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
        hybrid_patterns = {}

        for category in feature_categories:
            if category not in feature_embeddings:
                continue

            category_embeddings = feature_embeddings[category][node_list]

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
            person_data = df.iloc[node_idx]
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
