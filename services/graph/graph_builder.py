import base64
import heapq
import json
import logging
import os
import pickle
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

import settings
from services.analysis.matrix_builder import MatrixBuilder
from services.analysis.subgraph_analyzer import SubgraphAnalyzer
from services.graph.scoring.generalized_mean import combine_edge_weight, tune_parameters
from services.graph.scoring.similarity_calculator import SimilarityCalculator
from services.preprocessing.csv_loader import CSVLoader
from services.preprocessing.embedding_builder import EmbeddingBuilder
from services.redis.redis_cache import RedisEmbeddingCache

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Handles NetworkX graph construction and dense subgraph algorithms"""

    def __init__(self, csv_path: str, min_density: float = None):
        self.csv_path = csv_path
        self.min_density = min_density or settings.MIN_DENSITY
        self.graph = None
        self.df = None
        self.tuned_w_s = {
            "role": 1.0,
            "experience": 1.0,
            "industry": 1.0,
            "market": 1.0,
            "offering": 1.0,
            "persona": 1.0,
        }
        self.tuned_w_c = {
            "role": 1.0,
            "experience": 1.5,
            "industry": 1.1,
            "market": 1.1,
            "offering": 1.2,
            "persona": 0.9,
        }

        self.csv_loader = CSVLoader(csv_path)
        self.embedding_builder = EmbeddingBuilder()
        self.similarity_calc = SimilarityCalculator()
        self.matrix_builder = MatrixBuilder()
        self.subgraph_analyzer = SubgraphAnalyzer()
        self.cache = RedisEmbeddingCache(key_prefix="graph_cache")

        self.GRAPH_PREFIX = "networkx_graph"
        self.EMBEDDINGS_PREFIX = "feature_embeddings"

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the dataset"""
        self.df = self.csv_loader.load_data()
        return self.df

    def _get_graph_cache_key(self, job_id: str) -> str:
        """Generate cache key for graph"""
        return f"{self.GRAPH_PREFIX}_{job_id}"

    def _get_embeddings_cache_key(self, job_id: str) -> str:
        """Generate cache key for feature embeddings"""
        return f"{self.EMBEDDINGS_PREFIX}_{job_id}"

    def _serialize_graph(self, graph: nx.Graph) -> str:
        """Serialize NetworkX graph for caching"""
        try:

            graph_data = nx.node_link_data(graph)
            return json.dumps(graph_data)
        except Exception as e:
            logger.info(f"Error serializing graph: {e}")
            return None

    def _deserialize_graph(self, data: str) -> nx.Graph:
        """Deserialize NetworkX graph from cache"""
        try:
            graph_data = json.loads(data)
            return nx.node_link_graph(graph_data)
        except Exception as e:
            logger.info(f"Error deserializing graph: {e}")
            return None

    def _serialize_embeddings(self, embeddings: Dict[str, np.ndarray]) -> str:
        """Serialize embeddings for caching"""
        try:
            serialized = {}
            for key, arr in embeddings.items():
                serialized[key] = {
                    "data": base64.b64encode(pickle.dumps(arr)).decode("utf-8"),
                    "shape": arr.shape,
                    "dtype": str(arr.dtype),
                }
            return json.dumps(serialized)
        except Exception as e:
            logger.info(f"Error serializing embeddings: {e}")
            return None

    def _deserialize_embeddings(self, data: str) -> Dict[str, np.ndarray]:
        """Deserialize embeddings from cache"""
        try:
            serialized = json.loads(data)
            embeddings = {}
            for key, arr_data in serialized.items():
                arr = pickle.loads(base64.b64decode(arr_data["data"].encode("utf-8")))
                embeddings[key] = arr
            return embeddings
        except Exception as e:
            logger.info(f"Error deserializing embeddings: {e}")
            return None

    def load_graph_from_cache(self, job_id: str) -> bool:
        """Load graph from Redis cache"""
        cache_key = self._get_graph_cache_key(job_id)
        cached_graph = self.cache.get(cache_key)

        if cached_graph:
            graph = self._deserialize_graph(cached_graph)
            if graph:
                self.graph = graph
                logger.info(
                    f"✅ Loaded graph from cache ({len(graph.nodes)} nodes, {len(graph.edges)} edges)"
                )
                return True
        return False

    def save_graph_to_cache(self, job_id: str) -> bool:
        """Save graph to Redis cache"""
        if not self.graph:
            return False

        cache_key = self._get_graph_cache_key(job_id)
        serialized = self._serialize_graph(self.graph)

        if serialized:
            success = self.cache.set(cache_key, serialized)
            if success:
                logger.info(
                    f"💾 Cached graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
                )
            return success
        return False

    def load_embeddings_from_cache(self, job_id: str) -> Dict[str, np.ndarray]:
        """Load feature embeddings from Redis cache"""
        cache_key = self._get_embeddings_cache_key(job_id)
        cached_embeddings = self.cache.get(cache_key)

        if cached_embeddings:
            embeddings = self._deserialize_embeddings(cached_embeddings)
            if embeddings:
                logger.info(
                    f"✅ Loaded embeddings from cache ({len(embeddings)} features)"
                )
                return embeddings
        return None

    def save_embeddings_to_cache(
        self, embeddings: Dict[str, np.ndarray], job_id: str
    ) -> bool:
        """Save feature embeddings to Redis cache"""
        cache_key = self._get_embeddings_cache_key(job_id)
        serialized = self._serialize_embeddings(embeddings)

        if serialized:
            success = self.cache.set(cache_key, serialized)
            if success:
                logger.info(f"💾 Cached embeddings for {len(embeddings)} features")
            return success
        return False

    async def create_graph(
        self,
        feature_embeddings: Dict[str, np.ndarray],
        job_id: str,
        user_prompt: str = None,
    ) -> nx.Graph:
        """Create graph using FAISS optimization for performance on large datasets with caching"""
        cached_embeddings = self.load_embeddings_from_cache(job_id)
        if cached_embeddings is not None:
            feature_embeddings = cached_embeddings
        else:
            self.save_embeddings_to_cache(feature_embeddings, job_id)

        graph = await self.create_graph_optimized(feature_embeddings, user_prompt)

        self.save_graph_to_cache(job_id)

        return graph

    async def create_graph_optimized(
        self,
        feature_embeddings: Dict[str, np.ndarray],
        user_prompt: str = None,
    ) -> nx.Graph:
        self.graph = nx.Graph()
        num_people = len(self.df)
        for idx, row in self.df.iterrows():
            self.graph.add_node(
                idx, name=row["Person Name"], company=row["Person Company"]
            )
        matrices = await self.matrix_builder.build_all_complementarity_matrices(
            self.csv_path
        )
        self.matrix_builder.load_matrices_into_memory(matrices)
        self.similarity_calc.precompute_similarity_matrices(feature_embeddings)
        await self.matrix_builder.precompute_person_tags(
            self.df, self.embedding_builder
        )
        w_s, w_c = tune_parameters(prompt=user_prompt)

        self.tuned_w_s = w_s
        self.tuned_w_c = w_c

        edges_added = 0

        for i in range(num_people):
            for j in range(i + 1, num_people):
                similarities = self.similarity_calc.get_all_similarities(i, j)
                complementarities = self.matrix_builder.get_all_complementarities(i, j)
                score = combine_edge_weight(
                    similarities,
                    complementarities,
                    w_s=w_s,
                    w_c=w_c,
                    p_s=0.0,
                    p_c=0.5,
                    rho=0.5,
                    lam=0.5,
                    eta=0.2,
                    gamma_e=0.85,
                )
                self.graph.add_edge(i, j, weight=score)
                edges_added += 1

        logger.info(
            f"✅ Created optimized graph with {self.graph.number_of_nodes()} nodes and {edges_added} edges"
        )
        return self.graph

    def densest_subgraph_peeling(self, find_all: bool = False) -> List[Set[int]]:
        """Find dense subgraphs using iterative peeling algorithm"""
        if not self.graph:
            raise ValueError("Graph not created yet. Call create_graph() first.")

        logger.info(
            f"🔍 Finding dense subgraphs (min_density={self.min_density:.3f})..."
        )
        dense_subgraphs = []
        working_graph = self.graph.copy()

        heap = []
        for node in working_graph.nodes():
            weighted_degree = sum(
                data.get("weight", 0.0)
                for _, _, data in working_graph.edges(node, data=True)
            )
            heapq.heappush(heap, (weighted_degree, node))

        iteration = 0
        while heap and working_graph.number_of_nodes() > 0:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration} ---")
            logger.info(
                f"Working graph: {working_graph.number_of_nodes()} nodes, {working_graph.number_of_edges()} edges"
            )

            if working_graph.number_of_nodes() < 3:
                logger.info("⏹️ Stopping: Less than 3 nodes remaining")
                break

            current_nodes = set(working_graph.nodes())
            current_density = self.calculate_subgraph_density(current_nodes)
            logger.info(f"Current subgraph density: {current_density:.4f}")

            if current_density >= self.min_density:
                logger.info(f"✅ Found dense subgraph with {len(current_nodes)} nodes")

                edges_in_subgraph = working_graph.subgraph(current_nodes).edges(
                    data=True
                )
                if edges_in_subgraph:
                    total_weight = sum(
                        data.get("weight", 0.0) for _, _, data in edges_in_subgraph
                    )
                    avg_weight = total_weight / len(edges_in_subgraph)
                else:
                    avg_weight = 0.0

                dense_subgraphs.append(current_nodes)
                if not find_all:
                    logger.info(
                        "🎯 Found target dense subgraph, stopping early (find_all=False)"
                    )
                    return dense_subgraphs

            while heap:
                min_weighted_degree, min_weighted_node = heapq.heappop(heap)

                if min_weighted_node not in working_graph:
                    continue

                current_weighted_degree = sum(
                    data.get("weight", 0.0)
                    for _, _, data in working_graph.edges(min_weighted_node, data=True)
                )

                if abs(current_weighted_degree - min_weighted_degree) < 1e-6:
                    logger.info(
                        f"Removing node {min_weighted_node} (weighted_degree={min_weighted_degree:.4f})"
                    )

                    neighbors = list(working_graph.neighbors(min_weighted_node))
                    working_graph.remove_node(min_weighted_node)

                    for neighbor in neighbors:
                        if neighbor in working_graph:
                            new_weighted_degree = sum(
                                data.get("weight", 0.0)
                                for _, _, data in working_graph.edges(
                                    neighbor, data=True
                                )
                            )
                            heapq.heappush(heap, (new_weighted_degree, neighbor))
                    break
                else:

                    heapq.heappush(heap, (current_weighted_degree, min_weighted_node))

        if not dense_subgraphs:
            logger.info(
                f"❌ No dense subgraphs found with density >= {self.min_density}"
            )

        return dense_subgraphs

    def find_largest_dense_subgraph(self) -> Tuple[Set[int], float]:
        """Find the largest dense subgraph that meets minimum density requirement"""
        dense_subgraphs = self.densest_subgraph_peeling(find_all=False)

        if not dense_subgraphs:
            return set(), 0.0

        largest_subgraph = max(dense_subgraphs, key=len)
        largest_density = self.calculate_subgraph_density(largest_subgraph)

        logger.info(
            f"🏆 Largest dense subgraph: {len(largest_subgraph)} nodes, density={largest_density:.4f}"
        )

        return largest_subgraph, largest_density

    async def run_complete_analysis(self, job_id: str, user_prompt: str = None) -> Dict:
        """Run complete analysis pipeline"""
        logger.info(
            "Starting multi-feature graph matching analysis with tag deduplication..."
        )

        self.load_data()

        dedup_stats = await self.preprocess_tags()

        feature_embeddings = await self.embed_features()

        await self.create_graph(feature_embeddings, job_id, user_prompt)

        largest_dense_nodes, density = self.find_largest_dense_subgraph()

        result = self.get_subgraph_info(largest_dense_nodes, feature_embeddings)

        result["deduplication_stats"] = dedup_stats

        return result

    async def preprocess_tags(
        self,
        similarity_threshold: float = 0.7,
        fuzzy_threshold: float = 0.90,
        force_rebuild: bool = False,
    ) -> Dict[str, any]:
        """Delegate to embedding builder"""
        return await self.embedding_builder.preprocess_tags(
            self.csv_path, similarity_threshold, fuzzy_threshold, force_rebuild
        )

    def extract_tags(self, persona_titles: str) -> List[str]:
        """Delegate to embedding builder"""
        return self.embedding_builder.extract_tags(persona_titles)

    async def embed_features(self) -> Dict[str, np.ndarray]:
        """Delegate to embedding builder"""
        feature_columns = self.csv_loader.get_feature_columns()
        return await self.embedding_builder.embed_features(self.df, feature_columns)

    def get_subgraph_info(
        self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray]
    ) -> Dict:
        """Delegate to subgraph analyzer with complementarity scores and tuned parameters"""
        return self.subgraph_analyzer.get_subgraph_info(
            nodes,
            feature_embeddings,
            self.df,
            self.graph,
            self.matrix_builder,
            self.tuned_w_s,
            self.tuned_w_c,
        )

    def analyze_subgraph_centroids(
        self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray]
    ) -> Dict:
        """Delegate to subgraph analyzer"""
        return self.subgraph_analyzer.analyze_subgraph_centroids(
            nodes, feature_embeddings
        )

    def analyze_subgroups(self, nodes: Set[int]) -> Dict:
        """Delegate to subgraph analyzer"""
        return self.subgraph_analyzer.analyze_subgroups(nodes, self.graph, self.df)

    def calculate_subgraph_density(self, nodes: Set[int]) -> float:
        """Delegate to subgraph analyzer"""
        return self.subgraph_analyzer.calculate_subgraph_density(nodes, self.graph)
