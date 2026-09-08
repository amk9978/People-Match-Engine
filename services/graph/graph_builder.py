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
from services.cache.cache import Cache
from services.cache.factory import get_cache_backend
from services.graph.scoring.generalized_mean import combine_edge_weight
from services.graph.scoring.similarity_calculator import SimilarityCalculator
from services.preprocessing.csv_loader import CSVLoader
from services.preprocessing.embedding_builder import EmbeddingBuilder
from services.scoring.calibration import calibrate_all
from services.scoring.intent import create_intent_resolver, sample_values
from services.scoring.profile import ScoringProfile
from services.scoring.weight_resolver import WeightResolver

logger = logging.getLogger(__name__)


MIN_SUBGRAPH_NODES = 3


def _rounded(weights: Dict[str, float]) -> Dict[str, float]:
    return {name: round(value, 3) for name, value in weights.items()}


class GraphBuilder:
    """Handles NetworkX graph construction and dense subgraph algorithms"""

    def __init__(
        self,
        csv_path: str,
        min_density: float = None,
        mapping_path: str = None,
        csv_loader: CSVLoader = None,
        embedding_builder: EmbeddingBuilder = None,
        similarity_calc: SimilarityCalculator = None,
        matrix_builder: MatrixBuilder = None,
        subgraph_analyzer: SubgraphAnalyzer = None,
        cache: Cache = None,
        intent_resolver=None,
        weight_resolver: WeightResolver = None,
        scoring_profile: ScoringProfile = None,
    ):
        self.csv_path = csv_path
        self.min_density = min_density or settings.MIN_DENSITY
        self.graph = None
        self.df = None
        self.feature_set = None
        self.intent = None
        self.tuned_w_s = {}
        self.tuned_w_c = {}

        self.csv_loader = csv_loader or CSVLoader(
            csv_path, mapping_path or settings.FEATURE_MAPPING_PATH
        )
        self.embedding_builder = embedding_builder or EmbeddingBuilder()
        self.similarity_calc = similarity_calc or SimilarityCalculator()
        self.matrix_builder = matrix_builder or MatrixBuilder()
        self.subgraph_analyzer = subgraph_analyzer or SubgraphAnalyzer()
        self.cache = cache or Cache(get_cache_backend(), "graph_cache")
        self.intent_resolver = intent_resolver or create_intent_resolver()
        self.weight_resolver = weight_resolver or WeightResolver()
        self.scoring_profile = scoring_profile or ScoringProfile()

        self.GRAPH_PREFIX = "networkx_graph"
        self.EMBEDDINGS_PREFIX = "feature_embeddings"

    def load_data(self) -> pd.DataFrame:
        self.df = self.csv_loader.load_data()
        self.feature_set = self.csv_loader.feature_set
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
                    f"Loaded graph from cache: {len(graph.nodes)} nodes, {len(graph.edges)} edges"
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
                    f"Cached graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges"
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
                    f"Loaded embeddings from cache for {len(embeddings)} features"
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
                logger.info(f"Cached embeddings for {len(embeddings)} features")
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
        """Score every pair and build the complete weighted graph.

        Seven steps run in order. The features come from the dataset, each one
        gets a similarity and a complementarity matrix, informativeness is
        measured from the raw spread of those matrices, the prompt becomes a
        per-feature importance and direction, both matrices are calibrated onto a
        common percentile scale, the three factors compose into the weights, and
        the combiner turns each pair into one edge.

        Measurement comes before calibration and the two must not be swapped.
        Calibration makes every feature uniform by construction, so measuring
        afterwards returns the same number for every feature and the weights
        silently flatten.
        """
        assert self.df is not None, "load_data must run before the graph is built"
        assert self.feature_set is not None, "load_data must resolve the feature set"
        assert self.df.index.equals(pd.RangeIndex(len(self.df))), (
            "people are addressed by position throughout the pipeline, "
            "so the loaded frame must be indexed 0..n-1"
        )

        self.graph = nx.Graph()
        num_people = len(self.df)
        for position in range(num_people):
            row = self.df.iloc[position]
            self.graph.add_node(
                position,
                name=row[self.feature_set.name_column],
                company=self._company_of(row),
            )

        self.similarity_calc.precompute(feature_embeddings)
        await self.matrix_builder.build(self.df, self.feature_set)
        self.matrix_builder.index_people(self.df, self.feature_set)

        raw_similarity = self.similarity_calc.raw_matrices()
        raw_complementarity = self.matrix_builder.raw_matrices()

        self.intent = self.intent_resolver.resolve(
            user_prompt, self.feature_set, sample_values(self.df, self.feature_set)
        )

        self.similarity_calc.apply_calibrated(
            calibrate_all(raw_similarity, preserve_diagonal=True)
        )
        self.matrix_builder.apply_calibrated(
            calibrate_all(raw_complementarity, preserve_diagonal=False)
        )

        self.tuned_w_s, self.tuned_w_c = self.weight_resolver.resolve(
            raw_similarity, raw_complementarity, self.intent
        )
        logger.info(
            f"Similarity weights {_rounded(self.tuned_w_s)}, "
            f"complementarity weights {_rounded(self.tuned_w_c)}"
        )

        for i in range(num_people):
            for j in range(i + 1, num_people):
                score = combine_edge_weight(
                    self.similarity_calc.get_all_similarities(i, j),
                    self.matrix_builder.get_all_complementarities(i, j),
                    w_s=self.tuned_w_s,
                    w_c=self.tuned_w_c,
                    profile=self.scoring_profile,
                )
                self.graph.add_edge(i, j, weight=score)

        logger.info(
            f"Built graph with {self.graph.number_of_nodes()} nodes and "
            f"{self.graph.number_of_edges()} edges"
        )
        return self.graph

    def _company_of(self, row: pd.Series) -> str:
        if self.feature_set.company_column:
            return row[self.feature_set.company_column]
        return ""

    def _weighted_degree(self, graph: nx.Graph, node: int) -> float:
        return sum(
            data.get("weight", 0.0) for _, _, data in graph.edges(node, data=True)
        )

    def _peel_lightest_node(
        self, graph: nx.Graph, heap: List[Tuple[float, int]]
    ) -> bool:
        """Remove the node carrying the least weight and reprice its neighbours.

        Heap entries go stale as neighbours lose edges, so an entry is only acted
        on when it still matches the node's current degree."""
        while heap:
            recorded_degree, node = heapq.heappop(heap)

            if node not in graph:
                continue

            current_degree = self._weighted_degree(graph, node)
            if abs(current_degree - recorded_degree) >= 1e-6:
                heapq.heappush(heap, (current_degree, node))
                continue

            neighbors = list(graph.neighbors(node))
            graph.remove_node(node)
            for neighbor in neighbors:
                heapq.heappush(heap, (self._weighted_degree(graph, neighbor), neighbor))
            return True

        return False

    def densest_subgraph_peeling(self, find_all: bool = False) -> List[Set[int]]:
        """Peel the lightest node repeatedly, keeping every subgraph dense enough.

        Charikar-style greedy peeling on the weighted degree. Each pass records
        the working graph if it clears the density threshold, then drops the node
        contributing least."""
        if not self.graph:
            raise ValueError("Graph not created yet. Call create_graph() first.")

        logger.info(
            f"Peeling for subgraphs with density at least {self.min_density:.3f}"
        )

        dense_subgraphs = []
        working_graph = self.graph.copy()
        heap = [
            (self._weighted_degree(working_graph, node), node)
            for node in working_graph.nodes()
        ]
        heapq.heapify(heap)

        while heap and working_graph.number_of_nodes() >= MIN_SUBGRAPH_NODES:
            current_nodes = set(working_graph.nodes())
            current_density = self.calculate_subgraph_density(current_nodes)
            logger.debug(
                f"{len(current_nodes)} nodes remaining, density {current_density:.4f}"
            )

            if current_density >= self.min_density:
                dense_subgraphs.append(current_nodes)
                if not find_all:
                    logger.info(
                        f"Found a dense subgraph of {len(current_nodes)} nodes, "
                        f"density {current_density:.4f}"
                    )
                    return dense_subgraphs

            if not self._peel_lightest_node(working_graph, heap):
                break

        if not dense_subgraphs:
            logger.info(f"No subgraph reached density {self.min_density}")

        return dense_subgraphs

    def find_largest_dense_subgraph(self) -> Tuple[Set[int], float]:
        """Find the largest dense subgraph that meets minimum density requirement"""
        dense_subgraphs = self.densest_subgraph_peeling(find_all=False)

        if not dense_subgraphs:
            return set(), 0.0

        largest_subgraph = max(dense_subgraphs, key=len)
        largest_density = self.calculate_subgraph_density(largest_subgraph)

        logger.info(
            f"Largest dense subgraph has {len(largest_subgraph)} nodes at "
            f"density {largest_density:.4f}"
        )

        return largest_subgraph, largest_density

    async def run_complete_analysis(self, job_id: str, user_prompt: str = None) -> Dict:
        """Run complete analysis pipeline"""
        logger.info("Starting multi-feature graph matching analysis")

        self.load_data()

        feature_embeddings = await self.embed_features()

        await self.create_graph(feature_embeddings, job_id, user_prompt)

        largest_dense_nodes, density = self.find_largest_dense_subgraph()

        result = self.get_subgraph_info(largest_dense_nodes, feature_embeddings)
        result["complementarity"] = self.matrix_builder.scoring_report.to_dict()

        return result

    async def embed_features(self) -> Dict[str, np.ndarray]:
        return await self.embedding_builder.embed_features(self.df, self.feature_set)

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
            self.feature_set,
        )

    def calculate_subgraph_density(self, nodes: Set[int]) -> float:
        """Delegate to subgraph analyzer"""
        return self.subgraph_analyzer.calculate_subgraph_density(nodes, self.graph)
