#!/usr/bin/env python3

import os
from typing import Dict, List, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from services.graph_builder import GraphBuilder

# Legacy imports for backward compatibility
from services.redis_cache import RedisEmbeddingCache
from services.semantic_person_deduplicator import SemanticPersonDeduplicator

load_dotenv()


class GraphMatcher:
    """Thin orchestrator that delegates all work to specialized services"""
    
    def __init__(self, csv_path: str, min_density: float = None):
        self.csv_path = csv_path
        self.min_density = min_density or float(os.getenv("min_density", 0.1))
        
        # Single service that handles everything
        self.graph_builder = GraphBuilder(csv_path, min_density)
        
        # Maintain compatibility attributes
        self.df = None
        self.person_vectors = None
        self.graph = None
        self.cache = RedisEmbeddingCache()
        self.person_deduplicator = SemanticPersonDeduplicator()

    # Main pipeline method
    async def run_analysis(self) -> Dict:
        """Run complete analysis pipeline - delegates to GraphBuilder"""
        print("Starting multi-feature graph matching analysis...")
        
        # Delegate everything to GraphBuilder
        result = await self.graph_builder.run_complete_analysis()
        
        # Keep compatibility by setting instance variables
        self.df = self.graph_builder.df
        self.graph = self.graph_builder.graph
        
        print(f"\nAnalysis complete!")
        print(f"Found largest dense subgraph with {result['size']} people")
        print(f"Density: {result['density']:.4f} (threshold: {self.min_density})")
        
        return result

    # Pure delegation methods for compatibility
    def load_data(self) -> pd.DataFrame:
        """Delegate to GraphBuilder"""
        self.df = self.graph_builder.load_data()
        return self.df

    def filter_incomplete_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Delegate to GraphBuilder"""
        return self.graph_builder.csv_loader.filter_incomplete_rows(df)

    async def preprocess_tags(
        self, similarity_threshold: float = 0.7, fuzzy_threshold: float = 0.90, force_rebuild: bool = False
    ) -> Dict[str, any]:
        """Delegate to GraphBuilder"""
        return await self.graph_builder.preprocess_tags(similarity_threshold, fuzzy_threshold, force_rebuild)

    def extract_tags(self, persona_titles: str) -> List[str]:
        """Delegate to GraphBuilder"""
        return self.graph_builder.extract_tags(persona_titles)

    def calculate_2plus2_score(self, role_sim: float, exp_sim: float, role_comp: float, exp_comp: float,
                               industry_sim: float, market_sim: float, offering_sim: float, persona_sim: float,
                               business_comp: float, weights: List[float]) -> float:
        """Delegate to GraphBuilder"""
        return self.graph_builder.scorer.calculate_2plus2_score(
            role_sim, exp_sim, role_comp, exp_comp,
            industry_sim, market_sim, offering_sim, persona_sim,
            business_comp, weights
        )

    async def get_tuned_2plus2_weights(self, user_prompt: str = None) -> List[float]:
        """Delegate to GraphBuilder"""
        return await self.graph_builder.scorer.get_tuned_2plus2_weights(user_prompt)

    async def extract_and_deduplicate_tags(self, text: str, category: str) -> List[str]:
        """Delegate to GraphBuilder"""
        return await self.graph_builder.embedding_builder.extract_and_deduplicate_tags(text, category)

    async def get_cached_embedding(self, tag: str) -> List[float]:
        """Delegate to GraphBuilder"""
        return await self.graph_builder.embedding_builder.get_cached_embedding(tag)

    async def extract_business_tags_for_person(self, row) -> Dict[str, List[str]]:
        """Delegate to GraphBuilder"""
        return await self.graph_builder.embedding_builder.extract_business_tags_for_person(row)

    async def embed_features(self) -> Dict[str, np.ndarray]:
        """Delegate to GraphBuilder"""
        return await self.graph_builder.embed_features()

    async def create_graph(self, feature_embeddings: Dict[str, np.ndarray], user_prompt: str = None) -> nx.Graph:
        """Delegate to GraphBuilder"""
        self.graph = await self.graph_builder.create_graph(feature_embeddings, user_prompt)
        return self.graph

    async def create_graph_optimized(self, feature_embeddings: Dict[str, np.ndarray], user_prompt: str = None) -> nx.Graph:
        """Delegate to GraphBuilder"""
        self.graph = await self.graph_builder.create_graph_optimized(feature_embeddings, user_prompt)
        return self.graph

    async def create_graph_faiss(self, feature_embeddings: Dict[str, np.ndarray], user_prompt: str = None) -> nx.Graph:
        """Delegate to GraphBuilder"""
        self.graph = await self.graph_builder.create_graph_faiss(feature_embeddings, user_prompt)
        return self.graph

    def calculate_subgraph_density(self, nodes: Set[int]) -> float:
        """Delegate to GraphBuilder"""
        return self.graph_builder.calculate_subgraph_density(nodes)

    def densest_subgraph_peeling(self, find_all: bool = False) -> List[Set[int]]:
        """Delegate to GraphBuilder"""
        return self.graph_builder.densest_subgraph_peeling(find_all)

    def find_largest_dense_subgraph(self) -> Tuple[Set[int], float]:
        """Delegate to GraphBuilder"""
        return self.graph_builder.find_largest_dense_subgraph()

    # Analysis methods - delegate to GraphBuilder
    def get_subgraph_info(self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray]) -> Dict:
        """Delegate to GraphBuilder"""
        return self.graph_builder.get_subgraph_info(nodes, feature_embeddings)

    def analyze_subgraph_centroids(self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray]) -> Dict:
        """Delegate to GraphBuilder"""
        return self.graph_builder.analyze_subgraph_centroids(nodes, feature_embeddings)

    def analyze_subgroups(self, nodes: Set[int]) -> Dict:
        """Delegate to GraphBuilder"""
        return self.graph_builder.analyze_subgroups(nodes)

    # Compatibility methods
    def calculate_centroid_and_insights(self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray]) -> Dict:
        """Delegate to analyze_subgraph_centroids"""
        return self.analyze_subgraph_centroids(nodes, feature_embeddings)

    def find_closest_tags_to_centroid(self, centroid: np.ndarray, feature_name: str) -> List[Tuple[str, float]]:
        """Compatibility method"""
        return []

    def find_subgroups_in_subgraph(self, nodes: Set[int], min_subgroup_size: int = 3) -> List[Dict]:
        """Delegate to analyze_subgroups"""
        result = self.analyze_subgroups(nodes)
        return [sg for sg in result["subgroups"] if sg["size"] >= min_subgroup_size]

    def get_expansion_recommendations(self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray], top_n: int = 3) -> List[Dict]:
        """Compatibility method"""
        return []