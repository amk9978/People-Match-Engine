#!/usr/bin/env python3

import heapq
import os
from typing import Dict, List, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd

from services.data.csv_loader import CSVLoader
from services.data.embedding_builder import EmbeddingBuilder
from services.scoring.similarity_calculator import SimilarityCalculator
from services.scoring.complementarity_calculator import ComplementarityCalculator
from services.scoring.graph_scorer import GraphScorer
from services.optimization.faiss_engine import FaissOptimizer
from services.analysis.subgraph_analyzer import SubgraphAnalyzer


class GraphBuilder:
    """Handles NetworkX graph construction and dense subgraph algorithms"""

    def __init__(self, csv_path: str, min_density: float = None):
        self.csv_path = csv_path
        self.min_density = min_density or float(os.getenv("min_density", 0.1))
        self.graph = None
        self.df = None
        
        # Initialize components
        self.csv_loader = CSVLoader(csv_path)
        self.embedding_builder = EmbeddingBuilder()
        self.similarity_calc = SimilarityCalculator()
        self.complementarity_calc = ComplementarityCalculator()
        self.scorer = GraphScorer()
        self.faiss_optimizer = FaissOptimizer()
        self.subgraph_analyzer = SubgraphAnalyzer()

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the dataset"""
        self.df = self.csv_loader.load_data()
        return self.df

    async def create_graph(self, feature_embeddings: Dict[str, np.ndarray], user_prompt: str = None) -> nx.Graph:
        """Create graph using FAISS optimization for performance on large datasets"""
        
        if self.faiss_optimizer.is_enabled():
            return await self.create_graph_faiss(feature_embeddings, user_prompt)
        else:
            return await self.create_graph_optimized(feature_embeddings, user_prompt)

    async def create_graph_optimized(self, feature_embeddings: Dict[str, np.ndarray], user_prompt: str = None) -> nx.Graph:
        """Optimized graph creation with 2+2 architecture and ChatGPT weight tuning"""

        self.graph = nx.Graph()
        num_people = len(self.df)

        # Add nodes (people)
        for idx, row in self.df.iterrows():
            self.graph.add_node(
                idx, name=row["Person Name"], company=row["Person Company"]
            )

        # Initialize complementarity matrices
        await self.complementarity_calc.precompute_complementarity_matrices(self.csv_path)
        
        # Precompute similarity matrices
        self.similarity_calc.precompute_similarity_matrices(feature_embeddings)

        # Precompute person tags for complementarity
        await self.complementarity_calc.precompute_person_tags(self.df, self.embedding_builder)

        # Get 2+2 architecture weights using ChatGPT or defaults
        weights = await self.scorer.get_tuned_2plus2_weights(user_prompt)
        
        print(f"📊 Using 2+2 architecture with weights:")
        print(f"   Person Similarity: {weights[0]:.3f}")
        print(f"   Person Complementarity: {weights[1]:.3f}")
        print(f"   Business Similarity: {weights[2]:.3f}")
        print(f"   Business Complementarity: {weights[3]:.3f}")
        print(f"   Total: {sum(weights):.3f}")

        # Compute pairwise similarities and complementarities
        print(f"⚡ Computing {num_people * (num_people - 1) // 2} pairwise similarities...")
        edges_added = 0

        for i in range(num_people):
            for j in range(i + 1, num_people):

                # Get all similarities
                similarities = self.similarity_calc.get_all_similarities(i, j)
                
                # Get all complementarities
                complementarities = await self.complementarity_calc.get_all_complementarities(i, j)

                # Calculate 2+2 architecture score
                score_2plus2 = self.scorer.calculate_2plus2_score(
                    similarities["role"], similarities["experience"], 
                    complementarities["role"], complementarities["experience"],
                    similarities["industry"], similarities["market"], 
                    similarities["offering"], similarities["persona"],
                    complementarities["business"],
                    weights
                )

                # Add edge if score exceeds threshold
                if score_2plus2 > 0.1:  # Minimum edge threshold
                    self.graph.add_edge(i, j, weight=score_2plus2)
                    edges_added += 1

        print(f"✅ Created optimized graph with {self.graph.number_of_nodes()} nodes and {edges_added} edges")
        return self.graph

    async def create_graph_faiss(self, feature_embeddings: Dict[str, np.ndarray], user_prompt: str = None) -> nx.Graph:
        """Create graph using FAISS with 2+2 architecture for maximum performance"""

        self.graph = nx.Graph()

        # Add nodes (people)
        for idx, row in self.df.iterrows():
            self.graph.add_node(
                idx, name=row["Person Name"], company=row["Person Company"]
            )

        # Initialize FAISS optimization
        self.faiss_optimizer.initialize_faiss(feature_embeddings)

        # Get candidate pairs from FAISS
        candidate_pairs, feature_similarities = self.faiss_optimizer.get_candidate_pairs()

        # Get 2+2 architecture weights
        weights = await self.scorer.get_tuned_2plus2_weights(user_prompt)
        
        print(f"📊 Using 2+2 FAISS architecture with weights:")
        print(f"   Person Similarity: {weights[0]:.3f}")
        print(f"   Person Complementarity: {weights[1]:.3f}")
        print(f"   Business Similarity: {weights[2]:.3f}")
        print(f"   Business Complementarity: {weights[3]:.3f}")

        # Initialize complementarity matrices
        await self.complementarity_calc.precompute_complementarity_matrices(self.csv_path)
        await self.complementarity_calc.precompute_person_tags(self.df, self.embedding_builder)

        # Process candidate pairs
        edges_added = 0
        for person_i, person_j in candidate_pairs:
            
            # Get similarities from FAISS results
            similarities = self.faiss_optimizer.get_similarity_scores(person_i, person_j, feature_similarities)
            
            # Get complementarities
            complementarities = await self.complementarity_calc.get_all_complementarities(person_i, person_j)

            # Calculate 2+2 score
            score_2plus2 = self.scorer.calculate_2plus2_score(
                similarities["role"], similarities["experience"], 
                complementarities["role"], complementarities["experience"],
                similarities["industry"], similarities["market"], 
                similarities["offering"], similarities["persona"],
                complementarities["business"],
                weights
            )

            # Add edge if score exceeds threshold
            if score_2plus2 > 0.1:
                self.graph.add_edge(person_i, person_j, weight=score_2plus2)
                edges_added += 1

        print(f"✅ Created FAISS graph with {self.graph.number_of_nodes()} nodes and {edges_added} edges")
        return self.graph


    def densest_subgraph_peeling(self, find_all: bool = False) -> List[Set[int]]:
        """Find dense subgraphs using iterative peeling algorithm"""
        if not self.graph:
            raise ValueError("Graph not created yet. Call create_graph() first.")

        print(f"🔍 Finding dense subgraphs (min_density={self.min_density:.3f})...")
        dense_subgraphs = []
        working_graph = self.graph.copy()

        iteration = 0
        while working_graph.number_of_nodes() > 0:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            print(
                f"Working graph: {working_graph.number_of_nodes()} nodes, {working_graph.number_of_edges()} edges"
            )

            if working_graph.number_of_nodes() < 3:
                print("⏹️ Stopping: Less than 3 nodes remaining")
                break

            # Find node with minimum degree
            degrees = dict(working_graph.degree())
            if not degrees:
                break

            min_degree_node = min(degrees, key=degrees.get)
            min_degree = degrees[min_degree_node]

            print(f"Node with minimum degree: {min_degree_node} (degree={min_degree})")

            # Check if current graph forms a dense subgraph
            current_nodes = set(working_graph.nodes())
            current_density = self.calculate_subgraph_density(current_nodes)
            print(f"Current subgraph density: {current_density:.4f}")

            if current_density >= self.min_density:
                print(f"✅ Found dense subgraph with {len(current_nodes)} nodes")

                # Get member details
                members = []
                total_weight = 0.0
                for node in current_nodes:
                    person_data = self.df.iloc[node]
                    members.append(
                        {
                            "name": person_data["Person Name"],
                            "company": person_data["Person Company"],
                            "role": person_data[
                                "Professional Identity - Role Specification"
                            ],
                        }
                    )

                # Calculate average edge weight
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

                dense_subgraph = {
                    "nodes": current_nodes,
                    "size": len(current_nodes),
                    "density": current_density,
                    "avg_edge_weight": avg_weight,
                    "members": members,
                }

                dense_subgraphs.append(current_nodes)

                print(f"Subgraph summary:")
                print(f"  • Size: {len(current_nodes)} people")
                print(f"  • Density: {current_density:.4f}")
                print(f"  • Average edge weight: {avg_weight:.4f}")

                if not find_all:
                    print("🎯 Found target dense subgraph (find_all=False)")
                    break

            # Remove the minimum degree node and continue
            working_graph.remove_node(min_degree_node)

        if not dense_subgraphs:
            print(f"❌ No dense subgraphs found with density >= {self.min_density}")

        return dense_subgraphs

    def find_largest_dense_subgraph(self) -> Tuple[Set[int], float]:
        """Find the largest dense subgraph that meets minimum density requirement"""
        dense_subgraphs = self.densest_subgraph_peeling(find_all=True)

        if not dense_subgraphs:
            return set(), 0.0

        # Find the largest subgraph
        largest_subgraph = max(dense_subgraphs, key=len)
        largest_density = self.calculate_subgraph_density(largest_subgraph)

        print(f"🏆 Largest dense subgraph: {len(largest_subgraph)} nodes, density={largest_density:.4f}")
        
        return largest_subgraph, largest_density

    # Complete analysis pipeline
    async def run_complete_analysis(self) -> Dict:
        """Run complete analysis pipeline"""
        print("Starting multi-feature graph matching analysis with tag deduplication...")

        # Step 1: Load data
        self.load_data()

        # Step 2: Preprocess tags (deduplication)
        dedup_stats = await self.preprocess_tags()

        # Step 3: Create embeddings (now using deduplicated tags)
        feature_embeddings = await self.embed_features()

        # Step 4: Create hybrid similarity graph
        await self.create_graph(feature_embeddings)

        largest_dense_nodes, density = self.find_largest_dense_subgraph()

        result = self.get_subgraph_info(largest_dense_nodes, feature_embeddings)

        # Add deduplication stats to result
        result["deduplication_stats"] = dedup_stats

        return result

    # Delegation methods
    async def preprocess_tags(self, similarity_threshold: float = 0.7, fuzzy_threshold: float = 0.90, force_rebuild: bool = False) -> Dict[str, any]:
        """Delegate to embedding builder"""
        return await self.embedding_builder.preprocess_tags(self.csv_path, similarity_threshold, fuzzy_threshold, force_rebuild)

    def extract_tags(self, persona_titles: str) -> List[str]:
        """Delegate to embedding builder"""
        return self.embedding_builder.extract_tags(persona_titles)

    async def embed_features(self) -> Dict[str, np.ndarray]:
        """Delegate to embedding builder"""
        feature_columns = self.csv_loader.get_feature_columns()
        return await self.embedding_builder.embed_features(self.df, feature_columns)

    # Analysis methods - delegate to subgraph analyzer
    def get_subgraph_info(self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray]) -> Dict:
        """Delegate to subgraph analyzer"""
        return self.subgraph_analyzer.get_subgraph_info(nodes, feature_embeddings, self.df, self.graph)

    def analyze_subgraph_centroids(self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray]) -> Dict:
        """Delegate to subgraph analyzer"""
        return self.subgraph_analyzer.analyze_subgraph_centroids(nodes, feature_embeddings)

    def analyze_subgroups(self, nodes: Set[int]) -> Dict:
        """Delegate to subgraph analyzer"""
        return self.subgraph_analyzer.analyze_subgroups(nodes, self.graph, self.df)

    def calculate_subgraph_density(self, nodes: Set[int]) -> float:
        """Delegate to subgraph analyzer"""
        return self.subgraph_analyzer.calculate_subgraph_density(nodes, self.graph)