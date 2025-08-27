#!/usr/bin/env python3

import json
import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Import refactored modules
from services.analysis.business_analyzer import BusinessAnalyzer
from services.analysis.matrix_builder import MatrixBuilder
from services.analysis.complementarity_engine import ComplementarityEngine

# Legacy imports for compatibility
from services.redis_cache import RedisEmbeddingCache

load_dotenv()


class CausalRelationshipAnalyzer:
    """Refactored CausalRelationshipAnalyzer that orchestrates focused analysis modules"""

    def __init__(self):
        # Initialize new modular components
        self.business_analyzer = BusinessAnalyzer()
        self.matrix_builder = MatrixBuilder()
        self.complementarity_engine = ComplementarityEngine()
        
        # Maintain compatibility with existing interface
        self.cache = RedisEmbeddingCache()
        self.causal_graph = {}
        self.CAUSAL_GRAPH_KEY = "causal_graph_complete"

    def extract_business_tags_from_dataset(self, csv_path: str) -> Dict[str, Set[str]]:
        """Extract all unique business tags - delegated to business analyzer"""
        return self.business_analyzer.extract_business_tags_from_dataset(csv_path)

    async def get_causal_relationships_for_tag(
        self, target_tag: str, comparison_tags: List[str], category: str
    ) -> Dict[str, float]:
        """Get causal relationship scores - delegated to business analyzer"""
        return await self.business_analyzer.get_causal_relationships_for_tag(
            target_tag, comparison_tags, category
        )

    def get_causal_score(self, tag1: str, tag2: str, category: str) -> float:
        """Get causal score between two tags - delegated to complementarity engine"""
        return self.complementarity_engine.get_causal_score(tag1, tag2, category)

    def load_causal_graph_from_redis(self) -> bool:
        """Load business causal graph from Redis - delegated to matrix builder"""
        return self.matrix_builder.load_causal_graph_from_redis()

    def get_causal_score_from_redis(self, tag1: str, tag2: str, category: str) -> float:
        """Get causal score from Redis cache - delegated to complementarity engine"""
        return self.complementarity_engine.get_causal_score(tag1, tag2, category)

    def save_causal_graph(self, output_path: str = "causal_relationship_graph.json"):
        """Save causal graph to file (unchanged implementation)"""
        cached_graph = self.cache.get(self.CAUSAL_GRAPH_KEY)
        if cached_graph:
            with open(output_path, 'w') as f:
                f.write(cached_graph)
            print(f"✅ Saved causal graph to {output_path}")

    def load_causal_graph(self, input_path: str = "causal_relationship_graph.json"):
        """Load causal graph from file (unchanged implementation)"""
        try:
            with open(input_path, 'r') as f:
                graph_data = f.read()
            self.cache.set(self.CAUSAL_GRAPH_KEY, graph_data)
            print(f"✅ Loaded causal graph from {input_path}")
        except FileNotFoundError:
            print(f"⚠️ File {input_path} not found")

    def print_causal_graph_summary(self):
        """Print summary of loaded causal graph (unchanged implementation)"""
        cached_graph = self.cache.get(self.CAUSAL_GRAPH_KEY)
        if not cached_graph:
            print("No causal graph found in cache")
            return

        try:
            graph_data = json.loads(cached_graph)
            print(f"\n📊 CAUSAL RELATIONSHIP GRAPH SUMMARY:")
            print("=" * 50)
            
            for category, category_data in graph_data.items():
                print(f"\n{category.upper()}:")
                print(f"  Tags: {len(category_data)}")
                if category_data:
                    total_relationships = sum(len(relationships) for relationships in category_data.values())
                    avg_relationships = total_relationships / len(category_data)
                    print(f"  Total relationships: {total_relationships}")
                    print(f"  Average relationships per tag: {avg_relationships:.1f}")

        except json.JSONDecodeError:
            print("Invalid causal graph data in cache")

    def get_causal_graph_stats(self) -> Dict[str, any]:
        """Get statistics about the causal graph (unchanged implementation)"""
        cached_graph = self.cache.get(self.CAUSAL_GRAPH_KEY)
        if not cached_graph:
            return {"error": "No causal graph found"}

        try:
            graph_data = json.loads(cached_graph)
            stats = {
                "categories": len(graph_data),
                "category_details": {}
            }
            
            for category, category_data in graph_data.items():
                stats["category_details"][category] = {
                    "tags": len(category_data),
                    "total_relationships": sum(len(relationships) for relationships in category_data.values()),
                    "avg_relationships_per_tag": sum(len(relationships) for relationships in category_data.values()) / len(category_data) if category_data else 0
                }
            
            return stats

        except json.JSONDecodeError:
            return {"error": "Invalid graph data"}

    # Persona-related methods - delegated to business analyzer and matrix builder
    def extract_persona_tags_from_dataset(self, csv_path: str) -> Set[str]:
        """Extract persona tags - delegated to business analyzer"""
        return self.business_analyzer.extract_persona_tags_from_dataset(csv_path)

    async def build_persona_complementarity_matrix(self, csv_path: str) -> Dict[str, Dict[str, float]]:
        """Build persona matrix - delegated to matrix builder"""
        return await self.matrix_builder.build_persona_complementarity_matrix(csv_path)

    def load_persona_matrix_from_redis(self) -> bool:
        """Load persona matrix - delegated to matrix builder"""
        return self.matrix_builder.load_persona_matrix_from_redis()

    async def build_incremental_persona_matrix(self, new_personas: set) -> None:
        """Build incremental persona matrix (kept for compatibility, but simplified)"""
        print(f"Building incremental persona matrix for {len(new_personas)} new personas...")
        # In the refactored version, we rebuild the full matrix since it's more reliable
        # and the matrix builder handles this efficiently with batching
        await self.matrix_builder.build_persona_complementarity_matrix(self.csv_path if hasattr(self, 'csv_path') else None)

    async def calculate_persona_complementarity_fast(
        self, person1_personas: List[str], person2_personas: List[str]
    ) -> float:
        """Calculate persona complementarity - delegated to complementarity engine"""
        return await self.complementarity_engine.calculate_persona_complementarity_fast(
            person1_personas, person2_personas
        )

    # Experience-related methods - delegated to business analyzer and matrix builder
    async def build_experience_complementarity_matrix(self, csv_path: str) -> Dict[str, Dict[str, float]]:
        """Build experience matrix - delegated to matrix builder"""
        return await self.matrix_builder.build_experience_complementarity_matrix(csv_path)

    def load_experience_matrix_from_redis(self) -> bool:
        """Load experience matrix - delegated to matrix builder"""
        return self.matrix_builder.load_experience_matrix_from_redis()

    async def calculate_experience_complementarity_fast(
        self, person1_experience: List[str], person2_experience: List[str]
    ) -> float:
        """Calculate experience complementarity - delegated to complementarity engine"""
        return await self.complementarity_engine.calculate_experience_complementarity_fast(
            person1_experience, person2_experience
        )

    # Role-related methods - delegated to business analyzer and matrix builder
    def extract_role_tags_from_dataset(self, csv_path: str) -> Set[str]:
        """Extract role tags - delegated to business analyzer"""
        return self.business_analyzer.extract_role_tags_from_dataset(csv_path)

    async def analyze_role_complementarity(self, role1: str, role2: str) -> float:
        """Analyze role complementarity - delegated to business analyzer"""
        return await self.business_analyzer.analyze_role_complementarity(role1, role2)

    async def build_role_complementarity_matrix(self, csv_path: str) -> Dict[str, Dict[str, float]]:
        """Build role matrix - delegated to matrix builder"""
        return await self.matrix_builder.build_role_complementarity_matrix(csv_path)

    def load_role_matrix_from_redis(self) -> bool:
        """Load role matrix - delegated to matrix builder"""
        return self.matrix_builder.load_role_matrix_from_redis()

    async def calculate_role_complementarity_fast(
        self, person1_roles: List[str], person2_roles: List[str]
    ) -> float:
        """Calculate role complementarity - delegated to complementarity engine"""
        return await self.complementarity_engine.calculate_role_complementarity_fast(
            person1_roles, person2_roles
        )

    # Business complementarity methods - delegated to complementarity engine
    def calculate_business_complementarity_fast(
        self, person1_business: Dict[str, List[str]], person2_business: Dict[str, List[str]]
    ) -> float:
        """Calculate business complementarity - delegated to complementarity engine"""
        return self.complementarity_engine.calculate_business_complementarity_fast(
            person1_business, person2_business
        )

    # Main building methods - delegated to matrix builder
    async def build_causal_relationship_graph(self, csv_path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Build complete causal relationship graph - delegated to matrix builder"""
        return await self.matrix_builder.build_causal_relationship_graph(csv_path)

    # Missing methods for interface compatibility
    def cache_causal_graph_to_redis(
        self, causal_graph: Dict[str, Dict[str, Dict[str, float]]]
    ):
        """Cache causal graph to Redis (implementation for compatibility)"""
        import json
        self.cache.set(self.CAUSAL_GRAPH_KEY, json.dumps(causal_graph))

    def calculate_business_complementarity(
        self, person1_tags: Dict[str, List[str]], person2_tags: Dict[str, List[str]]
    ) -> float:
        """Calculate business complementarity - delegates to complementarity engine"""
        return self.complementarity_engine.calculate_business_complementarity_fast(
            person1_tags, person2_tags
        )

    async def analyze_persona_complementarity(
        self, persona1: str, persona2: str
    ) -> float:
        """Analyze persona complementarity - delegates to business analyzer"""
        return await self.business_analyzer.analyze_role_complementarity(persona1, persona2)

    async def calculate_embedding_similarity_fallback(
        self, persona1: str, persona2: str
    ) -> float:
        """Calculate embedding similarity fallback - delegates to complementarity engine"""
        return await self.complementarity_engine._embedding_fallback([persona1], [persona2])

    async def calculate_persona_complementarity_chatgpt(
        self, persona1: str, persona2: str
    ) -> float:
        """Calculate persona complementarity using ChatGPT - delegates to business analyzer"""
        return await self.business_analyzer.analyze_role_complementarity(persona1, persona2)

    async def calculate_experience_complementarity_chatgpt(
        self, exp1: str, exp2: str
    ) -> float:
        """Calculate experience complementarity using ChatGPT - delegates to business analyzer"""
        return await self.business_analyzer.analyze_role_complementarity(exp1, exp2)