#!/usr/bin/env python3

import json
from typing import Dict, List
from services.redis_cache import RedisEmbeddingCache
from services.embedding_service import embedding_service
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ComplementarityEngine:
    """Handles fast complementarity lookups using cached matrices"""

    def __init__(self):
        self.cache = RedisEmbeddingCache()
        
        # Cache keys
        self.CAUSAL_GRAPH_KEY = "causal_graph_complete"
        self.PERSONA_MATRIX_KEY = "persona_complementarity_matrix_complete"
        self.EXPERIENCE_MATRIX_KEY = "experience_complementarity_matrix_complete"
        self.ROLE_MATRIX_KEY = "role_complementarity_matrix_complete"
        
        # Loaded matrices
        self._causal_graph = None
        self._persona_matrix = None
        self._experience_matrix = None
        self._role_matrix = None

    def _load_causal_graph(self) -> Dict:
        """Load causal graph from cache if not already loaded"""
        if self._causal_graph is None:
            cached_graph = self.cache.get(self.CAUSAL_GRAPH_KEY)
            if cached_graph:
                self._causal_graph = json.loads(cached_graph)
        return self._causal_graph

    def _load_persona_matrix(self) -> Dict:
        """Load persona matrix from cache if not already loaded"""
        if self._persona_matrix is None:
            cached_matrix = self.cache.get(self.PERSONA_MATRIX_KEY)
            if cached_matrix:
                self._persona_matrix = json.loads(cached_matrix)
        return self._persona_matrix

    def _load_experience_matrix(self) -> Dict:
        """Load experience matrix from cache if not already loaded"""
        if self._experience_matrix is None:
            cached_matrix = self.cache.get(self.EXPERIENCE_MATRIX_KEY)
            if cached_matrix:
                self._experience_matrix = json.loads(cached_matrix)
        return self._experience_matrix

    def _load_role_matrix(self) -> Dict:
        """Load role matrix from cache if not already loaded"""
        if self._role_matrix is None:
            cached_matrix = self.cache.get(self.ROLE_MATRIX_KEY)
            if cached_matrix:
                self._role_matrix = json.loads(cached_matrix)
        return self._role_matrix

    def get_causal_score(self, tag1: str, tag2: str, category: str) -> float:
        """Get causal relationship score between two tags in a category"""
        causal_graph = self._load_causal_graph()
        if not causal_graph or category not in causal_graph:
            return 0.5  # Default moderate score

        # Try both directions
        if tag1 in causal_graph[category] and tag2 in causal_graph[category][tag1]:
            return causal_graph[category][tag1][tag2]
        elif tag2 in causal_graph[category] and tag1 in causal_graph[category][tag2]:
            return causal_graph[category][tag2][tag1]
        
        return 0.5  # Default if not found

    def calculate_business_complementarity_fast(
        self, person1_business: Dict[str, List[str]], person2_business: Dict[str, List[str]]
    ) -> float:
        """Fast business complementarity calculation using cached graph"""
        
        if not person1_business or not person2_business:
            return 0.0

        total_score = 0.0
        category_count = 0

        business_categories = ["industry", "market", "offering"]

        for category in business_categories:
            person1_tags = person1_business.get(category, [])
            person2_tags = person2_business.get(category, [])

            if not person1_tags or not person2_tags:
                continue

            category_scores = []
            for tag1 in person1_tags:
                for tag2 in person2_tags:
                    if tag1 != tag2:  # Skip identical tags
                        score = self.get_causal_score(tag1, tag2, category)
                        category_scores.append(score)

            if category_scores:
                # Use max score for each category
                category_max = max(category_scores)
                total_score += category_max
                category_count += 1

        return total_score / category_count if category_count > 0 else 0.0

    async def calculate_persona_complementarity_fast(
        self, person1_personas: List[str], person2_personas: List[str]
    ) -> float:
        """Fast persona complementarity calculation using cached matrix"""
        
        if not person1_personas or not person2_personas:
            return 0.0

        persona_matrix = self._load_persona_matrix()
        if not persona_matrix:
            # Fallback to embedding similarity
            return await self._embedding_fallback(person1_personas, person2_personas)

        scores = []
        for tag1 in person1_personas:
            for tag2 in person2_personas:
                if tag1 != tag2:
                    if tag1 in persona_matrix and tag2 in persona_matrix[tag1]:
                        scores.append(persona_matrix[tag1][tag2])
                    elif tag2 in persona_matrix and tag1 in persona_matrix[tag2]:
                        scores.append(persona_matrix[tag2][tag1])

        return max(scores) if scores else 0.0

    async def calculate_experience_complementarity_fast(
        self, person1_experience: List[str], person2_experience: List[str]
    ) -> float:
        """Fast experience complementarity calculation using cached matrix"""
        
        if not person1_experience or not person2_experience:
            return 0.0

        experience_matrix = self._load_experience_matrix()
        if not experience_matrix:
            # Fallback to embedding similarity
            return await self._embedding_fallback(person1_experience, person2_experience)

        scores = []
        for tag1 in person1_experience:
            for tag2 in person2_experience:
                if tag1 != tag2:
                    if tag1 in experience_matrix and tag2 in experience_matrix[tag1]:
                        scores.append(experience_matrix[tag1][tag2])
                    elif tag2 in experience_matrix and tag1 in experience_matrix[tag2]:
                        scores.append(experience_matrix[tag2][tag1])

        return max(scores) if scores else 0.0

    async def calculate_role_complementarity_fast(
        self, person1_roles: List[str], person2_roles: List[str]
    ) -> float:
        """Fast role complementarity calculation using cached matrix"""
        
        if not person1_roles or not person2_roles:
            return 0.0

        role_matrix = self._load_role_matrix()
        if not role_matrix:
            # Fallback to embedding similarity
            return await self._embedding_fallback(person1_roles, person2_roles)

        scores = []
        for tag1 in person1_roles:
            for tag2 in person2_roles:
                if tag1 != tag2:
                    if tag1 in role_matrix and tag2 in role_matrix[tag1]:
                        scores.append(role_matrix[tag1][tag2])
                    elif tag2 in role_matrix and tag1 in role_matrix[tag2]:
                        scores.append(role_matrix[tag2][tag1])

        return max(scores) if scores else 0.0

    async def _embedding_fallback(self, tags1: List[str], tags2: List[str]) -> float:
        """Fallback to embedding-based similarity when matrices unavailable"""
        if not tags1 or not tags2:
            return 0.0

        try:
            # Get embeddings for all tags
            embeddings1 = []
            embeddings2 = []
            
            for tag in tags1:
                embedding = await embedding_service.get_embedding(tag)
                embeddings1.append(embedding)
            
            for tag in tags2:
                embedding = await embedding_service.get_embedding(tag)
                embeddings2.append(embedding)

            # Calculate cross-similarities
            embeddings1 = np.array(embeddings1)
            embeddings2 = np.array(embeddings2)
            
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)
            max_similarity = np.max(similarity_matrix)
            
            # Convert similarity to complementarity (inverse relationship for many cases)
            # High similarity = low complementarity, except for very high similarities
            if max_similarity > 0.9:
                return max_similarity  # Very similar can still be valuable
            else:
                return 1.0 - max_similarity  # Different = more complementary

        except Exception as e:
            print(f"  ⚠️ Embedding fallback failed: {e}")
            return 0.5  # Default moderate score

    def clear_cache(self) -> None:
        """Clear all loaded matrices to force reload"""
        self._causal_graph = None
        self._persona_matrix = None
        self._experience_matrix = None
        self._role_matrix = None