#!/usr/bin/env python3

import asyncio
import json
from typing import Dict, List, Set
from services.redis_cache import RedisEmbeddingCache
from services.analysis.business_analyzer import BusinessAnalyzer


class MatrixBuilder:
    """Handles building and caching of complementarity matrices"""

    def __init__(self):
        self.cache = RedisEmbeddingCache()
        self.business_analyzer = BusinessAnalyzer()
        
        # Cache keys for different matrices
        self.CAUSAL_GRAPH_KEY = "causal_graph_complete"
        self.PERSONA_MATRIX_KEY = "persona_complementarity_matrix_complete"
        self.EXPERIENCE_MATRIX_KEY = "experience_complementarity_matrix_complete"
        self.ROLE_MATRIX_KEY = "role_complementarity_matrix_complete"

    async def build_causal_relationship_graph(self, csv_path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Build complete business complementarity graph using ChatGPT analysis"""
        print("🔨 Building business causal relationship graph...")

        # Extract business tags
        business_tags = self.business_analyzer.extract_business_tags_from_dataset(csv_path)
        
        causal_graph = {}
        
        # Process each category
        for category, tags in business_tags.items():
            print(f"\n🏢 Processing {category} category ({len(tags)} tags)...")
            causal_graph[category] = {}
            
            tags_list = sorted(list(tags))  # Convert to sorted list for consistent processing
            
            # Create async tasks for all tag combinations
            tasks = []
            tag_pairs = []
            
            for target_tag in tags_list:
                # Get complementarity scores for this tag against all others
                other_tags = [t for t in tags_list if t != target_tag]
                if other_tags:
                    tasks.append(self.business_analyzer.get_causal_relationships_for_tag(
                        target_tag, other_tags, category
                    ))
                    tag_pairs.append((target_tag, other_tags))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for (target_tag, other_tags), result in zip(tag_pairs, results):
                if isinstance(result, Exception):
                    print(f"  ✗ Error processing {target_tag}: {result}")
                    causal_graph[category][target_tag] = {tag: 0.5 for tag in other_tags}
                else:
                    causal_graph[category][target_tag] = result
                    print(f"  ✓ {target_tag}: got {len(result)} relationships")

        # Cache the complete graph
        print(f"\n💾 Caching complete causal relationship graph...")
        self.cache.set(self.CAUSAL_GRAPH_KEY, json.dumps(causal_graph))
        
        print(f"✅ Business causal relationship graph complete!")
        return causal_graph

    async def build_persona_complementarity_matrix(self, csv_path: str) -> Dict[str, Dict[str, float]]:
        """Build persona complementarity matrix using ChatGPT analysis"""
        print("🔨 Building persona complementarity matrix...")

        # Extract persona tags
        persona_tags = self.business_analyzer.extract_persona_tags_from_dataset(csv_path)
        persona_list = sorted(list(persona_tags))
        
        matrix = {}
        
        # Create batches for efficient processing
        batch_size = 20  # Process in smaller batches to avoid overwhelming ChatGPT
        
        for i in range(0, len(persona_list), batch_size):
            batch_end = min(i + batch_size, len(persona_list))
            batch_tags = persona_list[i:batch_end]
            
            print(f"Processing persona batch {i//batch_size + 1}: {len(batch_tags)} tags")
            
            # Create tasks for this batch
            tasks = []
            target_tags = []
            
            for target_tag in batch_tags:
                other_tags = [t for t in persona_list if t != target_tag]
                if other_tags:
                    tasks.append(self.business_analyzer.get_causal_relationships_for_tag(
                        target_tag, other_tags, "persona"
                    ))
                    target_tags.append(target_tag)
            
            # Execute batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for target_tag, result in zip(target_tags, results):
                if isinstance(result, Exception):
                    print(f"  ✗ Error processing {target_tag}: {result}")
                    matrix[target_tag] = {tag: 0.5 for tag in persona_list if tag != target_tag}
                else:
                    matrix[target_tag] = result
                    print(f"  ✓ {target_tag}: {len(result)} relationships")

        # Cache the matrix
        print(f"💾 Caching persona complementarity matrix...")
        self.cache.set(self.PERSONA_MATRIX_KEY, json.dumps(matrix))
        
        print(f"✅ Persona complementarity matrix complete!")
        return matrix

    async def build_experience_complementarity_matrix(self, csv_path: str) -> Dict[str, Dict[str, float]]:
        """Build experience complementarity matrix using ChatGPT analysis"""
        print("🔨 Building experience complementarity matrix...")

        # Extract experience tags (reuse business analyzer pattern)
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        experience_tags = set()
        for _, row in df.iterrows():
            if pd.notna(row["Professional Identity - Experience Level"]):
                tags = [tag.strip() for tag in str(row["Professional Identity - Experience Level"]).split("|")]
                experience_tags.update(tag for tag in tags if tag.strip())

        experience_list = sorted(list(experience_tags))
        print(f"Found {len(experience_list)} unique experience levels")
        
        matrix = {}
        
        # Process all experience combinations
        tasks = []
        target_tags = []
        
        for target_tag in experience_list:
            other_tags = [t for t in experience_list if t != target_tag]
            if other_tags:
                tasks.append(self.business_analyzer.get_causal_relationships_for_tag(
                    target_tag, other_tags, "experience"
                ))
                target_tags.append(target_tag)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for target_tag, result in zip(target_tags, results):
            if isinstance(result, Exception):
                print(f"  ✗ Error processing {target_tag}: {result}")
                matrix[target_tag] = {tag: 0.5 for tag in experience_list if tag != target_tag}
            else:
                matrix[target_tag] = result
                print(f"  ✓ {target_tag}: {len(result)} relationships")

        # Cache the matrix
        print(f"💾 Caching experience complementarity matrix...")
        self.cache.set(self.EXPERIENCE_MATRIX_KEY, json.dumps(matrix))
        
        print(f"✅ Experience complementarity matrix complete!")
        return matrix

    async def build_role_complementarity_matrix(self, csv_path: str) -> Dict[str, Dict[str, float]]:
        """Build role complementarity matrix using ChatGPT analysis"""
        print("🔨 Building role complementarity matrix...")

        # Extract role tags
        role_tags = self.business_analyzer.extract_role_tags_from_dataset(csv_path)
        role_list = sorted(list(role_tags))
        
        matrix = {}
        
        # Process all role combinations
        tasks = []
        target_tags = []
        
        for target_tag in role_list:
            other_tags = [t for t in role_list if t != target_tag]
            if other_tags:
                tasks.append(self.business_analyzer.get_causal_relationships_for_tag(
                    target_tag, other_tags, "role"
                ))
                target_tags.append(target_tag)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for target_tag, result in zip(target_tags, results):
            if isinstance(result, Exception):
                print(f"  ✗ Error processing {target_tag}: {result}")
                matrix[target_tag] = {tag: 0.5 for tag in role_list if tag != target_tag}
            else:
                matrix[target_tag] = result
                print(f"  ✓ {target_tag}: {len(result)} relationships")

        # Cache the matrix
        print(f"💾 Caching role complementarity matrix...")
        self.cache.set(self.ROLE_MATRIX_KEY, json.dumps(matrix))
        
        print(f"✅ Role complementarity matrix complete!")
        return matrix

    def load_causal_graph_from_redis(self) -> bool:
        """Load business causal graph from Redis cache"""
        cached_graph = self.cache.get(self.CAUSAL_GRAPH_KEY)
        if cached_graph:
            try:
                graph_data = json.loads(cached_graph)
                print(f"✅ Loaded business causal graph from cache")
                return True
            except json.JSONDecodeError:
                print(f"⚠️ Invalid cached business graph, will rebuild")
                return False
        return False

    def load_persona_matrix_from_redis(self) -> bool:
        """Load persona complementarity matrix from Redis cache"""
        cached_matrix = self.cache.get(self.PERSONA_MATRIX_KEY)
        if cached_matrix:
            try:
                matrix_data = json.loads(cached_matrix)
                print(f"✅ Loaded persona matrix from cache ({len(matrix_data)} entries)")
                return True
            except json.JSONDecodeError:
                print(f"⚠️ Invalid cached persona matrix, will rebuild")
                return False
        return False

    def load_experience_matrix_from_redis(self) -> bool:
        """Load experience complementarity matrix from Redis cache"""
        cached_matrix = self.cache.get(self.EXPERIENCE_MATRIX_KEY)
        if cached_matrix:
            try:
                matrix_data = json.loads(cached_matrix)
                print(f"✅ Loaded experience matrix from cache ({len(matrix_data)} entries)")
                return True
            except json.JSONDecodeError:
                print(f"⚠️ Invalid cached experience matrix, will rebuild")
                return False
        return False

    def load_role_matrix_from_redis(self) -> bool:
        """Load role complementarity matrix from Redis cache"""
        cached_matrix = self.cache.get(self.ROLE_MATRIX_KEY)
        if cached_matrix:
            try:
                matrix_data = json.loads(cached_matrix)
                print(f"✅ Loaded role matrix from cache ({len(matrix_data)} entries)")
                return True
            except json.JSONDecodeError:
                print(f"⚠️ Invalid cached role matrix, will rebuild")
                return False
        return False