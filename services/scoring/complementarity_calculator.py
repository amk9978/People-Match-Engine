#!/usr/bin/env python3

import pandas as pd
from typing import Dict, List
from services.analysis.complementarity_engine import ComplementarityEngine
from services.analysis.matrix_builder import MatrixBuilder
from services.tag_extractor import tag_extractor


class ComplementarityCalculator:
    """Handles all complementarity calculations using causal relationship matrices"""

    def __init__(self):
        self.complementarity_engine = ComplementarityEngine()
        self.matrix_builder = MatrixBuilder()
        self._person_tags_cache = {}

    async def precompute_complementarity_matrices(self, csv_path: str) -> None:
        """Load/build all complementarity matrices"""
        
        # Load/build business complementarity matrix
        if not self.matrix_builder.load_causal_graph_from_redis():
            print("🔧 Business complementarity matrix not found. Building automatically...")
            try:
                await self.matrix_builder.build_causal_relationship_graph(csv_path)
                print("✅ Business complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Error building business matrix: {e}")
                print("Using embedding-based complementarity fallback...")

        # Load/build experience complementarity matrix
        if not self.matrix_builder.load_experience_matrix_from_redis():
            print("🔧 Experience complementarity matrix not found. Building automatically...")
            try:
                await self.matrix_builder.build_experience_complementarity_matrix(csv_path)
                print("✅ Experience complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Using embedding-based experience complementarity fallback")

        # Load/build role complementarity matrix
        if not self.matrix_builder.load_role_matrix_from_redis():
            print("🔧 Role complementarity matrix not found. Building automatically...")
            try:
                await self.matrix_builder.build_role_complementarity_matrix(csv_path)
                print("✅ Role complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Using embedding-based role complementarity fallback")

    async def precompute_person_tags(self, df: pd.DataFrame, embedding_builder) -> None:
        """Precompute all person tags for fast lookups"""
        print("⚡ Precomputing person tags...")
        
        for idx, row in df.iterrows():
            self._person_tags_cache[idx] = {
                "roles": tag_extractor.extract_tags(
                    str(row.get("Professional Identity - Role Specification", "")), "role_spec"
                ),
                "experience": tag_extractor.extract_tags(
                    str(row.get("Professional Identity - Experience Level", "")), "experience"
                ),
                "personas": tag_extractor.extract_tags(
                    str(row.get("All Persona Titles", "")), "personas"
                ),
                "business": await embedding_builder.extract_business_tags_for_person(row),
            }

    async def get_role_complementarity(self, person_i: int, person_j: int) -> float:
        """Get role complementarity between two people"""
        person_i_roles = self._person_tags_cache.get(person_i, {}).get("roles", [])
        person_j_roles = self._person_tags_cache.get(person_j, {}).get("roles", [])
        
        return await self.complementarity_engine.calculate_role_complementarity_fast(
            person_i_roles, person_j_roles
        )

    async def get_experience_complementarity(self, person_i: int, person_j: int) -> float:
        """Get experience complementarity between two people"""
        person_i_exp = self._person_tags_cache.get(person_i, {}).get("experience", [])
        person_j_exp = self._person_tags_cache.get(person_j, {}).get("experience", [])
        
        return await self.complementarity_engine.calculate_experience_complementarity_fast(
            person_i_exp, person_j_exp
        )

    def get_business_complementarity(self, person_i: int, person_j: int) -> float:
        """Get business complementarity between two people"""
        person_i_business = self._person_tags_cache.get(person_i, {}).get("business", {})
        person_j_business = self._person_tags_cache.get(person_j, {}).get("business", {})
        
        return self.complementarity_engine.calculate_business_complementarity_fast(
            person_i_business, person_j_business
        )

    async def get_all_complementarities(self, person_i: int, person_j: int) -> Dict[str, float]:
        """Get all complementarity scores between two people"""
        role_comp = await self.get_role_complementarity(person_i, person_j)
        exp_comp = await self.get_experience_complementarity(person_i, person_j)
        business_comp = self.get_business_complementarity(person_i, person_j)
        
        return {
            "role": role_comp,
            "experience": exp_comp,
            "business": business_comp,
        }

    def clear_cache(self) -> None:
        """Clear all cached person tags"""
        self._person_tags_cache.clear()