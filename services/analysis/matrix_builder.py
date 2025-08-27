#!/usr/bin/env python3

import asyncio
import json
import pandas as pd
from typing import Dict, List, Set
from services.redis_cache import RedisEmbeddingCache
from services.analysis.business_analyzer import BusinessAnalyzer
from services.tag_extractor import tag_extractor


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
        
        # Runtime caches for fast access
        self._matrices = {}
        self._person_tags_cache = {}

    async def build_causal_relationship_graph(self, csv_path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Build complete business complementarity graph using complete profile vectors"""
        print("🔨 Building business causal relationship graph with complete profile vectors...")

        # Extract complete business profile vectors for each category
        business_profiles = self._extract_business_profile_vectors(csv_path)
        
        causal_graph = {}
        
        # Process each business category
        for category, profiles in business_profiles.items():
            print(f"\n🏢 Processing {category} category ({len(profiles)} profile vectors)...")
            causal_graph[category] = {}
            
            profiles_list = sorted(list(profiles))
            
            # Create async tasks for all profile combinations
            tasks = []
            profile_pairs = []
            
            for target_profile in profiles_list:
                other_profiles = [p for p in profiles_list if p != target_profile]
                if other_profiles:
                    tasks.append(self.business_analyzer.get_profile_complementarity(
                        target_profile, other_profiles, category
                    ))
                    profile_pairs.append((target_profile, other_profiles))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for (target_profile, other_profiles), result in zip(profile_pairs, results):
                if isinstance(result, Exception):
                    print(f"  ✗ Error processing {target_profile}: {result}")
                    causal_graph[category][target_profile] = {profile: 0.5 for profile in other_profiles}
                else:
                    causal_graph[category][target_profile] = result
                    print(f"  ✓ {target_profile}: got {str(result)} relationships")

        # Cache the complete graph
        print(f"\n💾 Caching complete causal relationship graph...")
        self.cache.set(self.CAUSAL_GRAPH_KEY, json.dumps(causal_graph))
        
        print(f"✅ Business causal relationship graph complete!")
        return causal_graph

    def _extract_business_profile_vectors(self, csv_path: str) -> Dict[str, Set[str]]:
        """Extract complete business profile vectors from dataset"""
        df = pd.read_csv(csv_path)
        
        business_columns = {
            "industry": "Company Identity - Industry Classification",
            "market": "Company Market - Market Traction", 
            "offering": "Company Offering - Value Proposition",
        }
        
        business_profiles = {category: set() for category in business_columns.keys()}
        
        for category, column_name in business_columns.items():
            for _, row in df.iterrows():
                if pd.notna(row[column_name]):
                    # Use the complete cell value as the profile vector
                    profile_vector = str(row[column_name]).strip()
                    if profile_vector:
                        business_profiles[category].add(profile_vector)
        
        print(f"Extracted business profile vectors:")
        for category, profiles in business_profiles.items():
            print(f"  {category}: {len(profiles)} unique profile vectors")
            
        return business_profiles

    async def build_complementarity_matrix(self, csv_path: str, category: str) -> Dict[str, Dict[str, float]]:
        """Build complementarity matrix for complete profile vectors using ChatGPT analysis"""
        print(f"🔨 Building {category} complementarity matrix with complete profiles...")

        # Extract complete profile vectors from dataset
        profile_vectors = self._extract_profile_vectors(csv_path, category)
        cache_key = self._get_cache_key(category)
        
        profile_list = sorted(list(profile_vectors))
        print(f"Found {len(profile_list)} unique {category} profile vectors")
        
        matrix = {}
        
        # Create tasks for all profile vector combinations
        tasks = []
        target_profiles = []
        
        for target_profile in profile_list:
            other_profiles = [p for p in profile_list if p != target_profile]
            if other_profiles:
                tasks.append(self.business_analyzer.get_profile_complementarity(
                    target_profile, other_profiles, category
                ))
                target_profiles.append(target_profile)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for target_profile, result in zip(target_profiles, results):
            if isinstance(result, Exception):
                print(f"  ✗ Error processing {target_profile}: {result}")
                matrix[target_profile] = {profile: 0.5 for profile in profile_list if profile != target_profile}
            else:
                matrix[target_profile] = result
                print(f"  ✓ {target_profile}: {len(result)} relationships")

        # Cache the matrix
        print(f"💾 Caching {category} complementarity matrix...")
        self.cache.set(cache_key, json.dumps(matrix))
        
        print(f"✅ {category.capitalize()} complementarity matrix complete!")
        return matrix

    def _extract_profile_vectors(self, csv_path: str, category: str) -> Set[str]:
        """Extract complete profile vectors from dataset for a category"""
        df = pd.read_csv(csv_path)
        profile_vectors = set()
        
        if category == "role":
            column = "Professional Identity - Role Specification"
        elif category == "experience":
            column = "Professional Identity - Experience Level"
        elif category == "persona":
            column = "All Persona Titles"
        else:
            raise ValueError(f"Unsupported category: {category}")
        
        for _, row in df.iterrows():
            if pd.notna(row[column]):
                # Use the complete cell value as the profile vector
                profile_vector = str(row[column]).strip()
                if profile_vector:
                    profile_vectors.add(profile_vector)
        
        return profile_vectors

    def _get_cache_key(self, category: str) -> str:
        """Get cache key for category"""
        if category == "persona":
            return self.PERSONA_MATRIX_KEY
        elif category == "experience":
            return self.EXPERIENCE_MATRIX_KEY
        elif category == "role":
            return self.ROLE_MATRIX_KEY
        else:
            raise ValueError(f"Unsupported category: {category}")

    def _extract_experience_tags(self, csv_path: str) -> set:
        """Extract experience tags from dataset"""
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        experience_tags = set()
        for _, row in df.iterrows():
            if pd.notna(row["Professional Identity - Experience Level"]):
                tags = [tag.strip() for tag in str(row["Professional Identity - Experience Level"]).split("|")]
                experience_tags.update(tag for tag in tags if tag.strip())
        
        return experience_tags


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

    # Complementarity calculation methods for GraphBuilder
    async def precompute_complementarity_matrices(self, csv_path: str) -> None:
        """Load/build all complementarity matrices and load into memory"""
        
        # Load/build business complementarity matrix
        if not self.load_causal_graph_from_redis():
            print("🔧 Business complementarity matrix not found. Building automatically...")
            try:
                await self.build_causal_relationship_graph(csv_path)
                print("✅ Business complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Error building business matrix: {e}")

        # Load/build experience complementarity matrix
        if not self.load_experience_matrix_from_redis():
            print("🔧 Experience complementarity matrix not found. Building automatically...")
            try:
                await self.build_complementarity_matrix(csv_path, "experience")
                print("✅ Experience complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Error building experience matrix: {e}")

        # Load/build role complementarity matrix
        if not self.load_role_matrix_from_redis():
            print("🔧 Role complementarity matrix not found. Building automatically...")
            try:
                await self.build_complementarity_matrix(csv_path, "role")
                print("✅ Role complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Error building role matrix: {e}")

        # Load/build persona complementarity matrix
        if not self.load_persona_matrix_from_redis():
            print("🔧 Persona complementarity matrix not found. Building automatically...")
            try:
                await self.build_complementarity_matrix(csv_path, "persona")
                print("✅ Persona complementarity matrix built and cached")
            except Exception as e:
                print(f"⚠️ Error building persona matrix: {e}")

        # Load matrices into memory for fast access
        self._load_matrices_from_cache()

    def _load_matrices_from_cache(self) -> None:
        """Load all complementarity matrices from Redis cache into memory"""
        
        # Load business categories from causal graph
        cached_business = self.cache.get(self.CAUSAL_GRAPH_KEY)
        if cached_business:
            try:
                business_data = json.loads(cached_business)
                self._matrices["industry"] = business_data.get("industry", {})
                self._matrices["market"] = business_data.get("market", {})
                self._matrices["offering"] = business_data.get("offering", {})
            except json.JSONDecodeError:
                print(f"⚠️ Invalid cached business matrix data")
                self._matrices["industry"] = {}
                self._matrices["market"] = {}
                self._matrices["offering"] = {}
        else:
            self._matrices["industry"] = {}
            self._matrices["market"] = {}
            self._matrices["offering"] = {}
        
        # Load individual matrices
        individual_matrices = {
            "experience": self.EXPERIENCE_MATRIX_KEY,
            "role": self.ROLE_MATRIX_KEY,
            "persona": self.PERSONA_MATRIX_KEY
        }
        
        for matrix_type, cache_key in individual_matrices.items():
            cached_data = self.cache.get(cache_key)
            if cached_data:
                try:
                    self._matrices[matrix_type] = json.loads(cached_data)
                except json.JSONDecodeError:
                    print(f"⚠️ Invalid cached {matrix_type} matrix data")
                    self._matrices[matrix_type] = {}
            else:
                self._matrices[matrix_type] = {}

    def _get_complementarity_score(self, person_i: int, person_j: int, category: str) -> float:
        """Get complementarity score between two complete profile vectors from cached matrices"""

        person_i_profile = self._get_person_profile_vector(person_i, category)
        person_j_profile = self._get_person_profile_vector(person_j, category)
        
        if not person_i_profile or not person_j_profile:
            return 0.5
            
        matrix = self._matrices.get(category, {})
        
        if person_i_profile in matrix and person_j_profile in matrix[person_i_profile]:
            return matrix[person_i_profile][person_j_profile]
        elif person_j_profile in matrix and person_i_profile in matrix[person_j_profile]:
            return matrix[person_j_profile][person_i_profile]
        
        return 0.5  # Default neutral score

    def _get_person_profile_vector(self, person_idx: int, category: str) -> str:
        """Get complete profile vector for a person in a category"""
        person_data = self._person_tags_cache.get(person_idx, {})
        
        if category == "role":
            return person_data.get("role_profile", "")
        elif category == "experience":
            return person_data.get("experience_profile", "")
        elif category == "persona":
            return person_data.get("persona_profile", "")
        elif category == "industry":
            return person_data.get("industry_profile", "")
        elif category == "market":
            return person_data.get("market_profile", "")
        elif category == "offering":
            return person_data.get("offering_profile", "")
        else:
            return ""

    async def precompute_person_tags(self, df: pd.DataFrame, embedding_builder) -> None:
        """Precompute all person profile vectors for fast lookups"""
        print("⚡ Precomputing person profile vectors...")
        
        for idx, row in df.iterrows():
            business = await embedding_builder.extract_business_tags_for_person(row)
            
            # Store complete profile vectors as they appear in the dataset
            self._person_tags_cache[idx] = {
                # Complete profile vectors (raw cell values)
                "role_profile": str(row.get("Professional Identity - Role Specification", "")).strip(),
                "experience_profile": str(row.get("Professional Identity - Experience Level", "")).strip(),
                "persona_profile": str(row.get("All Persona Titles", "")).strip(),
                
                # Business profile vectors (complete cell values)
                "industry_profile": str(row.get("Company Identity - Industry Classification", "")).strip(),
                "market_profile": str(row.get("Company Market - Market Traction", "")).strip(),
                "offering_profile": str(row.get("Company Offering - Value Proposition", "")).strip(),
                
                # Business tag lists (for backward compatibility)
                "industry": business["industry"],
                "market": business["market"],
                "offering": business["offering"],
                
                # Keep individual tags for backward compatibility if needed
                "roles": tag_extractor.extract_tags(
                    str(row.get("Professional Identity - Role Specification", "")), "role_spec"
                ),
                "experience": tag_extractor.extract_tags(
                    str(row.get("Professional Identity - Experience Level", "")), "experience"
                ),
                "personas": tag_extractor.extract_tags(
                    str(row.get("All Persona Titles", "")), "personas"
                ),
            }

    def get_all_complementarities(self, person_i: int, person_j: int) -> Dict[str, float]:
        """Get all complementarity scores between two people"""

        role_comp = self._get_complementarity_score(person_i, person_j, category="role")
        exp_comp = self._get_complementarity_score(person_i, person_j, category="experience")
        persona_comp = self._get_complementarity_score(person_i, person_j, category="persona")
        industry_comp = self._get_complementarity_score(person_i, person_j, category="industry")
        market_comp = self._get_complementarity_score(person_i, person_j, category="market")
        offering_comp = self._get_complementarity_score(person_i, person_j, category="offering")

        return {
            "role": role_comp,
            "experience": exp_comp,
            "persona": persona_comp,
            "industry": industry_comp,
            "market": market_comp,
            "offering": offering_comp,
        }

    def clear_cache(self) -> None:
        """Clear all cached person tags"""
        self._person_tags_cache.clear()
        self._matrices.clear()