import logging
import sys
from typing import Dict, Set

import numpy as np
import pandas as pd

from services.analysis.business_analyzer import BusinessAnalyzer
from services.analysis.dataset_insights import DatasetInsightsAnalyzer
from services.preprocessing.tag_extractor import tag_extractor
from services.redis.app_cache_service import app_cache_service
from shared.shared import (
    BUSINESS_FEATURES,
    FEATURE_COLUMN_MAPPING,
    FEATURES,
    FeatureNames,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class MatrixBuilder:
    """Handles building complementarity matrices using row-based caching via AppCacheService"""

    def __init__(
        self,
        cache=None,
        business_analyzer: BusinessAnalyzer = None,
        insight_analyzer: DatasetInsightsAnalyzer = None,
    ):
        self.cache = cache or app_cache_service
        self.business_analyzer = business_analyzer or BusinessAnalyzer()
        self.insight_analyzer = insight_analyzer or DatasetInsightsAnalyzer()
        self._matrices = {}
        self._person_tags_cache = {}

    async def build_causal_relationship_graph(
        self, csv_path: str
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Build complete business complementarity graph using row-based caching"""
        logger.info(
            "🔨 Building business causal relationship graph with row-based caching..."
        )

        # Extract complete business profile vectors for each category
        business_profiles = self._extract_business_profile_vectors(csv_path)
        causal_graph = {}

        # Process each business category
        for category, profiles in business_profiles.items():
            logger.info(
                f"\n🏢 Processing {category} category ({len(profiles)} profile vectors)..."
            )
            causal_graph[category] = {}

            profiles_list = sorted(list(profiles))
            other_profiles = [p for p in profiles_list]  # All profiles for comparison

            # Use batch processing with AppCacheService handling caching
            logger.info(
                f"  🚀 Using batch processing for {len(profiles_list)} {category} profiles"
            )
            batch_results = await self.business_analyzer.get_profile_complementarity(
                profiles_list, other_profiles, category
            )

            # Process batch results
            for target_profile in profiles_list:
                if target_profile in batch_results:
                    batch_result = batch_results[target_profile]

                    # Handle malformed batch result (should be dict, not float)
                    if isinstance(batch_result, dict):
                        # Remove self-comparison (set to 0)
                        profile_scores = batch_result.copy()
                        if target_profile in profile_scores:
                            profile_scores[target_profile] = 0.0  # Self-comparison

                        causal_graph[category][target_profile] = profile_scores
                        logger.info(
                            f"  ✓ {target_profile}: got {len(profile_scores)} relationships"
                        )
                    else:
                        # Fallback for malformed batch result (float instead of dict)
                        logger.warning(
                            f"  ⚠️ Batch result for {target_profile} is {type(batch_result).__name__}, expected dict. Using fallback."
                        )
                        other_profiles_only = [
                            p for p in profiles_list if p != target_profile
                        ]
                        causal_graph[category][target_profile] = {
                            profile: 0.5 for profile in other_profiles_only
                        }
                        logger.info(f"  ⚠️ {target_profile}: using fallback scores")
                else:
                    # Fallback for missing profiles
                    other_profiles_only = [
                        p for p in profiles_list if p != target_profile
                    ]
                    causal_graph[category][target_profile] = {
                        profile: 0.5 for profile in other_profiles_only
                    }
                    logger.info(f"  ⚠️ {target_profile}: using fallback scores")

        logger.info(f"✅ Business causal relationship graph complete!")
        return causal_graph

    def _extract_business_profile_vectors(self, csv_path: str) -> Dict[str, Set[str]]:
        """Extract complete business profile vectors from dataset"""
        df = pd.read_csv(csv_path)

        business_columns = {
            feature: FEATURE_COLUMN_MAPPING[feature] for feature in BUSINESS_FEATURES
        }

        business_profiles = {category: set() for category in business_columns.keys()}

        for category, column_name in business_columns.items():
            for _, row in df.iterrows():
                if pd.notna(row[column_name]):
                    # Use the complete cell value as the profile vector
                    profile_vector = str(row[column_name]).strip()
                    if profile_vector:
                        business_profiles[category].add(profile_vector)

        logger.info(f"Extracted business profile vectors:")
        for category, profiles in business_profiles.items():
            logger.info(f"  {category}: {len(profiles)} unique profile vectors")

        return business_profiles

    async def build_complementarity_matrix(
        self, csv_path: str, category: str
    ) -> Dict[str, Dict[str, float]]:
        """Build complementarity matrix for complete profile vectors using row-based caching"""
        logger.info(
            f"🔨 Building {category} complementarity matrix with row-based caching..."
        )

        # Extract complete profile vectors from dataset
        profile_vectors = self._extract_profile_vectors(csv_path, category)
        profile_list = sorted(list(profile_vectors))
        logger.info(f"Found {len(profile_list)} unique {category} profile vectors")

        matrix = {}

        # Use batch processing with AppCacheService handling caching
        logger.info(
            f"  🚀 Using batch processing for {len(profile_list)} {category} profiles"
        )
        batch_results = await self.business_analyzer.get_profile_complementarity(
            profile_list, profile_list, category
        )

        # Process batch results
        for target_profile in profile_list:
            if target_profile in batch_results:
                batch_result = batch_results[target_profile]

                # Handle malformed batch result (should be dict, not float)
                if isinstance(batch_result, dict):
                    # Remove self-comparison (set to 0)
                    profile_scores = batch_result.copy()
                    if target_profile in profile_scores:
                        profile_scores[target_profile] = 0.0  # Self-comparison

                    matrix[target_profile] = profile_scores
                    logger.info(
                        f"  ✓ {target_profile}: got {len(profile_scores)} relationships"
                    )
                else:
                    # Fallback for malformed batch result (float instead of dict)
                    logger.warning(
                        f"  ⚠️ Batch result for {target_profile} is {type(batch_result).__name__}, expected dict. Using fallback."
                    )
                    other_profiles_only = [
                        p for p in profile_list if p != target_profile
                    ]
                    matrix[target_profile] = {
                        profile: 0.5 for profile in other_profiles_only
                    }
                    logger.info(f"  ⚠️ {target_profile}: using fallback scores")
            else:
                # Fallback for missing profiles
                other_profiles_only = [p for p in profile_list if p != target_profile]
                matrix[target_profile] = {
                    profile: 0.5 for profile in other_profiles_only
                }
                logger.info(f"  ⚠️ {target_profile}: using fallback scores")

        logger.info(f"✅ {category.capitalize()} complementarity matrix complete!")
        return matrix

    def _extract_profile_vectors(self, csv_path: str, category: str) -> Set[str]:
        """Extract complete profile vectors from dataset for a category"""
        df = pd.read_csv(csv_path)
        profile_vectors = set()

        if category == "role":
            column = "Professional Identity - Role Specification"
        elif category == FeatureNames.EXPERIENCE.value:
            column = "Professional Identity - Experience Level"
        elif category == FeatureNames.PERSONA.value:
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

    async def build_all_complementarity_matrices(
        self, csv_path: str
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Build all complementarity matrices using row-based caching via AppCacheService"""
        logger.info(
            "🔨 Building all complementarity matrices with row-based caching..."
        )

        all_matrices = {}

        # Build business complementarity matrices (industry, market, offering)
        logger.info("Building business matrices...")
        business_matrices = await self.build_causal_relationship_graph(csv_path)
        all_matrices.update(business_matrices)

        # Build individual complementarity matrices
        for category in [
            FeatureNames.EXPERIENCE.value,
            "role",
            FeatureNames.PERSONA.value,
        ]:
            logger.info(f"Building {category} matrix...")
            all_matrices[category] = await self.build_complementarity_matrix(
                csv_path, category
            )

        logger.info("✅ All complementarity matrices built with row-based caching")
        return all_matrices

    def load_matrices_into_memory(
        self, matrices: Dict[str, Dict[str, Dict[str, float]]]
    ) -> None:
        """Load pre-built matrices into memory for fast access during graph building"""
        logger.info("Loading matrices into memory for fast access...")
        self._matrices.clear()

        # Load all provided matrices
        for category, matrix_data in matrices.items():
            self._matrices[category] = matrix_data
            logger.info(f"Loaded {category} matrix with {len(matrix_data)} profiles")

        logger.info("✅ All matrices loaded into memory")

    def get_complementarity_matrices(self) -> Dict[str, np.ndarray]:
        """Convert complementarity matrices to numpy arrays for analysis"""
        numpy_matrices = {}

        for feature in FEATURES:
            if feature in self._matrices:
                matrix_dict = self._matrices[feature]

                people_names = list(matrix_dict.keys())
                size = len(people_names)
                numpy_matrix = np.zeros((size, size))

                for i, person_i in enumerate(people_names):
                    for j, person_j in enumerate(people_names):
                        if person_j in matrix_dict[person_i]:
                            numpy_matrix[i, j] = matrix_dict[person_i][person_j]

                normalized_matrix = self.insight_analyzer.normalize_matrix(
                    numpy_matrix, preserve_diagonal=False
                )
                numpy_matrices[feature] = normalized_matrix

        return numpy_matrices

    def _get_complementarity_score(
        self, person_i: int, person_j: int, category: str
    ) -> float:
        """Get complementarity score between two complete profile vectors from cached matrices"""

        person_i_profile = self._get_person_profile_vector(person_i, category)
        person_j_profile = self._get_person_profile_vector(person_j, category)

        if not person_i_profile or not person_j_profile:
            return 0.5

        matrix = self._matrices.get(category, {})

        if person_i_profile in matrix and person_j_profile in matrix[person_i_profile]:
            return matrix[person_i_profile][person_j_profile]
        elif (
            person_j_profile in matrix and person_i_profile in matrix[person_j_profile]
        ):
            return matrix[person_j_profile][person_i_profile]

        return 0.5  # Default neutral score

    def _get_person_profile_vector(self, person_idx: int, category: str) -> str:
        """Get complete profile vector for a person in a category"""
        person_data = self._person_tags_cache.get(person_idx, {})

        if category == FeatureNames.ROLE.value:
            return person_data.get("role_profile", "")
        elif category == FeatureNames.EXPERIENCE.value:
            return person_data.get("experience_profile", "")
        elif category == FeatureNames.PERSONA.value:
            return person_data.get("persona_profile", "")
        elif category == FeatureNames.INDUSTRY.value:
            return person_data.get("industry_profile", "")
        elif category == FeatureNames.MARKET.value:
            return person_data.get("market_profile", "")
        elif category == FeatureNames.OFFERING.value:
            return person_data.get("offering_profile", "")
        else:
            return ""

    async def precompute_person_tags(self, df: pd.DataFrame, embedding_builder) -> None:
        """Precompute all person profile vectors for fast lookups"""
        logger.info("⚡ Precomputing person profile vectors...")

        for idx, row in df.iterrows():
            business = await embedding_builder.extract_business_tags_for_person(row)

            # Store complete profile vectors as they appear in the dataset
            self._person_tags_cache[idx] = {
                # Complete profile vectors (raw cell values)
                "role_profile": str(
                    row.get("Professional Identity - Role Specification", "")
                ).strip(),
                "experience_profile": str(
                    row.get("Professional Identity - Experience Level", "")
                ).strip(),
                "persona_profile": str(row.get("All Persona Titles", "")).strip(),
                # Business profile vectors (complete cell values)
                "industry_profile": str(
                    row.get("Company Identity - Industry Classification", "")
                ).strip(),
                "market_profile": str(
                    row.get("Company Market - Market Traction", "")
                ).strip(),
                "offering_profile": str(
                    row.get("Company Offering - Value Proposition", "")
                ).strip(),
                # Business tag lists (for backward compatibility)
                FeatureNames.INDUSTRY.value: business[FeatureNames.INDUSTRY.value],
                FeatureNames.MARKET.value: business[FeatureNames.MARKET.value],
                FeatureNames.OFFERING.value: business[FeatureNames.OFFERING.value],
                # Keep individual tags for backward compatibility if needed
                "roles": tag_extractor.extract_tags(
                    str(row.get("Professional Identity - Role Specification", "")),
                    "role",
                ),
                FeatureNames.EXPERIENCE.value: tag_extractor.extract_tags(
                    str(row.get("Professional Identity - Experience Level", "")),
                    FeatureNames.EXPERIENCE.value,
                ),
                "personas": tag_extractor.extract_tags(
                    str(row.get("All Persona Titles", "")), "personas"
                ),
            }

    def get_all_complementarities(
        self, person_i: int, person_j: int
    ) -> Dict[str, float]:
        """Get all complementarity scores between two people"""

        role_comp = self._get_complementarity_score(person_i, person_j, category="role")
        exp_comp = self._get_complementarity_score(
            person_i, person_j, category=FeatureNames.EXPERIENCE.value
        )
        persona_comp = self._get_complementarity_score(
            person_i, person_j, category=FeatureNames.PERSONA.value
        )
        industry_comp = self._get_complementarity_score(
            person_i, person_j, category=FeatureNames.INDUSTRY.value
        )
        market_comp = self._get_complementarity_score(
            person_i, person_j, category=FeatureNames.MARKET.value
        )
        offering_comp = self._get_complementarity_score(
            person_i, person_j, category=FeatureNames.OFFERING.value
        )

        return {
            "role": role_comp,
            FeatureNames.EXPERIENCE.value: exp_comp,
            FeatureNames.PERSONA.value: persona_comp,
            FeatureNames.INDUSTRY.value: industry_comp,
            FeatureNames.MARKET.value: market_comp,
            FeatureNames.OFFERING.value: offering_comp,
        }

    def clear_cache(self) -> None:
        """Clear all cached person tags"""
        self._person_tags_cache.clear()
        self._matrices.clear()
