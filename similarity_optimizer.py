#!/usr/bin/env python3

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from expanded_similarity_framework import ExpandedSimilarityFramework


@dataclass
class OptimizationResult:
    weights: Dict[str, float]
    score: float
    matches_evaluated: int
    detailed_scores: Dict[str, float]


class SimilarityOptimizer:
    """LLM-as-Judge optimization for similarity equation hyperparameters"""

    def __init__(self):
        load_dotenv()
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.framework = ExpandedSimilarityFramework()

    async def evaluate_match_quality_llm(
        self, person1_data: Dict, person2_data: Dict, similarity_score: float
    ) -> Dict[str, Any]:
        """Use LLM to evaluate the quality of a match"""

        prompt = f"""You are evaluating the quality of a professional match between two people.

PERSON 1:
- Role: {person1_data.get('role_spec', 'N/A')}
- Experience: {person1_data.get('experience', 'N/A')}
- Industry: {person1_data.get('industry', 'N/A')}
- Market: {person1_data.get('market', 'N/A')} 
- Offering: {person1_data.get('offering', 'N/A')}
- Personas: {person1_data.get('personas', 'N/A')}

PERSON 2:
- Role: {person2_data.get('role_spec', 'N/A')}
- Experience: {person2_data.get('experience', 'N/A')}
- Industry: {person2_data.get('industry', 'N/A')}
- Market: {person2_data.get('market', 'N/A')}
- Offering: {person2_data.get('offering', 'N/A')}
- Personas: {person2_data.get('personas', 'N/A')}

SIMILARITY SCORE: {similarity_score:.3f}

Rate this match on the following criteria (0-10 scale):

1. Professional_Synergy: How well do their roles complement or align?
2. Business_Value: How much business value could this partnership create?  
3. Learning_Opportunity: How much could they learn from each other?
4. Network_Strength: How valuable would this connection be long-term?
5. Overall_Quality: Overall assessment of match quality

Also provide:
6. Match_Type: Is this primarily a "networking" match (similar professionals) or "partnership" match (complementary capabilities)?

Respond ONLY in JSON format:
{{
    "professional_synergy": 8,
    "business_value": 7,
    "learning_opportunity": 6, 
    "network_strength": 8,
    "overall_quality": 7,
    "match_type": "networking"
}}"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            result = json.loads(response.choices[0].message.content.strip())
            return result

        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return {
                "professional_synergy": 5,
                "business_value": 5,
                "learning_opportunity": 5,
                "network_strength": 5,
                "overall_quality": 5,
                "match_type": "unknown",
            }

    async def evaluate_weight_configuration(
        self, weights: Dict[str, float], sample_matches: List[Tuple]
    ) -> OptimizationResult:
        """Evaluate a specific weight configuration using LLM-as-Judge"""

        # Set environment variables for this configuration
        for param, value in weights.items():
            os.environ[param] = str(value)

        evaluation_scores = []
        detailed_scores = {
            "professional_synergy": [],
            "business_value": [],
            "learning_opportunity": [],
            "network_strength": [],
            "overall_quality": [],
        }

        print(f"Evaluating weights: {weights}")

        for i, (person1, person2, components) in enumerate(sample_matches):
            if i >= 10:  # Limit to 10 matches for cost control
                break

            # Calculate similarity with current weights
            similarity = self.framework.calculate_expanded_similarity(**components)

            # Get LLM evaluation
            llm_eval = await self.evaluate_match_quality_llm(
                person1, person2, similarity
            )

            # Collect scores
            evaluation_scores.append(llm_eval["overall_quality"])
            for metric in detailed_scores.keys():
                detailed_scores[metric].append(llm_eval[metric])

            print(
                f"  Match {i+1}: Similarity={similarity:.3f}, LLM_Quality={llm_eval['overall_quality']}"
            )

        # Calculate aggregate score
        avg_score = np.mean(evaluation_scores)
        detailed_averages = {k: np.mean(v) for k, v in detailed_scores.items()}

        return OptimizationResult(
            weights=weights,
            score=avg_score,
            matches_evaluated=len(evaluation_scores),
            detailed_scores=detailed_averages,
        )

    def generate_candidate_weights(self) -> List[Dict[str, float]]:
        """Generate candidate weight configurations to test"""

        candidates = [
            # Current configuration
            {
                "ROLE_SIMILARITY_WEIGHT": 0.30,
                "EXPERIENCE_COMPLEMENTARITY_WEIGHT": 0.15,
                "BUSINESS_COMPLEMENTARITY_WEIGHT": 0.25,
                "PERSONA_COMPLEMENTARITY_WEIGHT": 0.15,
                "INDUSTRY_CROSS_SECTOR_WEIGHT": 0.15,
            },
            # Role-focused
            {
                "ROLE_SIMILARITY_WEIGHT": 0.50,
                "EXPERIENCE_COMPLEMENTARITY_WEIGHT": 0.10,
                "BUSINESS_COMPLEMENTARITY_WEIGHT": 0.20,
                "PERSONA_COMPLEMENTARITY_WEIGHT": 0.10,
                "INDUSTRY_CROSS_SECTOR_WEIGHT": 0.10,
            },
            # Business-focused
            {
                "ROLE_SIMILARITY_WEIGHT": 0.20,
                "EXPERIENCE_COMPLEMENTARITY_WEIGHT": 0.15,
                "BUSINESS_COMPLEMENTARITY_WEIGHT": 0.40,
                "PERSONA_COMPLEMENTARITY_WEIGHT": 0.15,
                "INDUSTRY_CROSS_SECTOR_WEIGHT": 0.10,
            },
            # Balanced complementarity
            {
                "ROLE_SIMILARITY_WEIGHT": 0.25,
                "EXPERIENCE_COMPLEMENTARITY_WEIGHT": 0.20,
                "BUSINESS_COMPLEMENTARITY_WEIGHT": 0.25,
                "PERSONA_COMPLEMENTARITY_WEIGHT": 0.20,
                "INDUSTRY_CROSS_SECTOR_WEIGHT": 0.10,
            },
            # Cross-sector focused
            {
                "ROLE_SIMILARITY_WEIGHT": 0.25,
                "EXPERIENCE_COMPLEMENTARITY_WEIGHT": 0.15,
                "BUSINESS_COMPLEMENTARITY_WEIGHT": 0.20,
                "PERSONA_COMPLEMENTARITY_WEIGHT": 0.15,
                "INDUSTRY_CROSS_SECTOR_WEIGHT": 0.25,
            },
        ]

        return candidates


def create_sample_matches() -> List[Tuple]:
    """Create sample matches for testing (mock data for demonstration)"""

    sample_matches = [
        # Match 1: Similar roles, different experience
        (
            {
                "role_spec": "CTO",
                "experience": "Senior",
                "industry": "Technology",
                "market": "Growth",
                "offering": "AI Platform",
                "personas": "Tech Leader",
            },
            {
                "role_spec": "VP Engineering",
                "experience": "Mid-level",
                "industry": "Technology",
                "market": "Early",
                "offering": "Dev Tools",
                "personas": "Engineering Manager",
            },
            {
                "role_similarity": 0.8,
                "experience_complementarity": 0.6,
                "business_complementarity": 0.7,
                "persona_complementarity": 0.5,
                "industry_cross_sector": 0.2,
            },
        ),
        # Match 2: Different roles, complementary business
        (
            {
                "role_spec": "CEO",
                "experience": "Senior",
                "industry": "Fintech",
                "market": "Scale",
                "offering": "Banking Platform",
                "personas": "Business Leader",
            },
            {
                "role_spec": "Data Scientist",
                "experience": "Senior",
                "industry": "Healthcare",
                "market": "Growth",
                "offering": "ML Analytics",
                "personas": "Technical Expert",
            },
            {
                "role_similarity": 0.3,
                "experience_complementarity": 0.2,
                "business_complementarity": 0.9,
                "persona_complementarity": 0.8,
                "industry_cross_sector": 0.7,
            },
        ),
        # Add more sample matches as needed
    ]

    return sample_matches


async def run_optimization():
    """Run the similarity equation optimization"""

    optimizer = SimilarityOptimizer()
    sample_matches = create_sample_matches()

    print("SIMILARITY EQUATION OPTIMIZATION")
    print("=" * 50)
    print(
        f"Testing {len(optimizer.generate_candidate_weights())} weight configurations"
    )
    print(f"Using {len(sample_matches)} sample matches per configuration")
    print("=" * 50)

    results = []

    for i, weights in enumerate(optimizer.generate_candidate_weights()):
        print(f"\nConfiguration {i+1}:")
        result = await optimizer.evaluate_weight_configuration(weights, sample_matches)
        results.append(result)

        print(f"Average LLM Score: {result.score:.2f}/10")
        print(
            "Detailed scores:",
            {k: f"{v:.2f}" for k, v in result.detailed_scores.items()},
        )

    # Find best configuration
    best_result = max(results, key=lambda r: r.score)

    print(f"\n{'=' * 50}")
    print("OPTIMIZATION RESULTS")
    print(f"{'=' * 50}")
    print(f"Best configuration (Score: {best_result.score:.2f}/10):")
    for param, weight in best_result.weights.items():
        print(f"  {param}: {weight:.2f}")

    print(f"\nDetailed best scores:")
    for metric, score in best_result.detailed_scores.items():
        print(f"  {metric}: {score:.2f}/10")

    return best_result


if __name__ == "__main__":
    asyncio.run(run_optimization())
