#!/usr/bin/env python3

import asyncio

from causal_relationship_analyzer import CausalRelationshipAnalyzer


async def build_persona_matrix():
    """Build persona complementarity matrix using ChatGPT"""

    analyzer = CausalRelationshipAnalyzer()

    print("Building persona complementarity matrix...")
    matrix = await analyzer.build_persona_complementarity_matrix("data/batch2.csv")

    print(f"\n✅ Built persona matrix with {len(matrix)} personas")

    # Show some example complementarity scores
    print("\nExample persona complementarity scores:")
    persona_list = list(matrix.keys())[:5]  # First 5 personas

    for i, persona1 in enumerate(persona_list):
        for j, persona2 in enumerate(persona_list):
            if i < j:  # Only show unique pairs
                score = matrix[persona1].get(persona2, 0.0)
                print(f"  {persona1} + {persona2}: {score:.2f}")

    print(f"\n✅ Matrix cached in Redis for fast lookup")
    return analyzer


if __name__ == "__main__":
    asyncio.run(build_persona_matrix())
