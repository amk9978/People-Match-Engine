#!/usr/bin/env python3

import asyncio
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from graph_matcher import GraphMatcher


async def test_edge_scoring_impact():
    """Test impact of expanded 5-component edge scoring vs 2-component"""

    load_dotenv()

    print("TESTING EDGE SCORING IMPACT ON SUBGRAPH QUALITY")
    print("=" * 60)

    # Create small test sample for faster testing
    df = pd.read_csv("data/batch2.csv")
    small_sample = df.head(10)  # 10 people for manageable testing
    small_sample.to_csv("test_edge_scoring.csv", index=False)

    # Test configurations
    configs = [
        {
            "name": "Original 2-Component",
            "weights": {
                "ROLE_SIMILARITY_WEIGHT": "0.70",  # Combined as professional_score
                "EXPERIENCE_COMPLEMENTARITY_WEIGHT": "0.00",
                "BUSINESS_COMPLEMENTARITY_WEIGHT": "0.30",
                "PERSONA_COMPLEMENTARITY_WEIGHT": "0.00",
                "INDUSTRY_CROSS_SECTOR_WEIGHT": "0.00",
            },
        },
        {
            "name": "Expanded 5-Component (Current)",
            "weights": {
                "ROLE_SIMILARITY_WEIGHT": "0.30",
                "EXPERIENCE_COMPLEMENTARITY_WEIGHT": "0.15",
                "BUSINESS_COMPLEMENTARITY_WEIGHT": "0.25",
                "PERSONA_COMPLEMENTARITY_WEIGHT": "0.15",
                "INDUSTRY_CROSS_SECTOR_WEIGHT": "0.15",
            },
        },
        {
            "name": "Business-Heavy 5-Component",
            "weights": {
                "ROLE_SIMILARITY_WEIGHT": "0.20",
                "EXPERIENCE_COMPLEMENTARITY_WEIGHT": "0.15",
                "BUSINESS_COMPLEMENTARITY_WEIGHT": "0.40",
                "PERSONA_COMPLEMENTARITY_WEIGHT": "0.15",
                "INDUSTRY_CROSS_SECTOR_WEIGHT": "0.10",
            },
        },
    ]

    results = []

    for config in configs:
        print(f"\\n{'-'*40}")
        print(f"Testing: {config['name']}")
        print(f"Weights: {config['weights']}")

        # Set environment variables
        for param, value in config["weights"].items():
            os.environ[param] = value

        # Create matcher
        matcher = GraphMatcher("test_edge_scoring.csv")
        matcher.load_data()

        # Create dummy feature embeddings for testing
        feature_embeddings = {}
        num_people = len(matcher.df)
        embedding_dim = 1536

        for feature in [
            "role_spec",
            "experience",
            "industry",
            "market",
            "offering",
            "personas",
        ]:
            feature_embeddings[feature] = np.random.rand(num_people, embedding_dim)

        # Create graph with current configuration
        graph = matcher.create_graph(feature_embeddings)

        # Calculate graph metrics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        if num_edges > 0:
            # Edge weight statistics
            edge_weights = [d["weight"] for u, v, d in graph.edges(data=True)]
            avg_weight = np.mean(edge_weights)
            max_weight = np.max(edge_weights)
            min_weight = np.min(edge_weights)
            weight_std = np.std(edge_weights)

            # High-quality connections (>0.6)
            high_quality_edges = sum(1 for w in edge_weights if w > 0.6)
            high_quality_ratio = high_quality_edges / num_edges

            print(f"Graph: {num_nodes} nodes, {num_edges} edges")
            print(
                f"Edge weights: avg={avg_weight:.3f}, max={max_weight:.3f}, min={min_weight:.3f}, std={weight_std:.3f}"
            )
            print(
                f"High-quality edges (>0.6): {high_quality_edges} ({high_quality_ratio:.1%})"
            )

            # Show sample edge breakdowns
            print("Sample edge component breakdowns:")
            sample_edges = list(graph.edges(data=True))[:3]
            for u, v, data in sample_edges:
                print(f"  Edge ({u},{v}): total={data['weight']:.3f}")
                if "role_similarity" in data:
                    print(
                        f"    role={data.get('role_similarity', 0):.3f}, "
                        + f"exp_comp={data.get('experience_complementarity', 0):.3f}, "
                        + f"biz_comp={data.get('business_complementarity', 0):.3f}, "
                        + f"persona_comp={data.get('persona_complementarity', 0):.3f}, "
                        + f"industry_cross={data.get('industry_cross_sector', 0):.3f}"
                    )
        else:
            avg_weight = max_weight = min_weight = weight_std = 0
            high_quality_ratio = 0
            print(f"Graph: {num_nodes} nodes, {num_edges} edges (no edges created)")

        results.append(
            {
                "name": config["name"],
                "edges": num_edges,
                "avg_weight": avg_weight,
                "max_weight": max_weight,
                "high_quality_ratio": high_quality_ratio,
                "weight_std": weight_std,
            }
        )

    # Summary comparison
    print(f"\\n{'='*60}")
    print("EDGE SCORING COMPARISON SUMMARY")
    print(f"{'='*60}")

    for result in results:
        print(
            f"{result['name']:25} | "
            + f"Edges: {result['edges']:2} | "
            + f"Avg: {result['avg_weight']:.3f} | "
            + f"Max: {result['max_weight']:.3f} | "
            + f"HQ%: {result['high_quality_ratio']:.1%} | "
            + f"Std: {result['weight_std']:.3f}"
        )

    print(f"\\nKey Insights:")
    print("- More edges = better connectivity for subgraph extraction")
    print("- Higher avg weight = stronger connections overall")
    print("- Higher HQ% = more meaningful high-quality matches")
    print("- Lower std = more consistent edge quality")

    return results


if __name__ == "__main__":
    asyncio.run(test_edge_scoring_impact())
