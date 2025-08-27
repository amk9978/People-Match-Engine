#!/usr/bin/env python3

import asyncio
import os

import pandas as pd
from dotenv import load_dotenv

from services.graph_matcher import GraphMatcher


async def test_different_weights():
    """Test different hyperparameter configurations"""

    # Create small test dataset
    df = pd.read_csv("data/batch2.csv")
    small_sample = df.head(5)
    small_sample.to_csv("test_sample.csv", index=False)

    # Test configurations
    configs = [
        {
            "PROFESSIONAL_WEIGHT": "0.7",
            "BUSINESS_COMPLEMENTARITY_WEIGHT": "0.3",
        },  # Current
        {
            "PROFESSIONAL_WEIGHT": "0.8",
            "BUSINESS_COMPLEMENTARITY_WEIGHT": "0.2",
        },  # More professional
        {
            "PROFESSIONAL_WEIGHT": "0.5",
            "BUSINESS_COMPLEMENTARITY_WEIGHT": "0.5",
        },  # Balanced
        {
            "PROFESSIONAL_WEIGHT": "0.3",
            "BUSINESS_COMPLEMENTARITY_WEIGHT": "0.7",
        },  # More business
    ]

    results = []

    for i, config in enumerate(configs):
        print(f"\n{'='*50}")
        print(
            f"Testing Config {i+1}: Prof={config['PROFESSIONAL_WEIGHT']}, Bus={config['BUSINESS_COMPLEMENTARITY_WEIGHT']}"
        )
        print(f"{'='*50}")

        # Set environment variables
        for key, value in config.items():
            os.environ[key] = value

        # Create matcher and test
        matcher = GraphMatcher("test_sample.csv")
        await matcher.load_data()

        # We need to create feature embeddings first - let's create a minimal version
        feature_embeddings = {
            "role_spec": [],
            "experience": [],
            "industry": [],
            "market": [],
            "offering": [],
            "personas": [],
        }

        # Use dummy embeddings for testing weights
        import numpy as np

        for key in feature_embeddings:
            feature_embeddings[key] = np.random.rand(5, 1536)  # 5 people, 1536 dims

        graph = matcher.create_graph(feature_embeddings)

        edge_count = graph.number_of_edges()
        avg_weight = (
            sum(d["weight"] for u, v, d in graph.edges(data=True)) / edge_count
            if edge_count > 0
            else 0
        )

        print(f"Edges: {edge_count}, Avg Weight: {avg_weight:.3f}")

        # Show sample weights
        edges = list(graph.edges(data=True))[:3]
        for u, v, data in edges:
            print(f"  Edge ({u},{v}): {data['weight']:.3f}")

        results.append(
            {
                "config": f"P{config['PROFESSIONAL_WEIGHT']}_B{config['BUSINESS_COMPLEMENTARITY_WEIGHT']}",
                "edges": edge_count,
                "avg_weight": avg_weight,
            }
        )

    print(f"\n{'='*50}")
    print("HYPERPARAMETER COMPARISON SUMMARY")
    print(f"{'='*50}")
    for result in results:
        print(
            f"{result['config']}: {result['edges']} edges, avg weight {result['avg_weight']:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(test_different_weights())
