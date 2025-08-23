#!/usr/bin/env python3

import pandas as pd

from graph_matcher import GraphMatcher


def test_small_sample():
    """Test with a small sample of the data"""
    print("Testing with small sample...")

    # Load the full dataset
    df = pd.read_csv("/home/ryan/PycharmProjects/match_engine/data/batch2.csv")

    # Take only first 5 rows for testing
    small_df = df.head(5)
    small_df.to_csv(
        "/home/ryan/PycharmProjects/match_engine/data/test_batch.csv", index=False
    )

    # Test with small dataset
    matcher = GraphMatcher(
        "/home/ryan/PycharmProjects/match_engine/data/test_batch.csv"
    )
    result = matcher.run_analysis()

    print("\n" + "=" * 50)
    print("TEST RESULTS (5 people)")
    print("=" * 50)

    if result["people"]:
        for i, person in enumerate(result["people"], 1):
            print(f"\n{i}. {person['name']}")
            print(f"   Title: {person['title']}")
            print(f"   Company: {person['company']}")
    else:
        print("No dense subgraph found above the minimum density threshold.")


if __name__ == "__main__":
    test_small_sample()
