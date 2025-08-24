#!/usr/bin/env python3

import os

from dotenv import load_dotenv


def test_configurable_equation():
    """Demonstrate the configurable Ax + By equation"""

    load_dotenv()

    # Sample scores
    professional_score = 0.8
    business_complementarity = 0.6

    configs = [
        ("0.7", "0.3", "Current (professional focus)"),
        ("0.5", "0.5", "Balanced"),
        ("0.3", "0.7", "Business complementarity focus"),
        ("0.9", "0.1", "Strong professional focus"),
        ("0.1", "0.9", "Strong business focus"),
    ]

    print("Testing Configurable Similarity Equation: Ax + By")
    print(f"Professional Score: {professional_score}")
    print(f"Business Complementarity: {business_complementarity}")
    print("=" * 60)

    for prof_w, bus_w, description in configs:
        # Set environment variables
        os.environ["PROFESSIONAL_WEIGHT"] = prof_w
        os.environ["BUSINESS_COMPLEMENTARITY_WEIGHT"] = bus_w

        # Apply the configurable equation (same as in graph_matcher.py)
        professional_weight = float(os.getenv("PROFESSIONAL_WEIGHT", "0.7"))
        business_weight = float(os.getenv("BUSINESS_COMPLEMENTARITY_WEIGHT", "0.3"))
        hybrid_similarity = (
            professional_weight * professional_score
            + business_weight * business_complementarity
        )

        print(
            f"{description:25} | A={prof_w}, B={bus_w} | Score: {hybrid_similarity:.3f}"
        )

    print("\nThe equation is now configurable via .env hyperparameters!")
    print("You can adjust PROFESSIONAL_WEIGHT and BUSINESS_COMPLEMENTARITY_WEIGHT")
    print("to test different weightings and see which scores better.")


if __name__ == "__main__":
    test_configurable_equation()
