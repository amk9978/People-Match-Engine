#!/usr/bin/env python3

import json
import os
from typing import Dict, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


class HyperparameterTuner:
    """ChatGPT-based hyperparameter tuning based on user intent"""

    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Default hyperparameters from .env
        self.default_weights = {
            "ROLE_SIMILARITY_WEIGHT": float(
                os.getenv("ROLE_SIMILARITY_WEIGHT", "0.30")
            ),
            "EXPERIENCE_SIMILARITY_WEIGHT": float(
                os.getenv("EXPERIENCE_SIMILARITY_WEIGHT", "0.15")
            ),
            "EXPERIENCE_COMPLEMENTARITY_WEIGHT": float(
                os.getenv("EXPERIENCE_COMPLEMENTARITY_WEIGHT", "0.15")
            ),
            "BUSINESS_COMPLEMENTARITY_WEIGHT": float(
                os.getenv("BUSINESS_COMPLEMENTARITY_WEIGHT", "0.25")
            ),
            "PERSONA_COMPLEMENTARITY_WEIGHT": float(
                os.getenv("PERSONA_COMPLEMENTARITY_WEIGHT", "0.15")
            ),
        }

    async def tune_hyperparameters_for_intent(
        self, user_prompt: str
    ) -> Dict[str, float]:
        """Use ChatGPT to adjust hyperparameters based on user intent"""

        prompt = f"""You are a hyperparameter optimization expert for a professional matching system. 

The system uses a 5-component similarity equation to score connections between professionals:
- ROLE_SIMILARITY_WEIGHT: How much to weight similar professional roles (CEO-CEO, Dev-Dev networking)
- EXPERIENCE_SIMILARITY_WEIGHT: How much to weight similar experience levels (Senior-Senior collaboration)  
- EXPERIENCE_COMPLEMENTARITY_WEIGHT: How much to weight different experience levels (Senior-Junior mentorship)
- BUSINESS_COMPLEMENTARITY_WEIGHT: How much to weight strategic business synergies (different markets/offerings that work together)
- PERSONA_COMPLEMENTARITY_WEIGHT: How much to weight complementary personas (Product Manager + Engineer partnerships)

CURRENT DEFAULT WEIGHTS:
- ROLE_SIMILARITY_WEIGHT: {self.default_weights["ROLE_SIMILARITY_WEIGHT"]}
- EXPERIENCE_SIMILARITY_WEIGHT: {self.default_weights["EXPERIENCE_SIMILARITY_WEIGHT"]}
- EXPERIENCE_COMPLEMENTARITY_WEIGHT: {self.default_weights["EXPERIENCE_COMPLEMENTARITY_WEIGHT"]}
- BUSINESS_COMPLEMENTARITY_WEIGHT: {self.default_weights["BUSINESS_COMPLEMENTARITY_WEIGHT"]}
- PERSONA_COMPLEMENTARITY_WEIGHT: {self.default_weights["PERSONA_COMPLEMENTARITY_WEIGHT"]}

USER REQUEST: "{user_prompt}"

Based on the user's intent, adjust these weights to better match their goals. The weights must sum to 1.0.

EXAMPLES:
- If user wants "hiring/recruitment" → increase EXPERIENCE_COMPLEMENTARITY_WEIGHT and PERSONA_COMPLEMENTARITY_WEIGHT (looking for different skills)
- If user wants "peer networking" → increase ROLE_SIMILARITY_WEIGHT and EXPERIENCE_SIMILARITY_WEIGHT (similar professionals)  
- If user wants "business partnerships" → increase BUSINESS_COMPLEMENTARITY_WEIGHT (strategic synergies)
- If user wants "mentorship connections" → increase EXPERIENCE_COMPLEMENTARITY_WEIGHT (senior-junior pairs)
- If user wants "team building" → increase PERSONA_COMPLEMENTARITY_WEIGHT (diverse skill sets)

Respond ONLY with a JSON object containing the adjusted weights:
{{
    "ROLE_SIMILARITY_WEIGHT": 0.25,
    "EXPERIENCE_SIMILARITY_WEIGHT": 0.10,
    "EXPERIENCE_COMPLEMENTARITY_WEIGHT": 0.25,
    "BUSINESS_COMPLEMENTARITY_WEIGHT": 0.20,
    "PERSONA_COMPLEMENTARITY_WEIGHT": 0.20
}}"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            tuned_weights = json.loads(result_text)

            # Validate weights sum to 1.0
            total = sum(tuned_weights.values())
            if abs(total - 1.0) > 0.01:
                # Normalize if slightly off due to rounding
                for key in tuned_weights:
                    tuned_weights[key] = tuned_weights[key] / total

            # Validate all required keys are present
            for key in self.default_weights:
                if key not in tuned_weights:
                    print(f"Warning: Missing weight {key}, using default")
                    return self.default_weights

            return tuned_weights

        except Exception as e:
            print(f"Error tuning hyperparameters: {e}")
            print("Falling back to default weights")
            return self.default_weights

    def apply_weights_to_environment(self, weights: Dict[str, float]) -> None:
        """Apply tuned weights to environment variables"""
        for param, value in weights.items():
            os.environ[param] = str(value)

    async def tune_and_apply(self, user_prompt: Optional[str]) -> Dict[str, float]:
        """Tune hyperparameters based on user prompt and apply to environment"""

        if not user_prompt or user_prompt.strip() == "":
            print("No user prompt provided, using default hyperparameters")
            weights = self.default_weights
        else:
            print(f"Tuning hyperparameters for user intent: '{user_prompt}'")
            weights = await self.tune_hyperparameters_for_intent(user_prompt)

            # Show what changed
            print("Hyperparameter adjustments:")
            for param in weights:
                default_val = self.default_weights[param]
                new_val = weights[param]
                change = new_val - default_val
                if abs(change) > 0.01:
                    print(
                        f"  {param}: {default_val:.2f} → {new_val:.2f} ({change:+.2f})"
                    )
                else:
                    print(f"  {param}: {new_val:.2f} (unchanged)")

        # Apply to environment
        self.apply_weights_to_environment(weights)

        return weights


# Test the tuner with different scenarios
async def test_hyperparameter_tuning():
    """Test different user intents and see how hyperparameters adjust"""

    tuner = HyperparameterTuner()

    test_cases = [
        "I want to find people to hire for my startup team",
        "I'm looking for peer-to-peer networking with other CEOs",
        "I need strategic business partnerships to expand my market",
        "I want mentorship connections for junior developers",
        "I'm building a diverse product team with complementary skills",
        "",  # Empty prompt (should use defaults)
    ]

    print("HYPERPARAMETER TUNING TEST CASES")
    print("=" * 60)

    for i, user_prompt in enumerate(test_cases, 1):
        print(
            f"\n{i}. User Intent: '{user_prompt if user_prompt else 'No specific intent'}'"
        )
        print("-" * 40)

        weights = await tuner.tune_and_apply(user_prompt)

        # Show final weights
        print("Final weights:")
        for param, weight in weights.items():
            print(f"  {param}: {weight:.2f}")

        print(f"Sum: {sum(weights.values()):.2f}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_hyperparameter_tuning())
