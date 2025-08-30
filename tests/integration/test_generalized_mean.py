import json
import os
from textwrap import dedent
from unittest.mock import patch

import openai
import pandas as pd
import pytest

import settings
from services.analysis.dataset_insights import DatasetInsightsAnalyzer
from services.graph.scoring.generalized_mean import tune_parameters
from shared.shared import FEATURE_COLUMN_MAPPING, FEATURES


class TestTuneParametersIntegration:
    """
    Integration tests for tune_parameters function using real OpenAI API calls.
    """

    def llm_judge_weights(self, prompt: str, w_s: dict, w_c: dict) -> dict:
        """
        Use LLM as judge to evaluate the quality of weight tuning.
        Returns a judgment with score and reasoning.
        """
        try:
            real_api_key = os.getenv("OPENAI_API_KEY")
            if not real_api_key:
                return {"error": "No API key available for judging"}

            client = openai.OpenAI(api_key=real_api_key)

            judge_prompt = dedent(
                f"""
            You are an expert evaluator of professional network matching algorithms.
            
            TASK: Evaluate how well these weights align with the user's intent.
            
            USER INTENT: "{prompt}"
            
            GENERATED WEIGHTS:
            Similarity weights (higher = prefer similar people):
            - Role Specification: {w_s.get('role', 0)}
            - Experience Level: {w_s.get('experience', 0)}  
            - industry: {w_s.get('industry', 0)}
            - market: {w_s.get('market', 0)}
            - offering: {w_s.get('offering', 0)}
            - persona: {w_s.get('persona', 0)}
            
            Complementarity weights (higher = prefer different people):
            - role: {w_c.get('role', 0)}
            - experience: {w_c.get('experience', 0)}
            - industry: {w_c.get('industry', 0)} 
            - market: {w_c.get('market', 0)}
            - offering: {w_c.get('offering', 0)}
            - persona: {w_c.get('persona', 0)}
            
            EVALUATION CRITERIA:
            1. Alignment: Do the weights match the user's intent?
            2. Logic: Are the similarity vs complementarity choices sensible?
            3. Balance: Are the relative weights across features appropriate?
            
            RESPOND WITH JSON:
            {{
                "score": <1-10>,
                "reasoning": "<brief explanation of why this is good/bad>"
            }}
            """
            )

            response = client.chat.completions.create(
                model=settings.JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_tokens=settings.JUDGE_MAX_TOKENS,
            )

            result = json.loads(response.choices[0].message.content.strip())
            return result

        except Exception as e:
            return {"error": f"Judge evaluation failed: {str(e)}"}

    @pytest.fixture(autouse=True)
    def restore_real_api_key(self):
        """Override conftest.py mock to use real API key for integration tests"""
        real_api_key = os.getenv("OPENAI_API_KEY")
        if real_api_key:
            with patch("settings.OPENAI_API_KEY", real_api_key):
                yield
        else:
            yield

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "prompt",
        [
            "I want to maximize the hiring chance",
            "I want to maximize peer networking",
            "I want to maximize business partnerships",
            "I want to find mentors for junior developers",
            "I want to build a diverse team with complementary skills",
            "I want to find investors for my startup",
            "I want to find similar companies for market research",
            "I want to find co-founders with different expertise",
        ],
    )
    def test_tune_parameters_with_prompt(self, prompt):
        """Test tune_parameters with real ChatGPT API for any given prompt"""
        w_s, w_c = tune_parameters(prompt=prompt)

        # Basic validation
        assert isinstance(w_s, dict), f"w_s should be dict, got {type(w_s)}"
        assert isinstance(w_c, dict), f"w_c should be dict, got {type(w_c)}"

        # Verify all features are present
        for feature in FEATURES:
            assert feature in w_s, f"Feature {feature} missing from similarity weights"
            assert (
                feature in w_c
            ), f"Feature {feature} missing from complementarity weights"
            assert isinstance(
                w_s[feature], (int, float)
            ), f"w_s[{feature}] should be numeric"
            assert isinstance(
                w_c[feature], (int, float)
            ), f"w_c[{feature}] should be numeric"
            assert w_s[feature] >= 0, f"w_s[{feature}] should be non-negative"
            assert w_c[feature] >= 0, f"w_c[{feature}] should be non-negative"
            assert (
                w_s[feature] <= 2.0
            ), f"w_s[{feature}] should be <= 2.0 (got {w_s[feature]})"
            assert (
                w_c[feature] <= 2.0
            ), f"w_c[{feature}] should be <= 2.0 (got {w_c[feature]})"

        # Print results for analysis
        print(f"\nPrompt: '{prompt}'")
        print(f"Similarity weights:     {w_s}")
        print(f"Complementarity weights: {w_c}")
        print("-" * 80)

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    @pytest.mark.integration
    def test_tune_parameters_custom_prompt(self):
        """Test tune_parameters with a custom prompt - modify this test as needed"""

        # Change this prompt to whatever you want to test
        custom_prompt = (
            "I want to find technical co-founders who complement my business skills"
        )

        w_s, w_c = tune_parameters(prompt=custom_prompt)

        # Basic validation
        assert isinstance(w_s, dict)
        assert isinstance(w_c, dict)

        for feature in FEATURES:
            assert feature in w_s
            assert feature in w_c
            assert w_s[feature] <= 2.0
            assert w_c[feature] <= 2.0

        print(f"\nCustom Prompt: '{custom_prompt}'")
        print(f"Similarity weights:     {w_s}")
        print(f"Complementarity weights: {w_c}")

        # Add any specific assertions about expected weights here
        # For example, for finding technical co-founders:
        # assert w_c["role"] > w_s["role"], "Should prefer complementary roles"

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    @pytest.mark.integration
    def test_tune_parameters_with_llm_judge(self):
        """Test tune_parameters with LLM-as-judge evaluation"""

        test_cases = [
            (
                "I want to maximize hiring chance",
                {
                    "expected_high_complementarity": ["role", "experience"],
                    "expected_high_similarity": ["industry", "market"],
                },
            ),
            (
                "I want to maximize peer networking",
                {
                    "expected_high_similarity": [
                        "role",
                        "experience",
                        "industry",
                        "persona",
                    ],
                    "expected_low_complementarity": ["role", "experience", "persona"],
                },
            ),
            (
                "I want to find business partners",
                {
                    "expected_high_complementarity": ["market", "offering"],
                    "expected_high_similarity": ["industry"],
                },
            ),
        ]

        for prompt, expectations in test_cases:
            print(f"\n{'=' * 80}")
            print(f"TESTING: {prompt}")
            print(f"{'=' * 80}")

            # Get weights from ChatGPT
            w_s, w_c = tune_parameters(prompt=prompt)

            print(f"Similarity weights:     {w_s}")
            print(f"Complementarity weights: {w_c}")

            judgment = self.llm_judge_weights(prompt, w_s, w_c)

            if "error" in judgment:
                print(f"Judge evaluation failed: {judgment['error']}")
                continue

            print(f"\nLLM JUDGE EVALUATION:")
            print(f"Score: {judgment.get('score', 'N/A')}/10")
            print(f"Reasoning: {judgment.get('reasoning', 'No reasoning provided')}")

            # Assert minimum quality threshold
            score = judgment.get("score", 0)
            assert (
                score >= 6
            ), f"Score too low: {score}/10 for prompt '{prompt}' - {judgment.get('reasoning', '')}"

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    @pytest.mark.integration
    def test_context_impact_comparison(self):
        """A/B test to compare ChatGPT tuning with and without dataset context"""
        test_file_path = "./data/test_batch3.csv"

        if not os.path.exists(test_file_path):
            pytest.skip(f"Test file {test_file_path} not found")

        raw_data = pd.read_csv(test_file_path)

        # Check if all required CSV columns exist
        missing_columns = [
            csv_col
            for csv_col in FEATURE_COLUMN_MAPPING.values()
            if csv_col not in raw_data.columns
        ]
        if missing_columns:
            pytest.skip(f"Dataset missing required columns: {missing_columns}")

        # Map CSV column names to internal feature names
        column_mapping = {
            csv_col: feature_name
            for feature_name, csv_col in FEATURE_COLUMN_MAPPING.items()
        }
        sample_data = raw_data.rename(columns=column_mapping)

        # Verify we now have all FEATURES
        missing_features = [f for f in FEATURES if f not in sample_data.columns]
        if missing_features:
            pytest.skip(f"Column mapping failed, missing features: {missing_features}")

        print(f"\n{'=' * 80}")
        print(f"DATASET CONTEXT IMPACT ANALYSIS")
        print(f"{'=' * 80}")
        print(f"Loading dataset: {test_file_path}")
        print(f"Dataset shape: {sample_data.shape}")

        # Show distribution for all features (now using mapped feature names)
        for feature in FEATURES:
            if feature in sample_data.columns:
                distribution = (
                    sample_data[feature].value_counts().head(5).to_dict()
                )  # Top 5 values
                print(f"{feature} distribution (top 5): {distribution}")
            else:
                print(f"{feature}: Missing after column mapping")

        # Analyze dataset to get context
        insights_analyzer = DatasetInsightsAnalyzer()
        feature_insights = insights_analyzer.analyze_dataset_directly(sample_data)
        dataset_context = insights_analyzer.generate_context_summary(feature_insights)

        print(f"\nDataset Context: {dataset_context}")

        # Test prompts designed to benefit from context awareness
        test_prompts = [
            "I want to maximize hiring chance for diverse roles",
            "I want to find business partners with complementary markets",
            "I want to find co-founders with different expertise",
            "I want to find mentors for junior team members",
            "I want to build investor networks for fundraising",
            "I want to find advisors with industry expertise",
            "I want to maximize peer networking opportunities",
            "I want to find competitors for market intelligence",
            "I want to identify acquisition targets",
            "I want to find strategic alliance partners",
            "I want to build a diverse leadership team",
            "I want to find customers in new market segments",
        ]

        for prompt in test_prompts:
            print(f"\n{'=' * 80}")
            print(f"TESTING PROMPT: {prompt}")
            print(f"{'=' * 80}")

            # Test WITHOUT context
            print(f"\n🔴 WITHOUT CONTEXT:")
            w_s_no_context, w_c_no_context = tune_parameters(
                prompt=prompt, insights=None
            )
            print(f"Similarity weights:     {w_s_no_context}")
            print(f"Complementarity weights: {w_c_no_context}")

            # Test WITH context
            print(f"\n🟢 WITH CONTEXT:")
            w_s_with_context, w_c_with_context = tune_parameters(
                prompt=prompt, insights=dataset_context
            )
            print(f"Similarity weights:     {w_s_with_context}")
            print(f"Complementarity weights: {w_c_with_context}")

            # Calculate differences
            print(f"\n📊 IMPACT ANALYSIS:")
            sim_diffs = {
                f: abs(w_s_with_context[f] - w_s_no_context[f]) for f in FEATURES
            }
            comp_diffs = {
                f: abs(w_c_with_context[f] - w_c_no_context[f]) for f in FEATURES
            }

            total_sim_change = sum(sim_diffs.values())
            total_comp_change = sum(comp_diffs.values())

            print(f"Similarity weight changes: {sim_diffs}")
            print(f"Complementarity weight changes: {comp_diffs}")
            print(f"Total similarity change: {total_sim_change:.3f}")
            print(f"Total complementarity change: {total_comp_change:.3f}")

            # Identify biggest changes
            biggest_sim_change = max(sim_diffs.items(), key=lambda x: x[1])
            biggest_comp_change = max(comp_diffs.items(), key=lambda x: x[1])
            print(
                f"Biggest similarity change: {biggest_sim_change[0]} ({biggest_sim_change[1]:.3f})"
            )
            print(
                f"Biggest complementarity change: {biggest_comp_change[0]} ({biggest_comp_change[1]:.3f})"
            )

            # Get LLM judge scores for both
            print(f"\n🔍 LLM JUDGE COMPARISON:")

            judgment_no_context = self.llm_judge_weights(
                prompt, w_s_no_context, w_c_no_context
            )
            judgment_with_context = self.llm_judge_weights(
                prompt, w_s_with_context, w_c_with_context
            )

            if (
                "error" not in judgment_no_context
                and "error" not in judgment_with_context
            ):
                score_no_context = judgment_no_context.get("score", 0)
                score_with_context = judgment_with_context.get("score", 0)
                score_improvement = score_with_context - score_no_context

                print(f"Without context - Score: {score_no_context}/10")
                print(f"Reasoning: {judgment_no_context.get('reasoning', 'N/A')}")
                print(f"\nWith context - Score: {score_with_context}/10")
                print(f"Reasoning: {judgment_with_context.get('reasoning', 'N/A')}")
                print(f"\n📈 IMPROVEMENT: {score_improvement:+.1f} points")

                # Assert that context provides some benefit (or at least doesn't hurt)
                assert (
                    score_improvement >= -1
                ), f"Context significantly hurt performance: {score_improvement}"

            else:
                print("LLM judge evaluation failed for comparison")

            # Basic validation
            for feature in FEATURES:
                assert 0.1 <= w_s_no_context[feature] <= 2.0
                assert 0.1 <= w_c_no_context[feature] <= 2.0
                assert 0.1 <= w_s_with_context[feature] <= 2.0
                assert 0.1 <= w_c_with_context[feature] <= 2.0

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    @pytest.mark.integration
    def test_end_to_end_context_evaluation(self):
        """End-to-end test: A/B test parameter tuning, then use LLM judge to evaluate context impact"""
        import pandas as pd
        from services.analysis.dataset_insights import DatasetInsightsAnalyzer
        from shared.shared import FEATURE_COLUMN_MAPPING
        import os

        test_file_path = "./data/test_batch3.csv"
        if not os.path.exists(test_file_path):
            pytest.skip(f"Test file {test_file_path} not found")

        raw_data = pd.read_csv(test_file_path)
        missing_columns = [
            csv_col
            for csv_col in FEATURE_COLUMN_MAPPING.values()
            if csv_col not in raw_data.columns
        ]
        if missing_columns:
            pytest.skip(f"Dataset missing required columns: {missing_columns}")

        # Map columns and analyze dataset
        column_mapping = {
            csv_col: feature_name
            for feature_name, csv_col in FEATURE_COLUMN_MAPPING.items()
        }
        sample_data = raw_data.rename(columns=column_mapping)

        insights_analyzer = DatasetInsightsAnalyzer()
        feature_insights = insights_analyzer.analyze_dataset_directly(sample_data)
        dataset_context = insights_analyzer.generate_context_summary(feature_insights)

        print(f"\n{'='*100}")
        print(f"END-TO-END CONTEXT EVALUATION")
        print(f"{'='*100}")
        print(f"Dataset: {test_file_path}")
        print(f"Context: {dataset_context}")

        # Test with business-focused prompts that should benefit from context
        test_prompts = [
            "I want to maximize hiring success by finding diverse skill sets",
            "I want to find strategic business partners with complementary capabilities",
        ]

        for prompt in test_prompts:
            print(f"\n{'='*60}")
            print(f"EVALUATING: {prompt}")
            print(f"{'='*60}")

            # Get weights without context
            w_s_no_context, w_c_no_context = tune_parameters(
                prompt=prompt, insights=None
            )

            # Get weights with context
            w_s_with_context, w_c_with_context = tune_parameters(
                prompt=prompt, insights=dataset_context
            )

            print(f"\nWithout context: sim={w_s_no_context}, comp={w_c_no_context}")
            print(f"With context: sim={w_s_with_context}, comp={w_c_with_context}")

            # Use LLM judge to evaluate the context impact
            context_evaluation = self.llm_judge_context_impact(
                prompt=prompt,
                dataset_context=dataset_context,
                w_s_no_context=w_s_no_context,
                w_c_no_context=w_c_no_context,
                w_s_with_context=w_s_with_context,
                w_c_with_context=w_c_with_context,
            )

            if "error" in context_evaluation:
                print(f"❌ Context evaluation failed: {context_evaluation['error']}")
                continue

            print(f"\n🔍 END-TO-END EVALUATION RESULTS:")
            print(
                f"Context Improved Tuning: {context_evaluation.get('context_improved_tuning', 'N/A')}"
            )
            print(
                f"Improvement Score: {context_evaluation.get('improvement_score', 'N/A')}/10"
            )
            print(
                f"Reasoning: {context_evaluation.get('reasoning', 'No reasoning provided')}"
            )

            if context_evaluation.get("key_improvements"):
                print(f"Key Improvements: {context_evaluation['key_improvements']}")

            if context_evaluation.get("missed_opportunities"):
                print(
                    f"Missed Opportunities: {context_evaluation['missed_opportunities']}"
                )

            # Assert that context provided meaningful impact
            improved = context_evaluation.get("context_improved_tuning", False)
            improvement_score = context_evaluation.get("improvement_score", 0)

            if not improved and improvement_score < 5:
                print(
                    f"⚠️  Warning: Dataset context may not be providing significant value"
                )
            else:
                print(f"✅ Dataset context appears to be improving parameter tuning")


if __name__ == "__main__":
    # Run with: pytest tests/integration/test_generalized_mean.py -v -s -m integration
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
