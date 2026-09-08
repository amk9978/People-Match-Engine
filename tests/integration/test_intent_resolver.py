import json
import os
from textwrap import dedent

import openai
import pandas as pd
import pytest

import settings
from services.features.schema_mapper import SchemaMapper
from services.scoring.intent import LLMIntentResolver

pytestmark = pytest.mark.integration

SAMPLE_CSV = "docs/sample.csv"
VENDOR_PRESET = "presets/vendor_six_column.yaml"
PASSING_SCORE = 7


@pytest.fixture(scope="module")
def feature_set():
    df = pd.read_csv(SAMPLE_CSV)
    return SchemaMapper().from_file(VENDOR_PRESET, df)


@pytest.fixture(autouse=True)
def require_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")


def judge(prompt: str, intent) -> dict:
    """Ask a stronger model whether the resolved intent matches what was asked for."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    question = dedent(
        f"""\
        A matching engine turned a user's request into per-feature weights.

        REQUEST: "{prompt}"

        importance (share of the total): {json.dumps(intent.importance)}
        direction (1.0 means match people who are alike, 0.0 means match people
        who differ): {json.dumps(intent.direction)}

        Score from 1 to 10 how well these serve the request, and say why.
        Return only {{"score": <1-10>, "reasoning": "<one sentence>"}}."""
    )

    response = client.chat.completions.create(
        model=settings.JUDGE_MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0,
        max_tokens=settings.JUDGE_MAX_TOKENS,
    )
    return json.loads(response.choices[0].message.content.strip())


class TestIntentResolution:
    @pytest.mark.parametrize(
        "prompt",
        [
            "I want to hire senior engineers for my fintech startup",
            "I want to meet peers doing the same job as me",
            "I am looking for a business partner in a different market",
        ],
    )
    def test_a_stated_intent_survives_the_round_trip(self, prompt, feature_set):
        intent = LLMIntentResolver().resolve(prompt, feature_set)

        verdict = judge(prompt, intent)
        assert verdict["score"] >= PASSING_SCORE, verdict["reasoning"]

    def test_hiring_wants_the_same_industry_and_a_different_role(self, feature_set):
        intent = LLMIntentResolver().resolve(
            "hiring an engineer for my fintech startup", feature_set
        )

        assert intent.direction["industry"] > intent.direction["role"]

    def test_the_importance_allocation_sums_to_one(self, feature_set):
        intent = LLMIntentResolver().resolve("find me an investor", feature_set)

        assert sum(intent.importance.values()) == pytest.approx(1.0)
        assert set(intent.importance) == set(feature_set.names)
