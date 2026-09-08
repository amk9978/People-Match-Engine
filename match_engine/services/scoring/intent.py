import json
import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Dict, List, Protocol

import openai

from match_engine import settings
from match_engine.services.features.feature_set import FeatureSet

logger = logging.getLogger(__name__)

NEUTRAL_DIRECTION = 0.5
SAMPLE_VALUES_PER_FEATURE = 3
SAMPLE_VALUE_CHARS = 80


@dataclass(frozen=True)
class Intent:
    """What the user asked for, per feature.

    importance says how much a feature matters, as a share of one. direction says
    which way, 1.0 meaning look for sameness and 0.0 meaning look for difference.
    """

    importance: Dict[str, float]
    direction: Dict[str, float]

    @staticmethod
    def uniform(feature_names) -> "Intent":
        names = list(feature_names)
        assert names, "an intent needs at least one feature"
        share = 1.0 / len(names)
        return Intent(
            importance={name: share for name in names},
            direction={name: NEUTRAL_DIRECTION for name in names},
        )


class IntentResolver(Protocol):
    def resolve(self, prompt: str, feature_set: FeatureSet, samples) -> Intent: ...


class UniformIntentResolver:
    """The offline default: every feature matters equally and neither way.

    Weights then reduce to informativeness alone, which is measured from the data
    and needs no key.
    """

    def resolve(self, prompt: str, feature_set: FeatureSet, samples=None) -> Intent:
        if prompt:
            logger.info("No intent resolver configured, ignoring the prompt")
        return Intent.uniform(feature_set.names)


class LLMIntentResolver:
    """Turns a sentence of intent into a per-feature importance and direction.

    The model sees only the runtime feature names and a few sample values, so the
    step works on any schema. It is asked for an allocation out of 100 rather than
    calibrated magnitudes, because normalization discards magnitude anyway.
    """

    def __init__(self, client=None):
        self.client = client
        self._resolved = {}

    def resolve(
        self, prompt: str, feature_set: FeatureSet, samples: Dict[str, List[str]] = None
    ) -> Intent:
        if not prompt or not prompt.strip():
            return Intent.uniform(feature_set.names)

        cache_key = (prompt.strip(), feature_set.names)
        if cache_key in self._resolved:
            return self._resolved[cache_key]

        try:
            intent = self._ask(prompt, feature_set, samples or {})
        except Exception as error:
            logger.warning(
                f"Intent resolution failed, falling back to uniform weights: {error}"
            )
            intent = Intent.uniform(feature_set.names)

        self._resolved[cache_key] = intent
        return intent

    def _ask(
        self, prompt: str, feature_set: FeatureSet, samples: Dict[str, List[str]]
    ) -> Intent:
        client = self.client or openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": self._build_prompt(prompt, feature_set, samples),
                }
            ],
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS_TUNING,
        )
        return self._parse(response.choices[0].message.content, feature_set.names)

    def _build_prompt(
        self, prompt: str, feature_set: FeatureSet, samples: Dict[str, List[str]]
    ) -> str:
        descriptions = []
        for name in feature_set.names:
            examples = samples.get(name, [])
            trimmed = [value[:SAMPLE_VALUE_CHARS] for value in examples]
            descriptions.append(f'- {name}: e.g. {"; ".join(trimmed) or "no samples"}')

        return dedent(f"""\
            People at a professional event are matched on the features below.
            Each feature is one column of their profile.

            FEATURES:
            {chr(10).join(descriptions)}

            The user wants: "{prompt}"

            For each feature give:
            - importance: how much this feature should influence the match,
              as an allocation out of 100 across all features
            - direction: 0 to 100, where 100 means match people who are ALIKE on
              this feature and 0 means match people who DIFFER on it

            Hiring for a fintech startup, for example, wants the same industry and
            different roles.

            Return only this JSON, no prose and no code fences:
            {{"importance": {{"<feature>": <number>}}, "direction": {{"<feature>": <number>}}}}

            Use exactly these feature names: {", ".join(feature_set.names)}.
            The importance values must sum to 100.""")

    def _parse(self, content: str, names) -> Intent:
        payload = json.loads(content.strip())
        raw_importance = payload["importance"]
        raw_direction = payload["direction"]

        importance = {
            name: max(0.0, float(raw_importance.get(name, 0.0))) for name in names
        }
        total = sum(importance.values())
        if total <= 0:
            return Intent.uniform(names)

        direction = {
            name: min(1.0, max(0.0, float(raw_direction.get(name, 50.0)) / 100.0))
            for name in names
        }
        return Intent(
            importance={name: value / total for name, value in importance.items()},
            direction=direction,
        )


def create_intent_resolver() -> IntentResolver:
    """Use the model when a key is configured, otherwise weigh on data alone."""
    if settings.OPENAI_API_KEY:
        return LLMIntentResolver()
    return UniformIntentResolver()


def sample_values(df, feature_set: FeatureSet) -> Dict[str, List[str]]:
    """A few distinct values per feature, for the intent prompt."""
    samples = {}
    for feature in feature_set:
        if feature.column not in df.columns:
            samples[feature.name] = []
            continue
        values = df[feature.column].dropna().astype(str).str.strip()
        distinct = [value for value in values.unique() if value]
        samples[feature.name] = list(distinct[:SAMPLE_VALUES_PER_FEATURE])
    return samples
