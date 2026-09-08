import asyncio
import json
import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from match_engine import settings
from match_engine.services.cache.app_cache_service import app_cache_service
from match_engine.services.scoring.report import ScoringReport

logger = logging.getLogger(__name__)

FALLBACK_VALUE = settings.FALLBACK_VALUE

TOKENS_PER_SCORE = 5
ROW_OVERHEAD_TOKENS = 8
RESPONSE_OVERHEAD_TOKENS = 32


class MalformedScores(ValueError):
    """The model's reply did not carry one score per comparison for every target."""


@dataclass(frozen=True)
class ComplementarityScores:
    scores: Dict[str, Dict[str, float]]
    report: ScoringReport


class LLMComplementarityScorer:
    """Scores how complementary two profiles are, one feature at a time.

    The model receives a numbered comparison list once per batch and returns a
    positional array of scores per target. Echoing profile text as JSON keys
    instead cost about forty tokens per score, which overran the completion cap
    on any realistic roster and silently turned most scores into the fallback."""

    def __init__(self, openai_client: AsyncOpenAI = None, cache=None):
        self.openai_client = openai_client or AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY, timeout=settings.OPENAI_TIMEOUT
        )
        self.cache = cache or app_cache_service

    async def get_profile_complementarity(
        self,
        target_profiles: List[str],
        comparison_profiles: List[str],
        category: str,
    ) -> ComplementarityScores:
        """Score every target against every comparison, asking the model only for
        the pairs the cache does not already hold."""
        status = self.cache.get_complementarity_cache_status(
            target_profiles, comparison_profiles, category
        )
        scores = {target: dict(row) for target, row in status.cached.items()}
        report = ScoringReport(
            cached_pairs=sum(len(row) for row in status.cached.values())
        )

        for comparisons, targets in self._group_by_missing(status.missing).items():
            group_scores, group_report = await self._score_group(
                list(targets), list(comparisons), category
            )
            for target, row in group_scores.items():
                scores.setdefault(target, {}).update(row)
            self.cache.cache_complementarity_scores(group_scores, category)
            report = report.merge(group_report)

        if report.fallback_pairs:
            logger.warning(
                f"Feature {category}: {report.fallback_pairs} of {report.total_pairs} "
                f"pairs fell back to the neutral value"
            )

        return ComplementarityScores(scores=scores, report=report)

    def _group_by_missing(
        self, missing: Dict[str, List[str]]
    ) -> Dict[Tuple[str, ...], List[str]]:
        """Group targets that need the same comparisons so each batch asks one question.

        Adding one person to a scored roster leaves every existing target missing
        exactly that person, which collapses into a single group."""
        groups: Dict[Tuple[str, ...], List[str]] = {}
        for target, comparisons in missing.items():
            groups.setdefault(tuple(comparisons), []).append(target)
        return groups

    def max_targets_per_batch(self, comparison_count: int) -> int:
        """How many targets fit in one completion, given the comparison count.

        The model caps completions, so batch size follows from the roster rather
        than from a constant. Raising a fixed batch size cannot buy headroom the
        cap does not have."""
        assert comparison_count > 0, "a batch needs at least one comparison"
        per_target = comparison_count * TOKENS_PER_SCORE + ROW_OVERHEAD_TOKENS
        budget = settings.COMPLEMENTARITY_MAX_COMPLETION_TOKENS
        return max(1, budget // per_target)

    async def _score_group(
        self, targets: List[str], comparisons: List[str], category: str
    ) -> Tuple[Dict[str, Dict[str, float]], ScoringReport]:
        batch_size = self.max_targets_per_batch(len(comparisons))
        batches = [
            targets[start : start + batch_size]
            for start in range(0, len(targets), batch_size)
        ]

        logger.info(
            f"Feature {category}: scoring {len(targets)} targets against "
            f"{len(comparisons)} comparisons in {len(batches)} calls of up to "
            f"{batch_size} targets"
        )

        outcomes = await asyncio.gather(
            *[self._score_batch(batch, comparisons, category) for batch in batches],
            return_exceptions=True,
        )

        scores: Dict[str, Dict[str, float]] = {}
        scored_pairs = 0
        fallback_pairs = 0

        for batch, outcome in zip(batches, outcomes):
            if isinstance(outcome, Exception):
                logger.error(
                    f"Feature {category}: a batch of {len(batch)} targets failed, "
                    f"filling with the neutral value: {outcome}"
                )
                for target in batch:
                    scores[target] = {
                        comparison: FALLBACK_VALUE for comparison in comparisons
                    }
                fallback_pairs += len(batch) * len(comparisons)
                continue

            scores.update(outcome)
            scored_pairs += len(batch) * len(comparisons)

        return scores, ScoringReport(
            scored_pairs=scored_pairs,
            fallback_pairs=fallback_pairs,
            model_calls=len(batches),
        )

    async def _score_batch(
        self, targets: List[str], comparisons: List[str], category: str
    ) -> Dict[str, Dict[str, float]]:
        prompt = self._build_prompt(targets, comparisons, category)
        max_tokens = min(
            settings.COMPLEMENTARITY_MAX_COMPLETION_TOKENS,
            len(targets) * (len(comparisons) * TOKENS_PER_SCORE + ROW_OVERHEAD_TOKENS)
            + RESPONSE_OVERHEAD_TOKENS,
        )

        raw = await self.openai_client.chat.completions.with_raw_response.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.TEMPERATURE,
            max_tokens=max_tokens,
        )
        self._log_rate_limits(raw.headers, category)

        content = raw.parse().choices[0].message.content
        rows = self._parse_scores(content, len(targets), len(comparisons))

        return {
            target: dict(zip(comparisons, row)) for target, row in zip(targets, rows)
        }

    def _build_prompt(
        self, targets: List[str], comparisons: List[str], category: str
    ) -> str:
        comparison_list = "\n".join(
            f"{index}. {profile}" for index, profile in enumerate(comparisons, start=1)
        )
        target_list = "\n".join(
            f"{index}. {profile}" for index, profile in enumerate(targets, start=1)
        )

        return dedent(
            f"""\
            Rate the complementarity of professional profiles on the {category} dimension.

            COMPARISON PROFILES:
            {comparison_list}

            TARGET PROFILES:
            {target_list}

            Score each target against every comparison, from 0.0 to 1.0:
            - 0.9-1.0: highly complementary, significant strategic value together
            - 0.7-0.8: strong complementarity with clear synergistic potential
            - 0.5-0.6: moderate complementarity, some collaboration opportunity
            - 0.3-0.4: different but not particularly synergistic
            - 0.1-0.2: too similar or directly conflicting
            - 0.0: identical or directly competing

            Return only this JSON object, no prose and no code fences:
            {{"scores": [[...], [...]]}}

            "scores" holds exactly {len(targets)} arrays, one per target in the
            order listed above. Each array holds exactly {len(comparisons)}
            numbers, one per comparison in the order listed above."""
        )

    def _parse_scores(
        self, content: Optional[str], target_count: int, comparison_count: int
    ) -> List[List[float]]:
        """Read the score arrays, refusing anything that is not the exact shape asked for."""
        if content is None:
            raise MalformedScores("the model returned no content")

        try:
            payload = json.loads(content.strip())
        except json.JSONDecodeError as error:
            raise MalformedScores(f"reply was not JSON: {error}") from error

        rows = payload.get("scores") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise MalformedScores("reply carried no 'scores' array")

        if len(rows) != target_count:
            raise MalformedScores(
                f"expected {target_count} score arrays, got {len(rows)}"
            )

        parsed = []
        for row in rows:
            if not isinstance(row, list) or len(row) != comparison_count:
                raise MalformedScores(
                    f"expected {comparison_count} scores per target, got "
                    f"{len(row) if isinstance(row, list) else type(row).__name__}"
                )
            parsed.append([self._parse_score(value) for value in row])

        return parsed

    def _parse_score(self, value) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise MalformedScores(f"score was not a number: {value!r}")
        return min(1.0, max(0.0, float(value)))

    def _log_rate_limits(self, headers, category: str) -> None:
        logger.debug(
            f"Feature {category}: "
            f"{headers.get('x-ratelimit-remaining-requests')} requests and "
            f"{headers.get('x-ratelimit-remaining-tokens')} tokens remaining, "
            f"request {headers.get('x-request-id')}"
        )
