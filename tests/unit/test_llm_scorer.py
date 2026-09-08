import json
from unittest.mock import MagicMock

import pytest

from services.scoring.llm_scorer import LLMComplementarityScorer, MalformedScores
from services.cache.app_cache_service import AppCacheService
from services.cache.memory import InMemoryBackend

ROLE = "role"

ADA = "Founder | Systems"
GRACE = "Compiler Engineer | Tooling"
KATHERINE = "Orbital Mechanics | Research"


class RecordingScorer:
    """Stands in for the model call and records what each batch was asked to score."""

    def __init__(self):
        self.requests = []

    async def __call__(self, batch_targets, comparison_profiles, category):
        self.requests.append((tuple(batch_targets), tuple(comparison_profiles)))
        return {
            target: {comparison: 0.7 for comparison in comparison_profiles}
            for target in batch_targets
        }

    @property
    def scored_pairs(self):
        return {
            (target, comparison)
            for targets, comparisons in self.requests
            for target in targets
            for comparison in comparisons
        }


class FakeCompletions:
    """Minimal stand-in for the OpenAI raw-response client."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.with_raw_response = self

    async def create(self, model, messages, temperature, max_tokens):
        self.requests.append(
            {"prompt": messages[0]["content"], "max_tokens": max_tokens}
        )
        return FakeRawResponse(self.replies.pop(0))


class FakeRawResponse:
    def __init__(self, content):
        self.headers = {}
        self._content = content

    def parse(self):
        message = MagicMock()
        message.content = self._content
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response


def make_analyzer(client=None):
    cache = AppCacheService(backend=InMemoryBackend())
    return LLMComplementarityScorer(openai_client=client or MagicMock(), cache=cache)


@pytest.fixture
def analyzer():
    analyzer = make_analyzer()
    analyzer._score_batch = RecordingScorer()
    return analyzer


class TestScoring:
    async def test_every_pair_gets_a_score(self, analyzer):
        roster = [ADA, GRACE, KATHERINE]

        result = await analyzer.get_profile_complementarity(roster, roster, ROLE)

        assert set(result.scores) == set(roster)
        assert result.scores[ADA] == {GRACE: 0.7, KATHERINE: 0.7}

    async def test_a_profile_is_never_scored_against_itself(self, analyzer):
        await analyzer.get_profile_complementarity([ADA, GRACE], [ADA, GRACE], ROLE)

        assert all(
            target != comparison
            for target, comparison in analyzer._score_batch.scored_pairs
        )

    async def test_a_second_run_asks_the_model_nothing(self, analyzer):
        roster = [ADA, GRACE, KATHERINE]
        await analyzer.get_profile_complementarity(roster, roster, ROLE)
        analyzer._score_batch.requests.clear()

        result = await analyzer.get_profile_complementarity(roster, roster, ROLE)

        assert analyzer._score_batch.requests == []
        assert result.scores[GRACE] == {ADA: 0.7, KATHERINE: 0.7}
        assert result.report.model_calls == 0
        assert result.report.cached_pairs == 6

    async def test_adding_one_person_scores_only_that_person_pairs(self, analyzer):
        await analyzer.get_profile_complementarity([ADA, GRACE], [ADA, GRACE], ROLE)
        analyzer._score_batch.requests.clear()

        roster = [ADA, GRACE, KATHERINE]
        result = await analyzer.get_profile_complementarity(roster, roster, ROLE)

        assert analyzer._score_batch.scored_pairs == {
            (ADA, KATHERINE),
            (GRACE, KATHERINE),
            (KATHERINE, ADA),
            (KATHERINE, GRACE),
        }
        assert result.scores[ADA] == {GRACE: 0.7, KATHERINE: 0.7}

    async def test_targets_needing_the_same_comparisons_share_one_request(
        self, analyzer
    ):
        await analyzer.get_profile_complementarity([ADA, GRACE], [ADA, GRACE], ROLE)
        analyzer._score_batch.requests.clear()

        roster = [ADA, GRACE, KATHERINE]
        await analyzer.get_profile_complementarity(roster, roster, ROLE)

        existing = [
            request
            for request in analyzer._score_batch.requests
            if request[1] == (KATHERINE,)
        ]
        assert len(existing) == 1
        assert set(existing[0][0]) == {ADA, GRACE}


class TestFailureReporting:
    async def test_a_failed_batch_falls_back_without_losing_the_others(self, analyzer):
        async def fail_on_ada(batch_targets, comparison_profiles, category):
            if ADA in batch_targets:
                raise RuntimeError("model unavailable")
            return {
                target: {comparison: 0.7 for comparison in comparison_profiles}
                for target in batch_targets
            }

        analyzer._score_batch = fail_on_ada
        analyzer.max_targets_per_batch = lambda comparison_count: 1
        roster = [ADA, GRACE, KATHERINE]

        result = await analyzer.get_profile_complementarity(roster, roster, ROLE)

        assert result.scores[ADA] == {GRACE: 0.5, KATHERINE: 0.5}
        assert result.scores[GRACE] == {ADA: 0.7, KATHERINE: 0.7}

    async def test_the_report_counts_the_fallback_pairs(self, analyzer):
        async def always_fail(batch_targets, comparison_profiles, category):
            raise RuntimeError("model unavailable")

        analyzer._score_batch = always_fail

        result = await analyzer.get_profile_complementarity(
            [ADA, GRACE], [ADA, GRACE], ROLE
        )

        assert result.report.fallback_pairs == 2
        assert result.report.fallback_rate == 1.0

    async def test_a_clean_run_reports_no_fallbacks(self, analyzer):
        result = await analyzer.get_profile_complementarity(
            [ADA, GRACE], [ADA, GRACE], ROLE
        )

        assert result.report.fallback_pairs == 0
        assert result.report.fallback_rate == 0.0
        assert result.report.scored_pairs == 2


class TestBatchSizing:
    def test_batch_size_shrinks_as_the_roster_grows(self):
        analyzer = make_analyzer()

        assert analyzer.max_targets_per_batch(10) > analyzer.max_targets_per_batch(400)

    def test_a_batch_never_exceeds_the_completion_budget(self, monkeypatch):
        monkeypatch.setattr("settings.COMPLEMENTARITY_MAX_COMPLETION_TOKENS", 8000)
        analyzer = make_analyzer()

        for comparison_count in (1, 10, 69, 400, 5000):
            targets = analyzer.max_targets_per_batch(comparison_count)
            assert targets * comparison_count * 5 <= 8000 or targets == 1

    def test_a_huge_comparison_set_still_scores_one_target_at_a_time(self):
        analyzer = make_analyzer()

        assert analyzer.max_targets_per_batch(100_000) == 1


class TestResponseParsing:
    async def test_a_well_formed_reply_maps_onto_the_comparison_order(self):
        client = MagicMock()
        client.chat.completions = FakeCompletions(
            [json.dumps({"scores": [[0.1, 0.2], [0.3, 0.4]]})]
        )
        analyzer = make_analyzer(client)

        scores = await analyzer._score_batch(
            [ADA, GRACE], [KATHERINE, "Designer | Brand"], ROLE
        )

        assert scores[ADA] == {KATHERINE: 0.1, "Designer | Brand": 0.2}
        assert scores[GRACE] == {KATHERINE: 0.3, "Designer | Brand": 0.4}

    async def test_the_prompt_lists_each_profile_once(self):
        client = MagicMock()
        client.chat.completions = FakeCompletions(
            [json.dumps({"scores": [[0.1, 0.2]]})]
        )
        analyzer = make_analyzer(client)

        await analyzer._score_batch([ADA], [GRACE, KATHERINE], ROLE)

        prompt = client.chat.completions.requests[0]["prompt"]
        assert prompt.count(GRACE) == 1
        assert prompt.count(ADA) == 1

    async def test_a_short_row_is_rejected_rather_than_padded(self):
        client = MagicMock()
        client.chat.completions = FakeCompletions([json.dumps({"scores": [[0.1]]})])
        analyzer = make_analyzer(client)

        with pytest.raises(MalformedScores):
            await analyzer._score_batch([ADA], [GRACE, KATHERINE], ROLE)

    async def test_a_missing_target_row_is_rejected(self):
        client = MagicMock()
        client.chat.completions = FakeCompletions(
            [json.dumps({"scores": [[0.1, 0.2]]})]
        )
        analyzer = make_analyzer(client)

        with pytest.raises(MalformedScores):
            await analyzer._score_batch(
                [ADA, GRACE], [KATHERINE, "Designer | Brand"], ROLE
            )

    async def test_a_truncated_reply_is_rejected(self):
        client = MagicMock()
        client.chat.completions = FakeCompletions(['{"scores": [[0.1, 0.2'])
        analyzer = make_analyzer(client)

        with pytest.raises(MalformedScores):
            await analyzer._score_batch([ADA], [GRACE, KATHERINE], ROLE)

    async def test_a_non_numeric_score_is_rejected(self):
        client = MagicMock()
        client.chat.completions = FakeCompletions(
            [json.dumps({"scores": [["high", 0.2]]})]
        )
        analyzer = make_analyzer(client)

        with pytest.raises(MalformedScores):
            await analyzer._score_batch([ADA], [GRACE, KATHERINE], ROLE)

    async def test_out_of_range_scores_are_clamped(self):
        client = MagicMock()
        client.chat.completions = FakeCompletions(
            [json.dumps({"scores": [[1.4, -0.2]]})]
        )
        analyzer = make_analyzer(client)

        scores = await analyzer._score_batch([ADA], [GRACE, KATHERINE], ROLE)

        assert scores[ADA] == {GRACE: 1.0, KATHERINE: 0.0}

    async def test_the_requested_completion_cap_fits_the_batch(self):
        client = MagicMock()
        client.chat.completions = FakeCompletions(
            [json.dumps({"scores": [[0.1, 0.2]]})]
        )
        analyzer = make_analyzer(client)

        await analyzer._score_batch([ADA], [GRACE, KATHERINE], ROLE)

        assert client.chat.completions.requests[0]["max_tokens"] >= 2 * 5
