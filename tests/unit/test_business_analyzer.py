from unittest.mock import MagicMock

import pytest

from services.analysis.business_analyzer import BusinessAnalyzer
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


@pytest.fixture
def analyzer():
    cache = AppCacheService(backend=InMemoryBackend())
    analyzer = BusinessAnalyzer(openai_client=MagicMock(), cache=cache)
    analyzer._process_single_batch = RecordingScorer()
    return analyzer


class TestScoring:
    async def test_every_pair_gets_a_score(self, analyzer):
        roster = [ADA, GRACE, KATHERINE]

        results = await analyzer.get_profile_complementarity(roster, roster, ROLE)

        assert set(results) == set(roster)
        assert results[ADA] == {GRACE: 0.7, KATHERINE: 0.7}

    async def test_a_profile_is_never_scored_against_itself(self, analyzer):
        roster = [ADA, GRACE]

        await analyzer.get_profile_complementarity(roster, roster, ROLE)

        assert all(
            target != comparison
            for target, comparison in analyzer._process_single_batch.scored_pairs
        )

    async def test_a_second_run_asks_the_model_nothing(self, analyzer):
        roster = [ADA, GRACE, KATHERINE]
        await analyzer.get_profile_complementarity(roster, roster, ROLE)
        analyzer._process_single_batch.requests.clear()

        results = await analyzer.get_profile_complementarity(roster, roster, ROLE)

        assert analyzer._process_single_batch.requests == []
        assert results[GRACE] == {ADA: 0.7, KATHERINE: 0.7}

    async def test_adding_one_person_scores_only_that_person_pairs(self, analyzer):
        await analyzer.get_profile_complementarity([ADA, GRACE], [ADA, GRACE], ROLE)
        analyzer._process_single_batch.requests.clear()

        roster = [ADA, GRACE, KATHERINE]
        results = await analyzer.get_profile_complementarity(roster, roster, ROLE)

        assert analyzer._process_single_batch.scored_pairs == {
            (ADA, KATHERINE),
            (GRACE, KATHERINE),
            (KATHERINE, ADA),
            (KATHERINE, GRACE),
        }
        assert results[ADA] == {GRACE: 0.7, KATHERINE: 0.7}

    async def test_targets_needing_the_same_comparisons_share_one_request(
        self, analyzer
    ):
        await analyzer.get_profile_complementarity([ADA, GRACE], [ADA, GRACE], ROLE)
        analyzer._process_single_batch.requests.clear()

        roster = [ADA, GRACE, KATHERINE]
        await analyzer.get_profile_complementarity(roster, roster, ROLE)

        existing = [
            request
            for request in analyzer._process_single_batch.requests
            if request[1] == (KATHERINE,)
        ]
        assert len(existing) == 1
        assert set(existing[0][0]) == {ADA, GRACE}

    async def test_a_failed_batch_falls_back_without_losing_the_other_batches(
        self, analyzer
    ):
        roster = [ADA, GRACE, KATHERINE]

        async def fail_on_first(batch_targets, comparison_profiles, category):
            if ADA in batch_targets:
                raise RuntimeError("model unavailable")
            return {
                target: {comparison: 0.7 for comparison in comparison_profiles}
                for target in batch_targets
            }

        analyzer._process_single_batch = fail_on_first

        results = await analyzer.get_profile_complementarity(
            roster, roster, ROLE, batch_size=1
        )

        assert results[ADA] == {GRACE: 0.5, KATHERINE: 0.5}
        assert results[GRACE] == {ADA: 0.7, KATHERINE: 0.7}
