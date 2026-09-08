import pytest

from services.cache.app_cache_service import AppCacheService
from services.cache.memory import InMemoryBackend

ROLE = "role"

ADA = "Founder | Systems"
GRACE = "Compiler Engineer | Tooling"
KATHERINE = "Orbital Mechanics | Research"


@pytest.fixture
def cache():
    return AppCacheService(backend=InMemoryBackend())


class TestPairScores:
    def test_a_stored_score_reads_back(self, cache):
        cache.set_pair_score(ROLE, ADA, GRACE, 0.8)
        assert cache.get_pair_score(ROLE, ADA, GRACE) == 0.8

    def test_an_unscored_pair_reads_as_missing(self, cache):
        assert cache.get_pair_score(ROLE, ADA, GRACE) is None

    def test_a_pair_is_readable_from_either_side(self, cache):
        cache.set_pair_score(ROLE, ADA, GRACE, 0.8)
        assert cache.get_pair_score(ROLE, GRACE, ADA) == 0.8

    def test_the_scored_direction_wins_over_the_transpose(self, cache):
        cache.set_pair_score(ROLE, ADA, GRACE, 0.8)
        cache.set_pair_score(ROLE, GRACE, ADA, 0.3)

        assert cache.get_pair_score(ROLE, ADA, GRACE) == 0.8
        assert cache.get_pair_score(ROLE, GRACE, ADA) == 0.3

    def test_categories_do_not_share_scores(self, cache):
        cache.set_pair_score(ROLE, ADA, GRACE, 0.8)
        assert cache.get_pair_score("industry", ADA, GRACE) is None


class TestCacheStatus:
    def test_an_empty_cache_reports_every_pair_missing(self, cache):
        status = cache.get_complementarity_cache_status(
            [ADA, GRACE], [ADA, GRACE], ROLE
        )

        assert status.cached == {}
        assert status.missing == {ADA: [GRACE], GRACE: [ADA]}

    def test_self_comparison_is_never_requested(self, cache):
        status = cache.get_complementarity_cache_status([ADA], [ADA, GRACE], ROLE)

        assert status.missing == {ADA: [GRACE]}

    def test_a_fully_cached_target_is_not_requested_again(self, cache):
        cache.set_pair_score(ROLE, ADA, GRACE, 0.8)

        status = cache.get_complementarity_cache_status(
            [ADA, GRACE], [ADA, GRACE], ROLE
        )

        assert status.cached[ADA] == {GRACE: 0.8}
        assert ADA not in status.missing

    def test_adding_one_person_reuses_every_previously_scored_pair(self, cache):
        cache.cache_complementarity_scores({ADA: {GRACE: 0.8}, GRACE: {ADA: 0.4}}, ROLE)

        roster = [ADA, GRACE, KATHERINE]
        status = cache.get_complementarity_cache_status(roster, roster, ROLE)

        assert status.cached[ADA] == {GRACE: 0.8}
        assert status.cached[GRACE] == {ADA: 0.4}
        assert status.missing == {
            ADA: [KATHERINE],
            GRACE: [KATHERINE],
            KATHERINE: [ADA, GRACE],
        }

    def test_removing_a_person_leaves_the_rest_cached(self, cache):
        roster = [ADA, GRACE, KATHERINE]
        cache.cache_complementarity_scores(
            {
                ADA: {GRACE: 0.8, KATHERINE: 0.2},
                GRACE: {ADA: 0.4, KATHERINE: 0.6},
                KATHERINE: {ADA: 0.1, GRACE: 0.9},
            },
            ROLE,
        )

        status = cache.get_complementarity_cache_status(
            [ADA, GRACE], [ADA, GRACE], ROLE
        )

        assert status.missing == {}
        assert status.cached == {ADA: {GRACE: 0.8}, GRACE: {ADA: 0.4}}
