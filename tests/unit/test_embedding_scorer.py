import numpy as np
import pytest

from match_engine.services.scoring.embedding_scorer import (
    BAND_WIDTH,
    EmbeddingComplementarityScorer,
    band,
)
from match_engine.services.scoring.informativeness import informativeness
from match_engine.services.scoring.scorer_factory import create_complementarity_scorer

DIM = 8


class StubEmbeddings:
    """Turns a profile into a vector the test chose, so cosines are exact."""

    embedding_dim = DIM

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = 0

    async def get_batch_embeddings(self, texts):
        self.calls += 1
        return [list(self.vectors[text]) for text in texts]


def _unit(*components) -> np.ndarray:
    vector = np.zeros(DIM)
    for index, value in enumerate(components):
        vector[index] = value
    return vector / np.linalg.norm(vector)


class TestBandCurve:
    def test_identical_profiles_score_zero(self):
        assert band(np.array([1.0]), median=0.5)[0] == pytest.approx(0.0)

    def test_a_near_identical_pair_scores_almost_zero(self):
        assert band(np.array([0.98]), median=0.5)[0] < 0.02

    def test_an_unrelated_pair_scores_below_the_peak(self):
        peak = band(np.array([0.5]), median=0.5)[0]
        unrelated = band(np.array([-0.4]), median=0.5)[0]

        assert unrelated < peak

    def test_the_peak_sits_near_the_median(self):
        grid = np.linspace(-1.0, 1.0, 401)
        scores = band(grid, median=0.6)

        assert grid[int(np.argmax(scores))] == pytest.approx(0.6, abs=0.1)

    def test_the_width_is_a_constant_not_the_features_own_spread(self):
        narrow = band(np.linspace(0.40, 0.45, 20), median=0.425)
        wide = band(np.linspace(0.0, 0.9, 20), median=0.45)

        assert narrow.max() - narrow.min() < wide.max() - wide.min()

    def test_scores_stay_inside_the_unit_interval(self):
        scores = band(np.linspace(-1.0, 1.0, 201), median=0.3)

        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


class TestEmbeddingScorer:
    @pytest.fixture
    def spread_profiles(self):
        return {
            "backend engineer": _unit(1.0),
            "frontend engineer": _unit(0.9, 0.4),
            "product manager": _unit(0.2, 1.0),
            "venture investor": _unit(0.0, 0.1, 1.0),
        }

    async def test_every_pair_gets_a_score_with_no_model_call(self, spread_profiles):
        scorer = EmbeddingComplementarityScorer(
            embedding_service=StubEmbeddings(spread_profiles)
        )
        profiles = list(spread_profiles)

        result = await scorer.get_profile_complementarity(profiles, profiles, "role")

        assert set(result.scores) == set(profiles)
        assert all(set(row) == set(profiles) for row in result.scores.values())
        assert result.report.model_calls == 0
        assert result.report.fallback_pairs == 0
        assert result.report.scored_pairs == len(profiles) ** 2

    async def test_the_report_names_the_scorer(self, spread_profiles):
        scorer = EmbeddingComplementarityScorer(
            embedding_service=StubEmbeddings(spread_profiles)
        )
        profiles = list(spread_profiles)

        result = await scorer.get_profile_complementarity(profiles, profiles, "role")

        assert result.report.scorer == "embedding"

    async def test_a_profile_against_itself_scores_zero(self, spread_profiles):
        scorer = EmbeddingComplementarityScorer(
            embedding_service=StubEmbeddings(spread_profiles)
        )
        profiles = list(spread_profiles)

        result = await scorer.get_profile_complementarity(profiles, profiles, "role")

        for profile in profiles:
            assert result.scores[profile][profile] == pytest.approx(0.0, abs=1e-9)

    async def test_the_matrix_is_symmetric(self, spread_profiles):
        scorer = EmbeddingComplementarityScorer(
            embedding_service=StubEmbeddings(spread_profiles)
        )
        profiles = list(spread_profiles)

        result = await scorer.get_profile_complementarity(profiles, profiles, "role")

        for source in profiles:
            for target in profiles:
                assert result.scores[source][target] == pytest.approx(
                    result.scores[target][source]
                )

    async def test_each_profile_is_embedded_once(self, spread_profiles):
        embeddings = StubEmbeddings(spread_profiles)
        scorer = EmbeddingComplementarityScorer(embedding_service=embeddings)
        profiles = list(spread_profiles)

        await scorer.get_profile_complementarity(profiles, profiles, "role")

        assert embeddings.calls == 1

    async def test_a_degenerate_feature_reads_as_uninformative(self):
        """Four profiles that all mean the same thing must not win weight."""
        alike = {
            "software engineer": _unit(1.0, 0.02),
            "software developer": _unit(1.0, 0.03),
            "swe": _unit(1.0, 0.04),
            "engineer, software": _unit(1.0, 0.05),
        }
        varied = {
            "software engineer": _unit(1.0),
            "chief marketing officer": _unit(0.1, 1.0),
            "seed investor": _unit(0.0, 0.1, 1.0),
            "clinical researcher": _unit(0.0, 0.0, 0.1, 1.0),
        }

        alike_matrix = await _matrix(alike)
        varied_matrix = await _matrix(varied)

        assert informativeness(alike_matrix) < informativeness(varied_matrix)

    async def test_one_profile_is_handled_without_dividing_by_zero(self):
        scorer = EmbeddingComplementarityScorer(
            embedding_service=StubEmbeddings({"solo": _unit(1.0)})
        )

        result = await scorer.get_profile_complementarity(["solo"], ["solo"], "role")

        assert result.scores["solo"]["solo"] == pytest.approx(0.0, abs=1e-9)

    async def test_an_empty_feature_returns_nothing(self):
        scorer = EmbeddingComplementarityScorer(embedding_service=StubEmbeddings({}))

        result = await scorer.get_profile_complementarity([], [], "role")

        assert result.scores == {}
        assert result.report.total_pairs == 0


async def _matrix(profiles):
    scorer = EmbeddingComplementarityScorer(embedding_service=StubEmbeddings(profiles))
    names = list(profiles)
    result = await scorer.get_profile_complementarity(names, names, "role")
    return np.array([[result.scores[a][b] for b in names] for a in names])


class TestScorerSelection:
    def test_an_explicit_choice_wins(self):
        scorer = create_complementarity_scorer("embedding")

        assert isinstance(scorer, EmbeddingComplementarityScorer)

    def test_auto_without_a_key_stays_offline(self, monkeypatch):
        monkeypatch.setattr("match_engine.settings.OPENAI_API_KEY", "")

        scorer = create_complementarity_scorer("auto")

        assert isinstance(scorer, EmbeddingComplementarityScorer)

    def test_auto_with_a_key_asks_the_model(self, monkeypatch):
        from match_engine.services.scoring.llm_scorer import LLMComplementarityScorer

        monkeypatch.setattr("match_engine.settings.OPENAI_API_KEY", "sk-test")

        scorer = create_complementarity_scorer("auto")

        assert isinstance(scorer, LLMComplementarityScorer)

    def test_an_unknown_choice_is_rejected(self):
        with pytest.raises(ValueError):
            create_complementarity_scorer("magic")
