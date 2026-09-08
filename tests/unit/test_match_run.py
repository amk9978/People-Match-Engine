import pandas as pd
import pytest

from match_engine.services.match_run import (
    MatchRequest,
    MatchRun,
    RecordingProgress,
)
from match_engine.services.scoring.profile import ScoringProfile
from match_engine.services.scoring.weights import (
    ExplicitWeightResolver,
    ExplicitWeights,
)
from tests.unit.test_graph_alignment import MAPPING, StubEmbeddingService
from tests.unit.test_pipeline_smoke import FOREIGN_ROWS, ScriptedModel

SAMPLE_CSV = "docs/sample.csv"


@pytest.fixture
def roster(tmp_path):
    df = pd.read_csv(SAMPLE_CSV).head(8)
    path = tmp_path / "roster.csv"
    df.to_csv(path, index=False)
    return str(path)


def _builder(csv_path, **kwargs):
    from unittest.mock import MagicMock

    from match_engine.services.analysis.matrix_builder import MatrixBuilder
    from match_engine.services.cache.app_cache_service import AppCacheService
    from match_engine.services.cache.memory import InMemoryBackend
    from match_engine.services.graph.graph_builder import GraphBuilder
    from match_engine.services.preprocessing.embedding_builder import EmbeddingBuilder
    from match_engine.services.scoring.llm_scorer import LLMComplementarityScorer

    client = MagicMock()
    client.chat.completions = ScriptedModel()
    cache = AppCacheService(backend=InMemoryBackend())

    return GraphBuilder(
        csv_path=csv_path,
        mapping_path=MAPPING,
        embedding_builder=EmbeddingBuilder(
            cache=cache, embedding_service=StubEmbeddingService()
        ),
        matrix_builder=MatrixBuilder(
            scorer=LLMComplementarityScorer(openai_client=client, cache=cache)
        ),
        cache=MagicMock(),
        **kwargs,
    )


class TestMatchRun:
    async def test_a_run_returns_a_group_and_a_density(self, roster):
        run = MatchRun(
            MatchRequest(csv_path=roster, mapping_path=MAPPING),
            graph_builder=_builder(roster),
        )

        result = await run.execute()

        assert len(result.nodes) >= 3
        assert result.density > 0.0
        assert result.row_count == 8

    async def test_progress_arrives_in_pipeline_order(self, roster):
        progress = RecordingProgress()
        run = MatchRun(
            MatchRequest(csv_path=roster, mapping_path=MAPPING),
            progress=progress,
            graph_builder=_builder(roster),
        )

        await run.execute()

        assert progress.stages == [
            "Loading data",
            "Creating feature embeddings",
            "Scoring pairs and building the graph",
            "Finding the densest subgraph",
            "Analyzing the group",
            "Ranking each person's matches",
        ]

    async def test_the_scoring_report_travels_with_the_result(self, roster):
        run = MatchRun(
            MatchRequest(csv_path=roster, mapping_path=MAPPING),
            graph_builder=_builder(roster),
        )

        result = await run.execute()

        assert result.report.fallback_pairs == 0
        assert result.report.model_calls > 0

    async def test_a_scoring_profile_reaches_the_builder(self, roster):
        profile = ScoringProfile(rho=0.9)
        builder = _builder(roster, scoring_profile=profile)
        run = MatchRun(
            MatchRequest(
                csv_path=roster, mapping_path=MAPPING, scoring_profile=profile
            ),
            graph_builder=builder,
        )

        await run.execute()

        assert builder.scoring_profile.rho == 0.9

    async def test_every_person_gets_a_ranked_list(self, roster):
        run = MatchRun(
            MatchRequest(csv_path=roster, mapping_path=MAPPING, top_k=3),
            graph_builder=_builder(roster),
        )

        result = await run.execute()

        assert set(result.recommendations) == set(range(8))
        assert all(len(matches) == 3 for matches in result.recommendations.values())
        for position, matches in result.recommendations.items():
            assert all(match.position != position for match in matches)

    async def test_the_run_reads_names_from_the_feature_set(self, roster):
        run = MatchRun(
            MatchRequest(csv_path=roster, mapping_path=MAPPING),
            graph_builder=_builder(roster),
        )

        result = await run.execute()

        assert len(result.names) == 8
        assert all(isinstance(name, str) and name for name in result.names)


class TestExplicitWeights:
    def test_explicit_weights_replace_the_resolver(self):
        resolver = ExplicitWeightResolver(
            ExplicitWeights(
                similarity={"role": 3.0, "industry": 1.0},
                complementarity={"role": 1.0, "industry": 1.0},
            )
        )
        raw = {"role": None, "industry": None}

        w_s, w_c = resolver.resolve(raw, raw, intent=None)

        assert w_s == {"role": 0.75, "industry": 0.25}
        assert w_c == {"role": 0.5, "industry": 0.5}

    def test_weights_naming_an_unknown_feature_are_rejected(self):
        resolver = ExplicitWeightResolver(
            ExplicitWeights(
                similarity={"role": 1.0, "salary": 1.0},
                complementarity={"role": 1.0, "salary": 1.0},
            )
        )

        with pytest.raises(AssertionError):
            resolver.resolve({"role": None}, {"role": None}, intent=None)

    def test_a_missing_feature_is_rejected(self):
        resolver = ExplicitWeightResolver(
            ExplicitWeights(similarity={"role": 1.0}, complementarity={"role": 1.0})
        )

        with pytest.raises(AssertionError):
            resolver.resolve(
                {"role": None, "industry": None}, {"role": None, "industry": None}, None
            )

    async def test_a_run_with_explicit_weights_never_measures(self, roster):
        weights = ExplicitWeights(
            similarity={
                "role": 1.0,
                "experience": 0.0,
                "industry": 0.0,
                "market": 0.0,
                "offering": 0.0,
                "persona": 0.0,
            },
            complementarity={
                "role": 0.0,
                "experience": 0.0,
                "industry": 0.0,
                "market": 0.0,
                "offering": 1.0,
                "persona": 0.0,
            },
        )
        builder = _builder(roster, weight_resolver=ExplicitWeightResolver(weights))
        run = MatchRun(
            MatchRequest(csv_path=roster, mapping_path=MAPPING, weights=weights),
            graph_builder=builder,
        )

        await run.execute()

        assert builder.tuned_w_s["role"] == 1.0
        assert builder.tuned_w_c["offering"] == 1.0
