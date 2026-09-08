import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from match_engine.services.analysis.matrix_builder import MatrixBuilder
from match_engine.services.cache.app_cache_service import AppCacheService
from match_engine.services.cache.memory import InMemoryBackend
from match_engine.services.graph.graph_builder import GraphBuilder
from match_engine.services.graph.scoring.similarity_calculator import (
    SimilarityCalculator,
)
from match_engine.services.preprocessing.csv_loader import CSVLoader
from match_engine.services.preprocessing.embedding_builder import EmbeddingBuilder
from match_engine.services.scoring.llm_scorer import LLMComplementarityScorer
from tests.unit.test_graph_alignment import MAPPING, StubEmbeddingService

SAMPLE_CSV = "docs/sample.csv"
ROSTER_SIZE = 12


class ScriptedModel:
    """Returns a well-formed reply for whatever shape the analyzer asks for."""

    def __init__(self):
        self.calls = 0
        self.with_raw_response = self

    async def create(self, model, messages, temperature, max_tokens):
        prompt = messages[0]["content"]
        target_count = int(
            prompt.split('"scores" holds exactly ')[1].split(" arrays")[0]
        )
        comparison_count = int(
            prompt.split("Each array holds exactly ")[1].split(" numbers")[0].strip()
        )
        self.calls += 1
        payload = {
            "scores": [
                [
                    round(0.3 + 0.01 * ((row + column) % 40), 2)
                    for column in range(comparison_count)
                ]
                for row in range(target_count)
            ]
        }
        return _RawResponse(json.dumps(payload))


class _RawResponse:
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


@pytest.fixture
def roster(tmp_path):
    df = pd.read_csv(SAMPLE_CSV).head(ROSTER_SIZE)
    path = tmp_path / "roster.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def builder(roster):
    model = ScriptedModel()
    client = MagicMock()
    client.chat.completions = model

    cache = AppCacheService(backend=InMemoryBackend())
    matrix_builder = MatrixBuilder(
        scorer=LLMComplementarityScorer(openai_client=client, cache=cache)
    )

    graph_builder = GraphBuilder(
        csv_path=str(roster),
        csv_loader=CSVLoader(str(roster), MAPPING),
        embedding_builder=EmbeddingBuilder(
            cache=cache, embedding_service=StubEmbeddingService()
        ),
        similarity_calc=SimilarityCalculator(),
        matrix_builder=matrix_builder,
        subgraph_analyzer=MagicMock(),
        cache=MagicMock(),
    )
    graph_builder.model = model
    return graph_builder


class TestPipeline:
    async def test_a_run_scores_every_pair_without_falling_back(self, builder):
        builder.load_data()
        embeddings = await builder.embed_features()

        await builder.create_graph_optimized(embeddings)

        report = builder.matrix_builder.scoring_report
        assert report.fallback_pairs == 0
        assert report.fallback_rate == 0.0
        assert report.model_calls == builder.model.calls

    async def test_the_graph_covers_every_person_exactly_once(self, builder):
        builder.load_data()
        embeddings = await builder.embed_features()

        graph = await builder.create_graph_optimized(embeddings)

        assert set(graph.nodes) == set(range(len(builder.df)))
        assert graph.number_of_edges() == len(builder.df) * (len(builder.df) - 1) // 2

    async def test_a_second_run_reuses_every_scored_pair(self, builder):
        builder.load_data()
        embeddings = await builder.embed_features()
        await builder.create_graph_optimized(embeddings)

        calls_after_first_run = builder.model.calls
        await builder.create_graph_optimized(embeddings)

        assert builder.model.calls == calls_after_first_run
        assert builder.matrix_builder.scoring_report.scored_pairs == 0


FOREIGN_ROWS = [
    ("Ada", "Analytical", "engine design | mathematics", "funding", "founder"),
    ("Grace", "Univac", "compilers | tooling", "hires", "principal"),
    ("Katherine", "NACA", "orbital mechanics | analysis", "collaborators", "principal"),
    ("Dorothy", "IBM", "systems research | fortran", "advisors", "director"),
    ("Mary", "Bell", "switching theory | logic", "funding", "staff"),
    ("Radia", "DEC", "routing | protocols", "hires", "principal"),
]


@pytest.fixture
def foreign_roster(tmp_path):
    """A roster whose column names share nothing with the vendor layout."""
    df = pd.DataFrame(
        FOREIGN_ROWS,
        columns=["Attendee", "Employer", "What they do", "Looking for", "Seniority"],
    )
    path = tmp_path / "foreign.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def foreign_builder(foreign_roster):
    model = ScriptedModel()
    client = MagicMock()
    client.chat.completions = model

    cache = AppCacheService(backend=InMemoryBackend())
    graph_builder = GraphBuilder(
        csv_path=str(foreign_roster),
        min_density=0.0,
        embedding_builder=EmbeddingBuilder(
            cache=cache, embedding_service=StubEmbeddingService()
        ),
        matrix_builder=MatrixBuilder(
            scorer=LLMComplementarityScorer(openai_client=client, cache=cache)
        ),
        cache=MagicMock(),
    )
    graph_builder.model = model
    return graph_builder


class TestForeignSchema:
    async def test_a_roster_with_unknown_columns_returns_a_ranked_group(
        self, foreign_builder
    ):
        foreign_builder.load_data()
        embeddings = await foreign_builder.embed_features()
        await foreign_builder.create_graph_optimized(embeddings)

        nodes, density = foreign_builder.find_largest_dense_subgraph()

        assert len(nodes) >= 3
        assert density > 0.0

    async def test_the_features_come_from_the_dataset(self, foreign_builder):
        foreign_builder.load_data()

        assert "What they do" in foreign_builder.feature_set.columns.values()
        assert foreign_builder.feature_set.name_column == "Attendee"

    async def test_weights_are_produced_without_a_prompt_or_a_key(
        self, foreign_builder
    ):
        foreign_builder.load_data()
        embeddings = await foreign_builder.embed_features()
        await foreign_builder.create_graph_optimized(embeddings)

        assert set(foreign_builder.tuned_w_s) == set(foreign_builder.feature_set.names)
        assert sum(foreign_builder.tuned_w_s.values()) == pytest.approx(1.0)
        assert sum(foreign_builder.tuned_w_c.values()) == pytest.approx(1.0)


class TestDensityThreshold:
    async def test_a_threshold_below_the_graph_density_returns_everyone(
        self, foreign_builder, caplog
    ):
        foreign_builder.load_data()
        embeddings = await foreign_builder.embed_features()
        await foreign_builder.create_graph_optimized(embeddings)
        foreign_builder.min_density = 0.0

        nodes, _ = foreign_builder.find_largest_dense_subgraph()

        assert len(nodes) == foreign_builder.graph.number_of_nodes()
        assert "whole roster came back" in caplog.text

    async def test_an_unreachable_threshold_returns_nobody(self, foreign_builder):
        foreign_builder.load_data()
        embeddings = await foreign_builder.embed_features()
        await foreign_builder.create_graph_optimized(embeddings)
        foreign_builder.min_density = 2.0

        nodes, density = foreign_builder.find_largest_dense_subgraph()

        assert nodes == set()
        assert density == 0.0
