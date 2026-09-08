import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from services.analysis.business_analyzer import BusinessAnalyzer
from services.analysis.matrix_builder import MatrixBuilder
from services.cache.app_cache_service import AppCacheService
from services.cache.memory import InMemoryBackend
from services.graph.graph_builder import GraphBuilder
from services.graph.scoring.similarity_calculator import SimilarityCalculator
from services.preprocessing.csv_loader import CSVLoader
from services.preprocessing.embedding_builder import EmbeddingBuilder
from tests.unit.test_graph_alignment import StubEmbeddingService

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
        business_analyzer=BusinessAnalyzer(openai_client=client, cache=cache)
    )

    graph_builder = GraphBuilder(
        csv_path=str(roster),
        csv_loader=CSVLoader(str(roster)),
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
    async def test_a_run_scores_every_pair_without_falling_back(
        self, builder, monkeypatch
    ):
        monkeypatch.setattr(
            "services.graph.graph_builder.tune_parameters",
            lambda prompt, insights: ({}, {}),
        )
        builder.load_data()
        embeddings = await builder.embed_features()

        await builder.create_graph_optimized(embeddings)

        report = builder.matrix_builder.scoring_report
        assert report.fallback_pairs == 0
        assert report.fallback_rate == 0.0
        assert report.model_calls == builder.model.calls

    async def test_the_graph_covers_every_person_exactly_once(
        self, builder, monkeypatch
    ):
        monkeypatch.setattr(
            "services.graph.graph_builder.tune_parameters",
            lambda prompt, insights: ({}, {}),
        )
        builder.load_data()
        embeddings = await builder.embed_features()

        graph = await builder.create_graph_optimized(embeddings)

        assert set(graph.nodes) == set(range(len(builder.df)))
        assert graph.number_of_edges() == len(builder.df) * (len(builder.df) - 1) // 2

    async def test_a_second_run_reuses_every_scored_pair(self, builder, monkeypatch):
        monkeypatch.setattr(
            "services.graph.graph_builder.tune_parameters",
            lambda prompt, insights: ({}, {}),
        )
        builder.load_data()
        embeddings = await builder.embed_features()
        await builder.create_graph_optimized(embeddings)

        calls_after_first_run = builder.model.calls
        await builder.create_graph_optimized(embeddings)

        assert builder.model.calls == calls_after_first_run
        assert builder.matrix_builder.scoring_report.scored_pairs == 0
