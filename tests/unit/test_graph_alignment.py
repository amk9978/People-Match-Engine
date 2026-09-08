from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from match_engine.services.features.feature_set import Feature, FeatureSet
from match_engine.services.graph.graph_builder import GraphBuilder
from match_engine.services.preprocessing.csv_loader import CSVLoader
from match_engine.services.preprocessing.embedding_builder import EmbeddingBuilder

MAPPING = "presets/vendor_six_column.yaml"

MARKET = Feature(
    name="market", column="Company Market - Market Traction", separator="|"
)
FEATURE_SET = FeatureSet(
    features=(MARKET,), name_column="Person Name", company_column="Person Company"
)

COLUMNS = [
    "Person Name",
    "Person Title",
    "Person Company",
    "Professional Identity - Role Specification",
    "Professional Identity - Experience Level",
    "Company Identity - Industry Classification",
    "Company Market - Market Traction",
    "Company Offering - Value Proposition",
    "All Persona Titles",
]


def write_roster(path, names, incomplete_positions):
    """Write a CSV where the named positions have one essential column blanked."""
    rows = []
    for position, name in enumerate(names):
        row = {column: f"{column} value {position}" for column in COLUMNS}
        row["Person Name"] = name
        row["Person Company"] = f"Company {position}"
        if position in incomplete_positions:
            row["Company Market - Market Traction"] = ""
        rows.append(row)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture
def roster_with_a_gap(tmp_path):
    return write_roster(
        tmp_path / "roster.csv",
        ["Ada", "Grace", "Katherine", "Dorothy", "Mary"],
        incomplete_positions={2},
    )


def build_graph_builder(csv_path, people_count):
    """A GraphBuilder whose embedding, scoring and analysis collaborators are stubbed."""
    flat = np.full((people_count, people_count), 0.5)

    similarity_calc = MagicMock()
    similarity_calc.raw_matrices.return_value = {"role": flat}
    similarity_calc.get_all_similarities.return_value = {"role": 0.5}

    matrix_builder = MagicMock()
    matrix_builder.build = AsyncMock(return_value=None)
    matrix_builder.index_people = MagicMock(return_value=None)
    matrix_builder.raw_matrices.return_value = {"role": flat}
    matrix_builder.get_all_complementarities.return_value = {"role": 0.5}

    return GraphBuilder(
        csv_path=str(csv_path),
        csv_loader=CSVLoader(str(csv_path), MAPPING),
        embedding_builder=MagicMock(),
        similarity_calc=similarity_calc,
        matrix_builder=matrix_builder,
        subgraph_analyzer=MagicMock(),
        cache=MagicMock(),
    )


class TestLoaderIndex:
    def test_filtering_leaves_a_contiguous_index(self, roster_with_a_gap):
        df = CSVLoader(str(roster_with_a_gap), MAPPING).load_data()

        assert len(df) == 4
        assert list(df.index) == [0, 1, 2, 3]

    def test_filtering_keeps_the_surviving_people(self, roster_with_a_gap):
        df = CSVLoader(str(roster_with_a_gap), MAPPING).load_data()

        assert list(df["Person Name"]) == ["Ada", "Grace", "Dorothy", "Mary"]


class TestGraphAlignment:
    async def test_node_set_equals_the_dataframe_index(self, roster_with_a_gap):
        builder = build_graph_builder(roster_with_a_gap, people_count=4)
        builder.load_data()

        graph = await builder.create_graph_optimized({"role": np.zeros((4, 8))})

        assert set(graph.nodes) == set(builder.df.index)

    async def test_no_person_is_left_without_edges(self, roster_with_a_gap):
        builder = build_graph_builder(roster_with_a_gap, people_count=4)
        builder.load_data()

        graph = await builder.create_graph_optimized({"role": np.zeros((4, 8))})

        orphans = [node for node in graph.nodes if graph.degree(node) == 0]
        assert orphans == []

    async def test_node_labels_name_the_person_scored_at_that_position(
        self, roster_with_a_gap
    ):
        builder = build_graph_builder(roster_with_a_gap, people_count=4)
        builder.load_data()

        graph = await builder.create_graph_optimized({"role": np.zeros((4, 8))})

        for position, name in enumerate(builder.df["Person Name"]):
            assert graph.nodes[position]["name"] == name


class StubEmbeddingService:
    """Deterministic embeddings so a row can be traced back to the text that made it."""

    embedding_dim = 8

    def _vector(self, text: str) -> list:
        digest = abs(hash(text))
        return [float((digest >> shift) & 0xFF) for shift in range(0, 64, 8)]

    async def get_embedding(self, text: str) -> list:
        return self._vector(text)

    async def get_embedding_array(self, text: str) -> np.ndarray:
        return np.array(self._vector(text))

    async def get_batch_embeddings(self, texts: list) -> list:
        return [self._vector(text) for text in texts]


class TestEmbeddingAlignment:
    async def test_matrix_row_belongs_to_the_person_at_that_position(
        self, roster_with_a_gap
    ):
        df = CSVLoader(str(roster_with_a_gap), MAPPING).load_data()
        builder = EmbeddingBuilder(embedding_service=StubEmbeddingService())

        matrices = await builder.embed_features(df, FEATURE_SET)
        matrix = matrices["market"]

        assert matrix.shape == (len(df), StubEmbeddingService.embedding_dim)

        stub = StubEmbeddingService()
        for position in range(len(df)):
            text = df.iloc[position]["Company Market - Market Traction"]
            expected = np.array(stub._vector(text))
            expected = expected / np.linalg.norm(expected)
            assert np.allclose(matrix[position], expected)

    async def test_a_gapped_frame_is_rejected(self, roster_with_a_gap):
        df = CSVLoader(str(roster_with_a_gap), MAPPING).load_data()
        gapped = df.drop(index=1)
        builder = EmbeddingBuilder(embedding_service=StubEmbeddingService())

        with pytest.raises(AssertionError):
            await builder.embed_features(gapped, FEATURE_SET)
