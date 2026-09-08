import networkx as nx
import pandas as pd
import pytest

from match_engine.services.features.feature_set import Feature, FeatureSet
from match_engine.services.graph.recommendations import (
    PersonNotFound,
    Recommender,
)

FEATURE_SET = FeatureSet(
    features=(
        Feature(name="role", column="Role", separator="|"),
        Feature(name="industry", column="Industry", separator="|"),
    ),
    name_column="Name",
    company_column="Company",
)

ROWS = [
    ("Ada", "Analytical", "engineer", "computing"),
    ("Grace", "Univac", "compiler", "computing"),
    ("Katherine", "NACA", "mathematician", "aerospace"),
    ("Dorothy", "IBM", "manager", "computing"),
]

WEIGHTS = {(0, 1): 0.9, (0, 2): 0.4, (0, 3): 0.7, (1, 2): 0.2, (1, 3): 0.5, (2, 3): 0.1}


class StubMatrices:
    """Returns a fixed per-feature score for any pair."""

    def __init__(self, value):
        self.value = value

    def get_all_similarities(self, i, j):
        return {"role": self.value, "industry": self.value / 2}

    def get_all_complementarities(self, i, j):
        return {"role": 1.0 - self.value, "industry": 0.5}


@pytest.fixture
def recommender():
    df = pd.DataFrame(ROWS, columns=["Name", "Company", "Role", "Industry"])
    graph = nx.Graph()
    graph.add_nodes_from(range(len(ROWS)))
    for (i, j), weight in WEIGHTS.items():
        graph.add_edge(i, j, weight=weight)

    return Recommender(
        graph=graph,
        similarity_calc=StubMatrices(0.8),
        matrix_builder=StubMatrices(0.8),
        feature_set=FEATURE_SET,
        df=df,
    )


class TestTopMatches:
    def test_matches_come_back_heaviest_first(self, recommender):
        matches = recommender.top_matches(0, k=3)

        assert [match.weight for match in matches] == [0.9, 0.7, 0.4]

    def test_k_caps_the_list(self, recommender):
        assert len(recommender.top_matches(0, k=1)) == 1

    def test_asking_for_more_than_exist_returns_every_neighbour(self, recommender):
        assert len(recommender.top_matches(0, k=99)) == 3

    def test_a_match_names_the_person(self, recommender):
        assert recommender.top_matches(0, k=1)[0].name == "Grace"

    def test_a_match_names_the_company(self, recommender):
        assert recommender.top_matches(0, k=1)[0].company == "Univac"

    def test_a_person_is_never_their_own_match(self, recommender):
        assert all(match.position != 0 for match in recommender.top_matches(0, k=99))

    def test_every_feature_carries_both_signals(self, recommender):
        match = recommender.top_matches(0, k=1)[0]

        assert [f.feature for f in match.features] == ["role", "industry"]
        assert match.features[0].similarity == 0.8
        assert match.features[0].complementarity == pytest.approx(0.2)

    def test_k_must_be_positive(self, recommender):
        with pytest.raises(AssertionError):
            recommender.top_matches(0, k=0)

    def test_an_unknown_position_is_rejected(self, recommender):
        with pytest.raises(PersonNotFound):
            recommender.top_matches(99, k=3)


class TestLookupByName:
    def test_a_name_resolves_to_its_row(self, recommender):
        assert recommender.position_of("Katherine") == 2

    def test_the_lookup_ignores_case_and_padding(self, recommender):
        assert recommender.position_of("  katherine ") == 2

    def test_an_unknown_name_is_rejected(self, recommender):
        with pytest.raises(PersonNotFound):
            recommender.position_of("Hedy")

    def test_a_duplicated_name_takes_the_first_row(self):
        df = pd.DataFrame(
            [("Ada", "A", "x", "y"), ("Ada", "B", "x", "y")],
            columns=["Name", "Company", "Role", "Industry"],
        )
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=0.5)
        recommender = Recommender(
            graph=graph,
            similarity_calc=StubMatrices(0.5),
            matrix_builder=StubMatrices(0.5),
            feature_set=FEATURE_SET,
            df=df,
        )

        assert recommender.position_of("Ada") == 0
