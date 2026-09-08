import logging
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import pandas as pd

from match_engine.services.features.feature_set import FeatureSet

logger = logging.getLogger(__name__)


class PersonNotFound(LookupError):
    """The roster holds nobody at that position or under that name."""


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    similarity: float
    complementarity: float


@dataclass(frozen=True)
class Match:
    position: int
    name: str
    company: str
    weight: float
    features: List[FeatureContribution]


class Recommender:
    """Answers who one person should talk to, out of the graph a run built.

    Every pair already carries a scored edge, so the per-person question is a
    sort over neighbours rather than new work. Edges hold only the combined
    weight, so the per-feature breakdown is recomputed from the calibrated
    matrices at query time."""

    def __init__(
        self,
        graph: nx.Graph,
        similarity_calc,
        matrix_builder,
        feature_set: FeatureSet,
        df: pd.DataFrame,
    ):
        self.graph = graph
        self.similarity_calc = similarity_calc
        self.matrix_builder = matrix_builder
        self.feature_set = feature_set
        self.df = df

    def top_matches(self, position: int, k: int) -> List[Match]:
        assert k > 0, "k must be positive"
        if position not in self.graph:
            raise PersonNotFound(f"no person at position {position}")

        ranked = sorted(
            self.graph[position].items(),
            key=lambda edge: edge[1].get("weight", 0.0),
            reverse=True,
        )
        return [
            self._match(position, neighbour, data.get("weight", 0.0))
            for neighbour, data in ranked[:k]
        ]

    def position_of(self, name: str) -> int:
        """Resolve a display name to the row position everything else uses."""
        wanted = name.strip().casefold()
        column = self.df[self.feature_set.name_column].astype(str)
        for position, value in enumerate(column):
            if value.strip().casefold() == wanted:
                return position
        raise PersonNotFound(f"no person named {name!r}")

    def _match(self, source: int, target: int, weight: float) -> Match:
        row = self.df.iloc[target]
        return Match(
            position=target,
            name=str(row[self.feature_set.name_column]),
            company=self._company(row),
            weight=float(weight),
            features=self._contributions(source, target),
        )

    def _company(self, row: pd.Series) -> str:
        if self.feature_set.company_column:
            return str(row[self.feature_set.company_column])
        return ""

    def _contributions(self, source: int, target: int) -> List[FeatureContribution]:
        similarities = self.similarity_calc.get_all_similarities(source, target)
        complementarities = self.matrix_builder.get_all_complementarities(
            source, target
        )
        return [
            FeatureContribution(
                feature=name,
                similarity=float(similarities.get(name, 0.0)),
                complementarity=float(complementarities.get(name, 0.0)),
            )
            for name in self.feature_set.names
        ]


def recommender_for(builder) -> Recommender:
    """Build a recommender over a graph the given run already produced."""
    assert (
        builder.graph is not None
    ), "the graph must be built before ranking neighbours"
    return Recommender(
        graph=builder.graph,
        similarity_calc=builder.similarity_calc,
        matrix_builder=builder.matrix_builder,
        feature_set=builder.feature_set,
        df=builder.df,
    )
