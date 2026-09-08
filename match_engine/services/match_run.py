import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from match_engine.services.analysis.matrix_builder import MatrixBuilder
from match_engine.services.graph.graph_builder import GraphBuilder
from match_engine.services.scoring.complementarity_scorer import ComplementarityScorer
from match_engine.services.scoring.profile import ScoringProfile
from match_engine.services.scoring.report import ScoringReport
from match_engine.services.scoring.scorer_factory import create_complementarity_scorer
from match_engine.services.scoring.weights import (
    ExplicitWeightResolver,
    ExplicitWeights,
)

logger = logging.getLogger(__name__)

LOADING = "Loading data"
EMBEDDING = "Creating feature embeddings"
SCORING = "Scoring pairs and building the graph"
PEELING = "Finding the densest subgraph"
ANALYZING = "Analyzing the group"


@dataclass(frozen=True)
class MatchRequest:
    """Everything one run needs, with no reference to how it was requested."""

    csv_path: str
    prompt: Optional[str] = None
    min_density: Optional[float] = None
    mapping_path: Optional[str] = None
    scoring_profile: Optional[ScoringProfile] = None
    weights: Optional[ExplicitWeights] = None
    scorer: Optional[ComplementarityScorer] = None
    scorer_choice: Optional[str] = None


@dataclass(frozen=True)
class MatchResult:
    nodes: List[int]
    names: List[str]
    density: float
    row_count: int
    report: ScoringReport
    info: Dict = field(default_factory=dict)


class ProgressReporter(Protocol):
    """Receives one line per pipeline stage, in the order the stages run."""

    async def report(self, stage: str) -> None: ...


class NullProgress:
    async def report(self, stage: str) -> None:
        return None


class RecordingProgress:
    """Keeps the stages in order, for tests and for a CLI that prints them."""

    def __init__(self):
        self.stages: List[str] = []

    async def report(self, stage: str) -> None:
        self.stages.append(stage)


class MatchRun:
    """The one path from a CSV to a scored group.

    A run loads the roster, embeds each feature, scores every pair, peels the
    graph down to its densest subgraph, and describes what it found. Callers
    differ only in how they report progress and what they do with the result, so
    the job store, the websocket, and the CLI all sit outside this class."""

    def __init__(
        self,
        request: MatchRequest,
        progress: ProgressReporter = None,
        graph_builder: GraphBuilder = None,
    ):
        self.request = request
        self.progress = progress or NullProgress()
        self.graph_builder = graph_builder or self._build(request)

    def _build(self, request: MatchRequest) -> GraphBuilder:
        weight_resolver = None
        if request.weights is not None:
            weight_resolver = ExplicitWeightResolver(request.weights)

        scorer = request.scorer
        if scorer is None and request.scorer_choice is not None:
            scorer = create_complementarity_scorer(request.scorer_choice)

        matrix_builder = None
        if scorer is not None:
            matrix_builder = MatrixBuilder(scorer=scorer)

        return GraphBuilder(
            csv_path=request.csv_path,
            min_density=request.min_density,
            mapping_path=request.mapping_path,
            matrix_builder=matrix_builder,
            weight_resolver=weight_resolver,
            scoring_profile=request.scoring_profile,
        )

    async def execute(self) -> MatchResult:
        builder = self.graph_builder

        await self.progress.report(LOADING)
        builder.load_data()
        logger.info(f"Loaded {len(builder.df)} rows from {self.request.csv_path}")

        await self.progress.report(EMBEDDING)
        embeddings = await builder.embed_features()

        await self.progress.report(SCORING)
        await builder.create_graph_optimized(embeddings, self.request.prompt)

        await self.progress.report(PEELING)
        nodes, density = builder.find_largest_dense_subgraph()

        await self.progress.report(ANALYZING)
        info = builder.get_subgraph_info(nodes, embeddings)

        return MatchResult(
            nodes=sorted(nodes),
            names=self._names(builder),
            density=density,
            row_count=len(builder.df),
            report=builder.matrix_builder.scoring_report,
            info=info,
        )

    def _names(self, builder: GraphBuilder) -> List[str]:
        column = builder.feature_set.name_column
        return [str(value) for value in builder.df[column]]
