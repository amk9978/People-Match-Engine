from typing import Dict, List, Protocol

from match_engine.services.scoring.report import ScoringReport


class ScoredPairs(Protocol):
    """A block of scores and an account of where they came from."""

    scores: Dict[str, Dict[str, float]]
    report: ScoringReport


class ComplementarityScorer(Protocol):
    """Scores how complementary two profile strings are, for one feature.

    An implementation may call a model or compute locally. The report says how
    many pairs came back with a real score, so a caller can tell an answer from
    a fallback.
    """

    async def get_profile_complementarity(
        self,
        target_profiles: List[str],
        comparison_profiles: List[str],
        category: str,
    ) -> ScoredPairs: ...
