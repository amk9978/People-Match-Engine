from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringReport:
    """What a complementarity run actually asked for and actually got back.

    fallback_pairs counts scores the model did not supply, which are filled with
    the neutral value. A run with a high fallback rate is reporting an opinion
    the model never gave, so the rate travels with the result.

    scorer names the implementation behind the numbers. The offline path reports
    no model calls and no fallbacks, which leaves the rate uninformative on its
    own, and the name is what tells the two apart."""

    scorer: str = "llm"
    cached_pairs: int = 0
    scored_pairs: int = 0
    fallback_pairs: int = 0
    model_calls: int = 0

    @property
    def total_pairs(self) -> int:
        return self.cached_pairs + self.scored_pairs + self.fallback_pairs

    @property
    def fallback_rate(self) -> float:
        if self.total_pairs == 0:
            return 0.0
        return self.fallback_pairs / self.total_pairs

    def merge(self, other: "ScoringReport") -> "ScoringReport":
        return ScoringReport(
            scorer=other.scorer,
            cached_pairs=self.cached_pairs + other.cached_pairs,
            scored_pairs=self.scored_pairs + other.scored_pairs,
            fallback_pairs=self.fallback_pairs + other.fallback_pairs,
            model_calls=self.model_calls + other.model_calls,
        )

    def to_dict(self) -> dict:
        return {
            "scorer": self.scorer,
            "cached_pairs": self.cached_pairs,
            "scored_pairs": self.scored_pairs,
            "fallback_pairs": self.fallback_pairs,
            "model_calls": self.model_calls,
            "fallback_rate": round(self.fallback_rate, 4),
        }
