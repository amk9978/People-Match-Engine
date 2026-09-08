from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringProfile:
    """The shape constants of the edge weight combiner, fixed once per run.

    p_s and p_c are power-mean exponents, 0 giving the geometric mean. rho trades
    similarity against complementarity, lam trades the additive blend against the
    multiplicative one, eta penalizes an edge that is strong on one signal and
    weak on the other, and gamma_e compresses the tails so peeling sees
    well-behaved degrees.
    """

    p_s: float = 0.0
    p_c: float = 0.5
    rho: float = 0.5
    lam: float = 0.5
    eta: float = 0.2
    gamma_e: float = 0.85
