import math
from typing import Dict, Iterable, Mapping

from match_engine.services.scoring.profile import ScoringProfile


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _norm_weights(weights: Mapping[str, float], features: Iterable[str]):
    """Weights for the given features, summing to one, uniform if none are positive."""
    features = list(features)
    total = sum(max(0.0, float(weights.get(name, 0.0))) for name in features)
    if total <= 0:
        return {name: 1.0 / len(features) for name in features}
    return {name: max(0.0, float(weights.get(name, 0.0))) / total for name in features}


def _power_mean(values, weights, p: float, eps: float = 1e-9) -> float:
    """Weighted generalized mean. p == 0 is the geometric mean, p < 0 is stricter."""
    names = [name for name in values if name in weights]
    if not names:
        return 0.0

    scores = [_clamp01(values[name]) for name in names]
    shares = [weights[name] for name in names]
    total = sum(shares)
    if total > 0:
        shares = [share / total for share in shares]
    else:
        shares = [1.0 / len(scores)] * len(scores)

    if abs(p) < 1e-12:
        return math.exp(
            sum(
                share * math.log(max(eps, score))
                for score, share in zip(scores, shares)
            )
        )
    return sum(share * (score**p) for score, share in zip(scores, shares)) ** (1.0 / p)


def combine_edge_weight(
    sim: Dict[str, float],
    comp: Dict[str, float],
    w_s: Mapping[str, float],
    w_c: Mapping[str, float],
    profile: ScoringProfile = None,
) -> float:
    """Blend per-feature similarity and complementarity into one edge weight in [0, 1].

    Each signal is collapsed by its own weighted power mean, then the two are
    combined additively and geometrically and mixed. An edge strong on one signal
    and weak on the other is penalized, and the tails are compressed so peeling
    sees well-behaved degrees.
    """
    profile = profile or ScoringProfile()

    features = [name for name in sim if name in comp]
    if not features:
        return 0.0

    weights_s = _norm_weights(w_s, features)
    weights_c = _norm_weights(w_c, features)

    similarity = _clamp01(
        _power_mean({name: sim[name] for name in features}, weights_s, profile.p_s)
    )
    complementarity = _clamp01(
        _power_mean({name: comp[name] for name in features}, weights_c, profile.p_c)
    )

    additive = profile.rho * similarity + (1.0 - profile.rho) * complementarity
    geometric = (similarity ** max(0.0, profile.rho)) * (
        complementarity ** max(0.0, 1.0 - profile.rho)
    )
    edge = profile.lam * additive + (1.0 - profile.lam) * geometric

    edge *= min(similarity, complementarity) ** max(0.0, profile.eta)

    return _clamp01(_clamp01(edge) ** profile.gamma_e)
