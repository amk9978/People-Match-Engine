import math

FEATURES = ["role", "experience", "industry", "market", "offering", "persona"]


def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


def _norm_weights(w, feats):
    # w can be dict or scalar; returns per-feature weights summing to 1 over provided feats
    if isinstance(w, (int, float)):
        return {f: 1.0 / len(feats) for f in feats}
    s = sum(max(0.0, float(w.get(f, 0.0))) for f in feats)
    if s <= 0:
        return {f: 1.0 / len(feats) for f in feats}
    return {f: max(0.0, float(w.get(f, 0.0))) / s for f in feats}


def _power_mean(values_by_feature, weights_by_feature, p, eps=1e-9):
    # values in [0,1]; weights nonnegative
    feats = [f for f in values_by_feature.keys() if f in weights_by_feature]
    if not feats:
        return 0.0
    vals = [_clamp01(values_by_feature[f]) for f in feats]
    w = [weights_by_feature[f] for f in feats]
    # normalize weights (already normalized upstream, but keep safe)
    s = sum(w);
    w = [wi / s for wi in w] if s > 0 else [1.0 / len(vals)] * len(vals)
    if abs(p) < 1e-12:  # geometric mean
        # avoid log(0)
        return math.exp(sum(wi * math.log(max(eps, vi)) for vi, wi in zip(vals, w)))
    return (sum(wi * (vi ** p) for vi, wi in zip(vals, w))) ** (1.0 / p)


def combine_edge_weight(
        sim, comp,
        w_s=1.0, w_c=1.0,  # per-feature dicts or scalar (uniform)
        p_s=0.0, p_c=0.5,  # power-mean exponents: 0=geom, <0 stricter (AND-like), >0 more tolerant
        rho=0.5,  # similarity vs complementarity trade (higher -> more weight on similarity)
        lam=0.5,  # additive vs multiplicative blend
        eta=0.2,  # gate lopsidedness with (min(S,C))**eta
        gamma_e=0.85  # mild compression of tails for robustness
):
    """
    sim: dict feature->similarity in [0,1]
    comp: dict feature->complementarity in [0,1]
    returns edge weight in (0,1]
    """
    feats = [f for f in FEATURES if f in sim and f in comp]
    if not feats:
        return 0.0

    w_s = _norm_weights(w_s, feats)
    w_c = _norm_weights(w_c, feats)

    S = _clamp01(_power_mean({f: sim[f] for f in feats}, w_s, p_s))
    C = _clamp01(_power_mean({f: comp[f] for f in feats}, w_c, p_c))

    # additive (compensatory) + geometric (synergistic) blend
    additive = rho * S + (1.0 - rho) * C
    geometric = (S ** max(0.0, rho)) * (C ** max(0.0, 1.0 - rho))
    e = lam * additive + (1.0 - lam) * geometric

    # softly penalize extreme imbalance (e.g., S>>C or C>>S)
    e *= (min(S, C) ** max(0.0, eta))

    # clip + compress tails to keep degrees well-behaved for peeling
    e = _clamp01(e)
    e = _clamp01(e ** gamma_e)

    return e

def tune_parameters(prompt: str):

    w_s = {"role": 1, "experience": 1, "industry": 1, "market": 1, "offering": 1, "persona": 1}
    w_c = {"role": 1, "experience": 1.2, "industry": 1.1, "market": 1.1, "offering": 1.2, "persona": 0.9}

    return w_s, w_c


if __name__ == '__main__':
    # sims/comp should each have 6 keys (role, experience, industry, market, offering, persona)
    sims = {
        "role": 0.5, "experience": 0.5, "industry": 0.5,
        "market": 0.5, "offering": 0.5, "persona": 0.5
    }
    comps = {
        "role": 0.8, "experience": 0.5, "industry": 0.5,
        "market": 0.5, "offering": 0.5, "persona": 0.5
    }

    w_s = {"role": 1, "experience": 1, "industry": 1, "market": 1, "offering": 1, "persona": 1}
    w_c = {"role": 1, "experience": 1.2, "industry": 1.1, "market": 1.1, "offering": 1.2, "persona": 0.9}

    score = combine_edge_weight(
        sims, comps,
        w_s=w_s, w_c=w_c,
        p_s=0.0,  # geometric mean for similarity (rewards consistent agreement)
        p_c=0.5,  # slightly tolerant for complementarity (allow tradeoffs across facets)
        rho=0.5,  # balance similarity vs complementarity
        lam=0.5,  # balance additive (compensation) vs multiplicative (synergy)
        eta=0.2,  # mild penalty for lopsided S/C
        gamma_e=0.85  # gentle compression
    )
    print(score)

    # self.graph.add_edge(i, j, weight=score)
