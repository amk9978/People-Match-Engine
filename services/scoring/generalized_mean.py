import json
import math
import os
from textwrap import dedent
from typing import Dict, Tuple

import openai

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


def tune_parameters(prompt: str = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Use ChatGPT to tune similarity and complementarity weights based on user intent.
    Returns (w_s, w_c) - similarity and complementarity weights for each feature.
    Falls back to default weights if ChatGPT fails.
    """
    # Default weights (fallback values)
    default_w_s = {"role": 1.0, "experience": 1.0, "industry": 1.0, "market": 1.0, "offering": 1.0, "persona": 1.0}
    default_w_c = {"role": 1.0, "experience": 1.2, "industry": 1.1, "market": 1.1, "offering": 1.2, "persona": 0.9}

    if not prompt or not prompt.strip():
        return default_w_s, default_w_c

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        chatgpt_prompt = dedent(f"""
You are an expert in professional network analysis and team matching algorithms.

USER INTENT: "{prompt}"

Based on this user intent, provide optimal weights for similarity and complementarity across 6 professional features.

FEATURES EXPLANATION:
- role: Job titles, responsibilities, professional functions
- experience: Career level, years of experience, seniority  
- industry: Business sector, domain expertise
- market: Target audience, customer segments, market focus
- offering: Products/services, value propositions
- persona: Professional personality, working style, leadership traits

WEIGHT INTERPRETATION:
- SIMILARITY weights: Higher = prefer people with similar traits (team cohesion, cultural fit)
- COMPLEMENTARITY weights: Higher = prefer people with different traits (skill diversity, covering gaps)

USER INTENT EXAMPLES:
- "I want to hire for my startup" → Higher complementarity for roles/experience (diverse skills), higher similarity for industry/market (same domain focus)
- "I need peer networking" → Higher similarity across all features (find similar professionals)
- "I want business partnerships" → Higher complementarity for market/offering (different customer bases), similarity for industry (same domain)

CRITICAL: You must respond with EXACTLY this JSON format:
{{
    "similarity_weights": {{
        "role": 0.XX,
        "experience": 0.XX, 
        "industry": 0.XX,
        "market": 0.XX,
        "offering": 0.XX,
        "persona": 0.XX
    }},
    "complementarity_weights": {{
        "role": 0.XX,
        "experience": 0.XX,
        "industry": 0.XX, 
        "market": 0.XX,
        "offering": 0.XX,
        "persona": 0.XX
    }}
}}

REQUIREMENTS:
- Each weight between 0.5 and 2.0 (0.5 = de-emphasize, 1.0 = neutral, 2.0 = emphasize strongly)
- Within each weight set, values should reflect relative importance for the user's intent
- NO OTHER TEXT - ONLY the JSON object
""")

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": chatgpt_prompt}],
            temperature=0.3,
            max_tokens=500
        )

        result_text = response.choices[0].message.content.strip()

        try:
            result_json = json.loads(result_text)

            similarity_weights = result_json.get("similarity_weights", {})
            complementarity_weights = result_json.get("complementarity_weights", {})

            w_s = {}
            w_c = {}

            for feature in FEATURES:
                # Similarity weights
                if feature in similarity_weights:
                    weight = float(similarity_weights[feature])
                    w_s[feature] = min(2.0, weight)
                else:
                    w_s[feature] = default_w_s[feature]

                # Complementarity weights  
                if feature in complementarity_weights:
                    weight = float(complementarity_weights[feature])
                    w_c[feature] = min(2.0, weight)
                else:
                    w_c[feature] = default_w_c[feature]

            return w_s, w_c

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as parse_error:
            print(f"⚠️ Failed to parse ChatGPT response: {parse_error}")
            print(f"   Response was: {result_text[:200]}...")
            return default_w_s, default_w_c

    except Exception as e:
        print(f"⚠️ ChatGPT weight tuning failed: {e}")
        print("   Using default weights")
        return default_w_s, default_w_c


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
