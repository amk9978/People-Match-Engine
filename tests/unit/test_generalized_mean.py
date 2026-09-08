import pytest

from services.graph.scoring.generalized_mean import (
    _clamp01,
    _norm_weights,
    _power_mean,
    combine_edge_weight,
)
from services.scoring.profile import ScoringProfile

FEATURES = ("role", "experience", "industry")


def flat(value):
    return {feature: value for feature in FEATURES}


class TestClamp:
    @pytest.mark.parametrize(
        "value,expected", [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 1.0)]
    )
    def test_values_land_in_the_unit_interval(self, value, expected):
        assert _clamp01(value) == expected


class TestNormWeights:
    def test_weights_sum_to_one(self):
        normalized = _norm_weights({"a": 2.0, "b": 6.0}, ["a", "b"])
        assert normalized == {"a": 0.25, "b": 0.75}

    def test_all_zero_weights_become_uniform(self):
        normalized = _norm_weights({"a": 0.0, "b": 0.0}, ["a", "b"])
        assert normalized == {"a": 0.5, "b": 0.5}

    def test_negative_weights_are_treated_as_zero(self):
        normalized = _norm_weights({"a": -3.0, "b": 1.0}, ["a", "b"])
        assert normalized == {"a": 0.0, "b": 1.0}

    def test_an_absent_feature_weighs_nothing(self):
        normalized = _norm_weights({"a": 1.0}, ["a", "b"])
        assert normalized == {"a": 1.0, "b": 0.0}


class TestPowerMean:
    def test_p_of_one_is_the_arithmetic_mean(self):
        result = _power_mean({"a": 0.4, "b": 0.6}, {"a": 0.5, "b": 0.5}, 1.0)
        assert result == pytest.approx(0.5)

    def test_p_of_zero_is_the_geometric_mean(self):
        result = _power_mean({"a": 0.25, "b": 1.0}, {"a": 0.5, "b": 0.5}, 0.0)
        assert result == pytest.approx(0.5)

    def test_no_shared_feature_scores_zero(self):
        assert _power_mean({"a": 0.5}, {"b": 1.0}, 1.0) == 0.0

    def test_a_zero_value_does_not_blow_up_the_geometric_mean(self):
        result = _power_mean({"a": 0.0, "b": 1.0}, {"a": 0.5, "b": 0.5}, 0.0)
        assert 0.0 <= result < 0.01


class TestCombineEdgeWeight:
    def test_an_ordinary_pair_scores_inside_the_unit_interval(self):
        result = combine_edge_weight(flat(0.5), flat(0.7), flat(1.0), flat(1.0))
        assert 0.0 < result <= 1.0

    def test_no_shared_feature_scores_zero(self):
        assert combine_edge_weight({}, {}, flat(1.0), flat(1.0)) == 0.0

    def test_only_features_present_in_both_signals_count(self):
        both = combine_edge_weight(
            {"role": 0.9, "market": 0.1}, {"role": 0.9}, flat(1.0), flat(1.0)
        )
        role_only = combine_edge_weight(
            {"role": 0.9}, {"role": 0.9}, flat(1.0), flat(1.0)
        )
        assert both == role_only

    def test_a_lopsided_pair_scores_below_a_balanced_one(self):
        balanced = combine_edge_weight(flat(0.6), flat(0.6), flat(1.0), flat(1.0))
        lopsided = combine_edge_weight(flat(1.0), flat(0.2), flat(1.0), flat(1.0))
        assert lopsided < balanced

    def test_raising_both_signals_raises_the_edge(self):
        weak = combine_edge_weight(flat(0.3), flat(0.3), flat(1.0), flat(1.0))
        strong = combine_edge_weight(flat(0.8), flat(0.8), flat(1.0), flat(1.0))
        assert strong > weak

    def test_weights_steer_which_feature_decides(self):
        sim = {"role": 1.0, "market": 0.0}
        comp = {"role": 1.0, "market": 1.0}

        role_led = combine_edge_weight(
            sim, comp, {"role": 1.0, "market": 0.0}, flat(1.0)
        )
        market_led = combine_edge_weight(
            sim, comp, {"role": 0.0, "market": 1.0}, flat(1.0)
        )
        assert role_led > market_led

    def test_a_profile_that_ignores_complementarity_follows_similarity(self):
        similarity_only = ScoringProfile(rho=1.0, lam=1.0, eta=0.0)

        strong = combine_edge_weight(
            flat(0.9), flat(0.1), flat(1.0), flat(1.0), similarity_only
        )
        weak = combine_edge_weight(
            flat(0.1), flat(0.9), flat(1.0), flat(1.0), similarity_only
        )
        assert strong > weak

    def test_the_default_profile_is_used_when_none_is_given(self):
        assert combine_edge_weight(
            flat(0.5), flat(0.5), flat(1.0), flat(1.0)
        ) == combine_edge_weight(
            flat(0.5), flat(0.5), flat(1.0), flat(1.0), ScoringProfile()
        )
