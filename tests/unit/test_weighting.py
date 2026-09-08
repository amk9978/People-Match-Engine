import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from match_engine.services.features.feature_set import Feature, FeatureSet
from match_engine.services.scoring.calibration import calibrate_all, rank_normalize
from match_engine.services.scoring.informativeness import informativeness, measure_all
from match_engine.services.scoring.intent import (
    Intent,
    LLMIntentResolver,
    UniformIntentResolver,
)
from match_engine.services.scoring.weight_resolver import WeightResolver

FEATURE_SET = FeatureSet(
    features=(
        Feature(name="industry", column="Industry"),
        Feature(name="role", column="Role"),
    ),
    name_column="Name",
)


def symmetric(values):
    """Build a symmetric matrix whose off-diagonal entries are the given values."""
    size = 1
    while size * (size - 1) // 2 < len(values):
        size += 1
    matrix = np.zeros((size, size))
    matrix[np.triu_indices(size, k=1)] = values
    return matrix + matrix.T


FLAT = symmetric([0.5] * 10)
NARROW = symmetric(np.linspace(0.40, 0.45, 10))
SPREAD = symmetric(np.linspace(0.0, 1.0, 10))


class TestInformativeness:
    def test_a_feature_that_scores_every_pair_alike_is_useless(self):
        assert informativeness(FLAT) == 0.0

    def test_a_well_spread_feature_scores_high(self):
        assert informativeness(SPREAD) > 0.9

    def test_the_two_differ_by_an_order_of_magnitude(self):
        assert informativeness(SPREAD) > 10 * informativeness(FLAT) + 0.5

    def test_one_outlying_pair_does_not_set_the_scale(self):
        tight = symmetric([0.50, 0.51, 0.49, 0.50, 0.52, 0.48, 0.50, 0.51, 0.49, 1.0])
        assert informativeness(tight) < 0.1

    def test_a_matrix_too_small_to_have_pairs_scores_zero(self):
        assert informativeness(np.zeros((1, 1))) == 0.0

    def test_every_feature_is_measured(self):
        assert set(measure_all({"a": FLAT, "b": SPREAD})) == {"a", "b"}


class TestCalibration:
    def test_scores_become_percentiles(self):
        calibrated = rank_normalize(SPREAD, preserve_diagonal=False)
        upper = calibrated[np.triu_indices_from(calibrated, k=1)]

        assert upper.min() >= 0.0
        assert upper.max() <= 1.0
        assert upper.mean() == pytest.approx(0.5, abs=0.01)

    def test_the_pair_order_within_a_feature_is_preserved(self):
        raw = symmetric([0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4, 0.6, 0.05])
        calibrated = rank_normalize(raw, preserve_diagonal=False)

        upper = np.triu_indices_from(raw, k=1)
        assert list(np.argsort(raw[upper])) == list(np.argsort(calibrated[upper]))

    def test_the_result_stays_symmetric(self):
        calibrated = rank_normalize(SPREAD, preserve_diagonal=False)
        assert np.allclose(calibrated, calibrated.T)

    def test_two_features_on_different_scales_become_comparable(self):
        narrow = symmetric(np.linspace(0.40, 0.45, 10))
        wide = symmetric(np.linspace(0.0, 1.0, 10))

        calibrated = calibrate_all(
            {"narrow": narrow, "wide": wide}, preserve_diagonal=False
        )
        upper = np.triu_indices(calibrated["narrow"].shape[0], k=1)

        assert np.allclose(
            np.sort(calibrated["narrow"][upper]), np.sort(calibrated["wide"][upper])
        )

    def test_ties_share_a_percentile(self):
        tied = symmetric([0.5, 0.5, 0.5, 0.1, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5])
        calibrated = rank_normalize(tied, preserve_diagonal=False)
        upper = np.triu_indices_from(tied, k=1)

        percentiles = calibrated[upper][tied[upper] == 0.5]
        assert len(set(percentiles)) == 1

    def test_the_similarity_diagonal_survives(self):
        raw = SPREAD.copy()
        np.fill_diagonal(raw, 1.0)
        calibrated = rank_normalize(raw, preserve_diagonal=True)
        assert np.allclose(np.diag(calibrated), 1.0)


class TestIntent:
    def test_the_offline_default_is_uniform_and_neutral(self):
        intent = UniformIntentResolver().resolve("hire an engineer", FEATURE_SET)

        assert intent.importance == {"industry": 0.5, "role": 0.5}
        assert intent.direction == {"industry": 0.5, "role": 0.5}

    def test_an_empty_prompt_needs_no_model(self):
        client = MagicMock()
        intent = LLMIntentResolver(client=client).resolve("", FEATURE_SET)

        client.chat.completions.create.assert_not_called()
        assert intent.importance == {"industry": 0.5, "role": 0.5}

    def test_a_mixed_intent_splits_the_direction(self):
        client = self._client_returning(
            {
                "importance": {"industry": 60, "role": 40},
                "direction": {"industry": 90, "role": 10},
            }
        )

        intent = LLMIntentResolver(client=client).resolve(
            "hiring for a fintech startup", FEATURE_SET
        )

        assert intent.importance["industry"] == pytest.approx(0.6)
        assert intent.direction["industry"] == pytest.approx(0.9)
        assert intent.direction["role"] == pytest.approx(0.1)

    def test_the_same_prompt_is_only_asked_once(self):
        client = self._client_returning(
            {
                "importance": {"industry": 50, "role": 50},
                "direction": {"industry": 50, "role": 50},
            }
        )
        resolver = LLMIntentResolver(client=client)

        first = resolver.resolve("find a co-founder", FEATURE_SET)
        second = resolver.resolve("find a co-founder", FEATURE_SET)

        assert first == second
        assert client.chat.completions.create.call_count == 1

    def test_the_prompt_names_the_runtime_features(self):
        client = self._client_returning(
            {
                "importance": {"industry": 50, "role": 50},
                "direction": {"industry": 50, "role": 50},
            }
        )

        LLMIntentResolver(client=client).resolve("find a hire", FEATURE_SET)

        sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "industry" in sent
        assert "role" in sent

    def test_an_unparseable_reply_falls_back_to_uniform(self):
        client = MagicMock()
        client.chat.completions.create.return_value = self._reply("not json")

        intent = LLMIntentResolver(client=client).resolve("find a hire", FEATURE_SET)

        assert intent.importance == {"industry": 0.5, "role": 0.5}

    def test_a_feature_the_model_forgot_gets_no_importance(self):
        client = self._client_returning(
            {"importance": {"industry": 100}, "direction": {"industry": 100}}
        )

        intent = LLMIntentResolver(client=client).resolve("same industry", FEATURE_SET)

        assert intent.importance == {"industry": 1.0, "role": 0.0}

    def _client_returning(self, payload):
        client = MagicMock()
        client.chat.completions.create.return_value = self._reply(json.dumps(payload))
        return client

    def _reply(self, content):
        message = MagicMock()
        message.content = content
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response


class TestWeightResolver:
    def test_a_useless_feature_gets_almost_no_weight(self):
        w_s, _ = WeightResolver().resolve(
            {"flat": FLAT, "spread": SPREAD},
            {"flat": FLAT, "spread": SPREAD},
            Intent.uniform(["flat", "spread"]),
        )

        assert w_s["flat"] < 0.05
        assert w_s["spread"] > 0.95

    def test_with_no_intent_the_weights_are_pure_informativeness(self):
        w_s, w_c = WeightResolver().resolve(
            {"a": SPREAD, "b": FLAT},
            {"a": FLAT, "b": SPREAD},
            Intent.uniform(["a", "b"]),
        )

        assert w_s["a"] > w_s["b"]
        assert w_c["b"] > w_c["a"]

    def test_sameness_sends_weight_to_similarity(self):
        intent = Intent(importance={"a": 1.0}, direction={"a": 1.0})
        w_s, w_c = WeightResolver().resolve({"a": SPREAD}, {"a": SPREAD}, intent)

        assert w_s["a"] == 1.0
        assert w_c["a"] == 1.0

    def test_a_mixed_intent_routes_each_feature_to_one_signal(self):
        intent = Intent(
            importance={"industry": 0.5, "role": 0.5},
            direction={"industry": 1.0, "role": 0.0},
        )
        matrices = {"industry": SPREAD, "role": SPREAD}

        w_s, w_c = WeightResolver().resolve(matrices, matrices, intent)

        assert w_s["industry"] > w_s["role"]
        assert w_c["role"] > w_c["industry"]

    def test_both_vectors_sum_to_one(self):
        w_s, w_c = WeightResolver().resolve(
            {"a": SPREAD, "b": SPREAD},
            {"a": SPREAD, "b": SPREAD},
            Intent.uniform(["a", "b"]),
        )

        assert sum(w_s.values()) == pytest.approx(1.0)
        assert sum(w_c.values()) == pytest.approx(1.0)

    def test_all_zero_weights_fall_back_to_uniform(self):
        w_s, _ = WeightResolver().resolve(
            {"a": FLAT, "b": FLAT}, {"a": FLAT, "b": FLAT}, Intent.uniform(["a", "b"])
        )

        assert w_s == {"a": 0.5, "b": 0.5}


class TestMeasurementBeforeCalibration:
    def test_measuring_after_calibration_destroys_the_signal(self):
        """Named by specs/dynamic-features-and-weighting.md. Rank normalization makes
        every feature uniform, so measuring afterwards cannot tell them apart."""
        raw = {"narrow": NARROW, "spread": SPREAD}

        before = measure_all(raw)
        after = measure_all(calibrate_all(raw, preserve_diagonal=False))

        assert before["narrow"] < 0.2 < before["spread"]
        assert after["narrow"] == pytest.approx(after["spread"])

    def test_the_resolver_reads_raw_matrices_not_calibrated_ones(self):
        raw = {"narrow": NARROW, "spread": SPREAD}
        intent = Intent.uniform(["narrow", "spread"])

        from_raw, _ = WeightResolver().resolve(raw, raw, intent)
        calibrated = calibrate_all(raw, preserve_diagonal=False)
        from_calibrated, _ = WeightResolver().resolve(calibrated, calibrated, intent)

        assert from_raw["narrow"] < from_raw["spread"]
        assert from_calibrated["narrow"] == pytest.approx(from_calibrated["spread"])
