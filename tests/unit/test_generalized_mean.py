import json
from unittest.mock import MagicMock, patch

import pytest

from services.graph.scoring.generalized_mean import (
    _clamp01,
    _norm_weights,
    _power_mean,
    combine_edge_weight,
    tune_parameters,
)
from shared.shared import DEFAULT_FEATURE_WEIGHTS, FEATURES, OPTIMIZED_FEATURE_WEIGHTS


class TestTuneParameters:

    @pytest.fixture
    def mock_openai_client(self):
        return MagicMock()

    @pytest.fixture
    def mock_chatgpt_response(self, mock_openai_client):
        mock_response = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response
        return mock_response

    def test_tune_parameters_empty_prompt_returns_defaults(self):
        """Test that empty or None prompt returns default weights"""
        w_s, w_c = tune_parameters("")
        assert w_s == DEFAULT_FEATURE_WEIGHTS
        assert w_c == OPTIMIZED_FEATURE_WEIGHTS

        w_s, w_c = tune_parameters(None)
        assert w_s == DEFAULT_FEATURE_WEIGHTS
        assert w_c == OPTIMIZED_FEATURE_WEIGHTS

        w_s, w_c = tune_parameters("   ")
        assert w_s == DEFAULT_FEATURE_WEIGHTS
        assert w_c == OPTIMIZED_FEATURE_WEIGHTS

    @patch("services.graph.scoring.generalized_mean.openai.OpenAI")
    def test_tune_parameters_hiring_maximization_prompt(
        self, mock_openai_class, mock_openai_client, mock_chatgpt_response
    ):
        """Test ChatGPT response for hiring maximization prompt"""
        mock_openai_class.return_value = mock_openai_client

        chatgpt_json_response = {
            "similarity_weights": {
                "role": 0.8,
                "experience": 0.6,
                "industry": 1.2,
                "market": 1.1,
                "offering": 0.9,
                "persona": 1.0,
            },
            "complementarity_weights": {
                "role": 1.5,
                "experience": 1.8,
                "industry": 0.7,
                "market": 0.8,
                "offering": 1.3,
                "persona": 1.1,
            },
        }

        mock_chatgpt_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(chatgpt_json_response)))
        ]

        w_s, w_c = tune_parameters("I want to maximize the hiring chance")

        assert w_s["role"] == 0.8
        assert w_s["experience"] == 0.6
        assert w_s["industry"] == 1.2
        assert w_c["role"] == 1.5
        assert w_c["experience"] == 1.8
        assert w_c["industry"] == 0.7

        mock_openai_client.chat.completions.create.assert_called_once()

    @patch("services.graph.scoring.generalized_mean.openai.OpenAI")
    def test_tune_parameters_peer_networking_prompt(
        self, mock_openai_class, mock_openai_client, mock_chatgpt_response
    ):
        """Test ChatGPT response for peer networking prompt"""
        mock_openai_class.return_value = mock_openai_client

        chatgpt_json_response = {
            "similarity_weights": {
                "role": 1.8,
                "experience": 1.6,
                "industry": 1.5,
                "market": 1.3,
                "offering": 1.2,
                "persona": 1.7,
            },
            "complementarity_weights": {
                "role": 0.3,
                "experience": 0.4,
                "industry": 0.5,
                "market": 0.6,
                "offering": 0.5,
                "persona": 0.4,
            },
        }

        mock_chatgpt_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(chatgpt_json_response)))
        ]

        w_s, w_c = tune_parameters("I want to maximize peer networking")

        assert w_s["role"] == 1.8
        assert w_s["experience"] == 1.6
        assert w_s["persona"] == 1.7
        assert w_c["role"] == 0.3
        assert w_c["experience"] == 0.4
        assert w_c["persona"] == 0.4

    @patch("services.graph.scoring.generalized_mean.openai.OpenAI")
    def test_tune_parameters_business_partnerships_prompt(
        self, mock_openai_class, mock_openai_client, mock_chatgpt_response
    ):
        """Test ChatGPT response for business partnerships prompt"""
        mock_openai_class.return_value = mock_openai_client

        chatgpt_json_response = {
            "similarity_weights": {
                "role": 0.9,
                "experience": 1.0,
                "industry": 1.6,
                "market": 0.7,
                "offering": 0.8,
                "persona": 1.1,
            },
            "complementarity_weights": {
                "role": 1.1,
                "experience": 1.2,
                "industry": 0.8,
                "market": 1.8,
                "offering": 1.9,
                "persona": 1.0,
            },
        }

        mock_chatgpt_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(chatgpt_json_response)))
        ]

        w_s, w_c = tune_parameters("I want to maximize business partnerships")

        assert w_s["industry"] == 1.6
        assert w_s["market"] == 0.7
        assert w_s["offering"] == 0.8
        assert w_c["market"] == 1.8
        assert w_c["offering"] == 1.9

    @patch("services.graph.scoring.generalized_mean.openai.OpenAI")
    def test_tune_parameters_weight_clamping(
        self, mock_openai_class, mock_openai_client, mock_chatgpt_response
    ):
        """Test that weights above 2.0 are clamped to 2.0"""
        mock_openai_class.return_value = mock_openai_client

        chatgpt_json_response = {
            "similarity_weights": {
                "role": 3.5,
                "experience": 2.8,
                "industry": 1.2,
                "market": 1.1,
                "offering": 0.9,
                "persona": 1.0,
            },
            "complementarity_weights": {
                "role": 1.5,
                "experience": 4.2,
                "industry": 0.7,
                "market": 0.8,
                "offering": 1.3,
                "persona": 1.1,
            },
        }

        mock_chatgpt_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(chatgpt_json_response)))
        ]

        w_s, w_c = tune_parameters("test prompt")

        assert w_s["role"] == 2.0
        assert w_s["experience"] == 2.0
        assert w_c["experience"] == 2.0

    @patch("services.graph.scoring.generalized_mean.openai.OpenAI")
    def test_tune_parameters_invalid_json_fallback(
        self, mock_openai_class, mock_openai_client, mock_chatgpt_response
    ):
        """Test fallback to defaults when ChatGPT returns invalid JSON"""
        mock_openai_class.return_value = mock_openai_client
        mock_chatgpt_response.choices = [
            MagicMock(message=MagicMock(content="This is not valid JSON"))
        ]

        w_s, w_c = tune_parameters("test prompt")

        assert w_s == DEFAULT_FEATURE_WEIGHTS
        assert w_c == OPTIMIZED_FEATURE_WEIGHTS

    @patch("services.graph.scoring.generalized_mean.openai.OpenAI")
    def test_tune_parameters_missing_features_fallback(
        self, mock_openai_class, mock_openai_client, mock_chatgpt_response
    ):
        """Test fallback for missing features in ChatGPT response"""
        mock_openai_class.return_value = mock_openai_client

        chatgpt_json_response = {
            "similarity_weights": {"role": 1.5, "experience": 1.2},
            "complementarity_weights": {"role": 0.8},
        }

        mock_chatgpt_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(chatgpt_json_response)))
        ]

        w_s, w_c = tune_parameters("test prompt")

        assert w_s["role"] == 1.5
        assert w_s["experience"] == 1.2
        assert w_s["industry"] == DEFAULT_FEATURE_WEIGHTS["industry"]
        assert w_c["role"] == 0.8
        assert w_c["experience"] == OPTIMIZED_FEATURE_WEIGHTS["experience"]

    @patch("services.graph.scoring.generalized_mean.openai.OpenAI")
    def test_tune_parameters_api_exception_fallback(
        self, mock_openai_class, mock_openai_client
    ):
        """Test fallback to defaults when OpenAI API raises exception"""
        mock_openai_class.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        w_s, w_c = tune_parameters("test prompt")

        assert w_s == DEFAULT_FEATURE_WEIGHTS
        assert w_c == OPTIMIZED_FEATURE_WEIGHTS


class TestHelperFunctions:

    def test_clamp01(self):
        """Test _clamp01 function"""
        assert _clamp01(-0.5) == 0.0
        assert _clamp01(0.3) == 0.3
        assert _clamp01(1.5) == 1.0
        assert _clamp01(0.0) == 0.0
        assert _clamp01(1.0) == 1.0

    def test_norm_weights_scalar(self):
        """Test _norm_weights with scalar input"""
        feats = ["role", "experience", "industry"]
        result = _norm_weights(1.0, feats)
        expected_weight = 1.0 / len(feats)

        for feat in feats:
            assert result[feat] == expected_weight

    def test_norm_weights_dict(self):
        """Test _norm_weights with dictionary input"""
        feats = ["role", "experience", "industry"]
        weights = {"role": 2.0, "experience": 1.0, "industry": 1.0}
        result = _norm_weights(weights, feats)

        total = sum(weights[f] for f in feats)
        assert result["role"] == 2.0 / total
        assert result["experience"] == 1.0 / total
        assert result["industry"] == 1.0 / total

    def test_norm_weights_zero_sum(self):
        """Test _norm_weights with weights summing to zero"""
        feats = ["role", "experience"]
        weights = {"role": 0.0, "experience": 0.0}
        result = _norm_weights(weights, feats)

        assert result["role"] == 0.5
        assert result["experience"] == 0.5

    def test_power_mean_geometric(self):
        """Test _power_mean with geometric mean (p=0)"""
        values = {"role": 0.4, "experience": 0.6}
        weights = {"role": 0.5, "experience": 0.5}
        result = _power_mean(values, weights, 0.0)

        expected = (0.4**0.5) * (0.6**0.5)
        assert abs(result - expected) < 1e-10

    def test_power_mean_arithmetic(self):
        """Test _power_mean with arithmetic mean (p=1)"""
        values = {"role": 0.4, "experience": 0.6}
        weights = {"role": 0.5, "experience": 0.5}
        result = _power_mean(values, weights, 1.0)

        expected = 0.5 * 0.4 + 0.5 * 0.6
        assert abs(result - expected) < 1e-10


class TestCombineEdgeWeight:

    def test_combine_edge_weight_basic(self):
        """Test basic functionality of combine_edge_weight"""
        sim = {f: 0.5 for f in FEATURES}
        comp = {f: 0.7 for f in FEATURES}
        w_s = {f: 1.0 for f in FEATURES}
        w_c = {f: 1.0 for f in FEATURES}

        result = combine_edge_weight(sim, comp, w_s, w_c)

        assert 0.0 < result <= 1.0

    def test_combine_edge_weight_empty_features(self):
        """Test combine_edge_weight with no matching features"""
        sim = {}
        comp = {}
        w_s = {f: 1.0 for f in FEATURES}
        w_c = {f: 1.0 for f in FEATURES}

        result = combine_edge_weight(sim, comp, w_s, w_c)

        assert result == 0.0

    def test_combine_edge_weight_extreme_values(self):
        """Test combine_edge_weight with extreme similarity/complementarity values"""
        sim_high = {f: 1.0 for f in FEATURES}
        comp_low = {f: 0.1 for f in FEATURES}
        w_s = {f: 1.0 for f in FEATURES}
        w_c = {f: 1.0 for f in FEATURES}

        result = combine_edge_weight(sim_high, comp_low, w_s, w_c)
        assert 0.0 < result <= 1.0

        sim_low = {f: 0.1 for f in FEATURES}
        comp_high = {f: 1.0 for f in FEATURES}

        result2 = combine_edge_weight(sim_low, comp_high, w_s, w_c)
        assert 0.0 < result2 <= 1.0


class TestIntegration:

    @patch("services.graph.scoring.generalized_mean.openai.OpenAI")
    def test_tune_parameters_integration_with_combine_edge_weight(
        self, mock_openai_class
    ):
        """Test integration between tune_parameters and combine_edge_weight"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "similarity_weights": {f: 1.2 for f in FEATURES},
                            "complementarity_weights": {f: 0.8 for f in FEATURES},
                        }
                    )
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        w_s, w_c = tune_parameters("maximize peer networking")

        sim = {f: 0.6 for f in FEATURES}
        comp = {f: 0.4 for f in FEATURES}

        result = combine_edge_weight(sim, comp, w_s, w_c)
        assert 0.0 < result <= 1.0

        for feature in FEATURES:
            assert w_s[feature] == 1.2
            assert w_c[feature] == 0.8


if __name__ == "__main__":
    pytest.main([__file__])
