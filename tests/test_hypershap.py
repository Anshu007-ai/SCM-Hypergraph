"""
Test suite for HT-HGNN v2.0 Explainability Components.

Tests the SHAP-based and gradient-based explainability modules:
- HyperSHAP:                     Shapley-value explanations over hyperedge coalitions
- HyperedgeImportanceAnalyzer:    gradient / removal / attention importance
- FeatureAttributionAnalyzer:     integrated-gradient feature attribution

All tests use a tiny mock model and synthetic tensors so no GPU or real
data is required.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from src.explainability.hypershap import HyperSHAP, NodeExplanation
except ImportError:
    HyperSHAP = None
    NodeExplanation = None

try:
    from src.explainability.hyperedge_importance import HyperedgeImportanceAnalyzer
except ImportError:
    HyperedgeImportanceAnalyzer = None

try:
    from src.explainability.feature_attribution import FeatureAttributionAnalyzer
except ImportError:
    FeatureAttributionAnalyzer = None

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Tiny mock model -- mimics the HT-HGNN forward interface
# ---------------------------------------------------------------------------

class _MockHTHGNN(nn.Module):
    """Minimal stand-in for the full HT-HGNN model.

    Accepts the same arguments as the real model's forward() and returns
    a dict with criticality, price_pred, and change_pred keys.
    """

    def __init__(self, in_features: int, num_nodes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.num_nodes = num_nodes

    def forward(self, node_features, incidence_matrix, node_types,
                edge_index, edge_types, timestamps):
        # Simple linear projection per node -> scalar per node
        out = self.linear(node_features).squeeze(-1)  # (N,)
        return {
            "criticality": out,
            "price_pred": out * 0.5,
            "change_pred": out * 0.1,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model_and_data():
    """Return a tiny mock model plus matching synthetic data tensors."""
    torch.manual_seed(0)
    N = 6       # nodes
    M = 3       # hyperedges
    in_ch = 8

    model = _MockHTHGNN(in_features=in_ch, num_nodes=N)

    # Incidence matrix (M x N) for HyperSHAP (hyperedge-major)
    H_he_major = torch.zeros(M, N)
    H_he_major[0, 0] = 1; H_he_major[0, 1] = 1; H_he_major[0, 2] = 1
    H_he_major[1, 2] = 1; H_he_major[1, 3] = 1; H_he_major[1, 4] = 1
    H_he_major[2, 4] = 1; H_he_major[2, 5] = 1

    # Incidence matrix (N x M) for CascadeRiskHead / FeatureAttribution (node-major)
    H_node_major = H_he_major.t()

    node_features = torch.randn(N, in_ch)
    node_types = ["supplier"] * N
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_types = ["supplies"] * 4
    timestamps = torch.arange(N, dtype=torch.float32)
    feature_names = [f"feat_{i}" for i in range(in_ch)]

    return {
        "model": model,
        "node_features": node_features,
        "H_he_major": H_he_major,
        "H_node_major": H_node_major,
        "node_types": node_types,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "timestamps": timestamps,
        "feature_names": feature_names,
        "N": N,
        "M": M,
        "in_ch": in_ch,
    }


# ---------------------------------------------------------------------------
# Tests -- HyperSHAP
# ---------------------------------------------------------------------------

@pytest.mark.skipif(HyperSHAP is None, reason="HyperSHAP not importable")
class TestHyperSHAP:
    """Tests for the HyperSHAP explainability module."""

    def test_hypershap_init(self, mock_model_and_data):
        """HyperSHAP should initialise and store references correctly."""
        d = mock_model_and_data
        explainer = HyperSHAP(
            model=d["model"],
            incidence_matrix=d["H_he_major"],
            num_samples=5,
            feature_names=d["feature_names"],
        )
        assert explainer.num_hyperedges == d["M"]
        assert explainer.num_nodes == d["N"]
        assert explainer.num_samples == 5

    def test_node_explanation_structure(self, mock_model_and_data):
        """explain_node should return a dict with all required keys."""
        d = mock_model_and_data
        explainer = HyperSHAP(
            model=d["model"],
            incidence_matrix=d["H_he_major"],
            num_samples=3,
            feature_names=d["feature_names"],
        )
        explanation = explainer.explain_node(
            node_id=0,
            node_features=d["node_features"],
            node_types=d["node_types"],
            edge_index=d["edge_index"],
            edge_types=d["edge_types"],
            timestamps=d["timestamps"],
            prediction_type="criticality",
        )
        expected_keys = {
            "node_id",
            "prediction_type",
            "node_attribution",
            "hyperedge_attributions",
            "feature_attributions",
            "prediction_value",
            "base_value",
            "recommendations",
        }
        assert expected_keys.issubset(set(explanation.keys())), (
            f"Missing keys: {expected_keys - set(explanation.keys())}"
        )

    def test_hyperedge_attribution_values(self, mock_model_and_data):
        """Each hyperedge should receive a numeric Shapley value."""
        d = mock_model_and_data
        explainer = HyperSHAP(
            model=d["model"],
            incidence_matrix=d["H_he_major"],
            num_samples=3,
        )
        explanation = explainer.explain_node(
            node_id=0,
            node_features=d["node_features"],
            node_types=d["node_types"],
            edge_index=d["edge_index"],
            edge_types=d["edge_types"],
            timestamps=d["timestamps"],
        )
        he_attr = explanation["hyperedge_attributions"]
        assert len(he_attr) == d["M"], (
            f"Should have {d['M']} hyperedge attributions, got {len(he_attr)}"
        )
        for idx, val in he_attr.items():
            assert isinstance(val, float), (
                f"Attribution for hyperedge {idx} should be float, got {type(val)}"
            )

    def test_feature_attribution_values(self, mock_model_and_data):
        """Feature attributions should be a dict keyed by feature name."""
        d = mock_model_and_data
        explainer = HyperSHAP(
            model=d["model"],
            incidence_matrix=d["H_he_major"],
            num_samples=3,
            feature_names=d["feature_names"],
        )
        explanation = explainer.explain_node(
            node_id=0,
            node_features=d["node_features"],
            node_types=d["node_types"],
            edge_index=d["edge_index"],
            edge_types=d["edge_types"],
            timestamps=d["timestamps"],
        )
        feat_attr = explanation["feature_attributions"]
        assert len(feat_attr) == d["in_ch"]
        for name in d["feature_names"]:
            assert name in feat_attr

    def test_explanation_batch(self, mock_model_and_data):
        """explain_batch should return one explanation dict per requested node."""
        d = mock_model_and_data
        explainer = HyperSHAP(
            model=d["model"],
            incidence_matrix=d["H_he_major"],
            num_samples=2,
        )
        batch = explainer.explain_batch(
            node_ids=[0, 1],
            node_features=d["node_features"],
            node_types=d["node_types"],
            edge_index=d["edge_index"],
            edge_types=d["edge_types"],
            timestamps=d["timestamps"],
        )
        assert len(batch) == 2
        assert batch[0]["node_id"] == 0
        assert batch[1]["node_id"] == 1


# ---------------------------------------------------------------------------
# Tests -- HyperedgeImportanceAnalyzer
# ---------------------------------------------------------------------------

@pytest.mark.skipif(HyperedgeImportanceAnalyzer is None,
                    reason="HyperedgeImportanceAnalyzer not importable")
class TestHyperedgeImportance:
    """Tests for the gradient / removal based hyperedge importance."""

    def _build_analyzer(self, mock_model_and_data):
        """Helper: build a HyperedgeImportanceAnalyzer from mock fixtures."""
        d = mock_model_and_data

        # HyperedgeImportanceAnalyzer expects a hypergraph with
        # get_incidence_matrix() -> (np_array, he_ids, node_ids).
        # We create a simple stub.
        class _StubHypergraph:
            def get_incidence_matrix(self_inner):
                H_np = d["H_he_major"].numpy()
                he_ids = [f"HE_{i}" for i in range(d["M"])]
                node_ids = [f"N_{i}" for i in range(d["N"])]
                return H_np, he_ids, node_ids

        analyzer = HyperedgeImportanceAnalyzer(
            model=d["model"],
            hypergraph=_StubHypergraph(),
        )
        return analyzer

    def test_hyperedge_importance_gradient(self, mock_model_and_data):
        """Gradient-based importance should produce scores for each hyperedge."""
        d = mock_model_and_data
        analyzer = self._build_analyzer(d)

        scores = analyzer.compute_importance(
            node_features=d["node_features"],
            node_types=d["node_types"],
            edge_index=d["edge_index"],
            edge_types=d["edge_types"],
            timestamps=d["timestamps"],
            method="gradient",
            prediction_type="criticality",
        )
        assert len(scores) == d["M"]
        for hid, val in scores.items():
            assert 0.0 <= val <= 1.0 + 1e-6

    def test_hyperedge_importance_removal(self, mock_model_and_data):
        """Removal-based importance should produce scores for each hyperedge."""
        d = mock_model_and_data
        analyzer = self._build_analyzer(d)

        scores = analyzer.compute_importance(
            node_features=d["node_features"],
            node_types=d["node_types"],
            edge_index=d["edge_index"],
            edge_types=d["edge_types"],
            timestamps=d["timestamps"],
            method="removal",
            prediction_type="criticality",
        )
        assert len(scores) == d["M"]
        for hid, val in scores.items():
            assert 0.0 <= val <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Tests -- FeatureAttributionAnalyzer
# ---------------------------------------------------------------------------

@pytest.mark.skipif(FeatureAttributionAnalyzer is None,
                    reason="FeatureAttributionAnalyzer not importable")
class TestFeatureAttribution:
    """Tests for integrated-gradient feature attribution."""

    def test_feature_attribution_top_features(self, mock_model_and_data):
        """get_top_features should return k (name, score) pairs sorted by |score|."""
        d = mock_model_and_data
        analyzer = FeatureAttributionAnalyzer(
            model=d["model"],
            feature_names=d["feature_names"],
            n_steps=5,
        )

        attr = analyzer.compute_attributions(
            node_features=d["node_features"],
            node_id=0,
            incidence_matrix=d["H_node_major"],
            node_types=d["node_types"],
            edge_index=d["edge_index"],
            edge_types=d["edge_types"],
            timestamps=d["timestamps"],
            prediction_type="criticality",
        )

        top = analyzer.get_top_features(node_id=0, attributions=attr, k=3)
        assert len(top) == 3
        # Sorted by absolute value descending
        abs_scores = [abs(s) for _, s in top]
        assert abs_scores == sorted(abs_scores, reverse=True)
