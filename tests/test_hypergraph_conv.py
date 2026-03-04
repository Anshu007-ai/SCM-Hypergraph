"""
Test suite for HT-HGNN v2.0 Model Components.

Tests the PyTorch neural network layers used in the hypergraph risk model:
- SpectralHypergraphConv  (spectral convolution on hypergraphs)
- TemporalFusionEncoder   (Bi-LSTM + Transformer with learned gating)
- HeterogeneousRelationFusion (multi-head relation-aware attention)
- CascadeRiskHead         (multi-step cascade risk propagation)
- DynamicHyperedgeWeightLearner

All tests use small synthetic tensors so no GPU or real data is required.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Graceful imports -- skip entire module if torch / model deps missing
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from src.models.hypergraph_conv import SpectralHypergraphConv
except ImportError:
    SpectralHypergraphConv = None

try:
    from src.models.temporal_encoder import TemporalFusionEncoder
except ImportError:
    TemporalFusionEncoder = None

try:
    from src.models.relation_fusion import HeterogeneousRelationFusion
except ImportError:
    HeterogeneousRelationFusion = None

try:
    from src.models.cascade_model import CascadeRiskHead, DynamicHyperedgeWeightLearner
except ImportError:
    CascadeRiskHead = None
    DynamicHyperedgeWeightLearner = None

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Fixtures -- small synthetic data shared across tests
# ---------------------------------------------------------------------------

@pytest.fixture
def small_hypergraph_data():
    """Create a small synthetic hypergraph compatible with all model layers.

    Returns a dict with node features, an incidence matrix H (N x M),
    and optional hyperedge weights.
    """
    torch.manual_seed(42)
    N = 10   # nodes
    M = 4    # hyperedges
    in_ch = 16

    x = torch.randn(N, in_ch)

    # Build a random incidence matrix where each hyperedge has 2-4 members
    H = torch.zeros(N, M)
    for e in range(M):
        size = torch.randint(2, 5, (1,)).item()
        members = torch.randperm(N)[:size]
        H[members, e] = 1.0

    W = torch.rand(M)

    return {"x": x, "H": H, "W": W, "N": N, "M": M, "in_ch": in_ch}


@pytest.fixture
def temporal_sequence():
    """Create a small batch of temporal sequences for the TemporalFusionEncoder."""
    torch.manual_seed(0)
    batch = 2
    seq_len = 8
    input_dim = 16
    return torch.randn(batch, seq_len, input_dim)


@pytest.fixture
def relation_graph_data():
    """Create a small heterogeneous graph for the RelationFusion layer."""
    torch.manual_seed(1)
    N = 10
    hidden_dim = 16
    num_edges = 20
    num_rel_types = 5

    node_emb = torch.randn(N, hidden_dim)
    edge_index = torch.stack([
        torch.randint(0, N, (num_edges,)),
        torch.randint(0, N, (num_edges,)),
    ])
    edge_types = torch.randint(0, num_rel_types, (num_edges,))

    return {
        "node_emb": node_emb,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "N": N,
        "hidden_dim": hidden_dim,
        "num_rel_types": num_rel_types,
    }


# ---------------------------------------------------------------------------
# Tests -- SpectralHypergraphConv
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SpectralHypergraphConv is None,
                    reason="SpectralHypergraphConv not importable")
class TestSpectralHypergraphConv:
    """Tests for the spectral hypergraph convolution layer."""

    def test_spectral_hypergraph_conv_forward(self, small_hypergraph_data):
        """Forward pass should run without error and return three tensors."""
        d = small_hypergraph_data
        out_ch = 32
        conv = SpectralHypergraphConv(d["in_ch"], out_ch, use_attention=True)
        conv.eval()

        node_out, he_out, attn = conv(d["x"], d["H"], d["W"])

        assert node_out is not None
        assert he_out is not None
        assert attn is not None

    def test_spectral_conv_output_shape(self, small_hypergraph_data):
        """Output shapes must match (N, out_ch) and (M, out_ch)."""
        d = small_hypergraph_data
        out_ch = 32
        conv = SpectralHypergraphConv(d["in_ch"], out_ch, use_attention=True)
        conv.eval()

        node_out, he_out, attn = conv(d["x"], d["H"], d["W"])

        assert node_out.shape == (d["N"], out_ch), (
            f"Expected node_out shape ({d['N']}, {out_ch}), got {node_out.shape}"
        )
        assert he_out.shape == (d["M"], out_ch), (
            f"Expected he_out shape ({d['M']}, {out_ch}), got {he_out.shape}"
        )
        assert attn.shape == (d["M"],), (
            f"Expected attn shape ({d['M']},), got {attn.shape}"
        )

    def test_attention_weights_sum(self, small_hypergraph_data):
        """When use_attention=True, attention weights should sum to ~1 (softmax)."""
        d = small_hypergraph_data
        conv = SpectralHypergraphConv(d["in_ch"], 32, use_attention=True)
        conv.eval()

        _, _, attn = conv(d["x"], d["H"], d["W"])
        assert abs(attn.sum().item() - 1.0) < 1e-4, (
            f"Attention sum should be ~1.0, got {attn.sum().item()}"
        )

    def test_residual_connection(self, small_hypergraph_data):
        """When in_ch == out_ch a residual path should exist (output differs from zero-input)."""
        d = small_hypergraph_data
        in_ch = d["in_ch"]
        conv = SpectralHypergraphConv(in_ch, in_ch, use_attention=False)
        conv.eval()

        node_out, _, _ = conv(d["x"], d["H"])
        # With residual, the output should not be all zeros even if theta is zero-init
        assert node_out.abs().sum().item() > 0

    def test_layer_norm(self, small_hypergraph_data):
        """LayerNorm should keep output values in a reasonable range."""
        d = small_hypergraph_data
        conv = SpectralHypergraphConv(d["in_ch"], 32, use_attention=True)
        conv.eval()

        node_out, _, _ = conv(d["x"], d["H"], d["W"])
        # After LayerNorm + ELU, values should not explode
        assert node_out.abs().max().item() < 100, (
            "Output values after LayerNorm should stay bounded"
        )


# ---------------------------------------------------------------------------
# Tests -- TemporalFusionEncoder
# ---------------------------------------------------------------------------

@pytest.mark.skipif(TemporalFusionEncoder is None,
                    reason="TemporalFusionEncoder not importable")
class TestTemporalFusionEncoder:
    """Tests for the Bi-LSTM + Transformer temporal encoder."""

    def test_temporal_fusion_encoder_forward(self, temporal_sequence):
        """Forward pass should return a tensor of the correct dtype."""
        hidden_dim = 16
        input_dim = temporal_sequence.shape[-1]
        encoder = TemporalFusionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=1,
            num_transformer_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        encoder.eval()

        out = encoder(temporal_sequence)
        assert out.dtype == torch.float32

    def test_temporal_encoder_output_shape(self, temporal_sequence):
        """Output should be (batch, seq_len, hidden_dim)."""
        batch, seq_len, input_dim = temporal_sequence.shape
        hidden_dim = 16
        encoder = TemporalFusionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=1,
            num_transformer_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        encoder.eval()

        out = encoder(temporal_sequence)
        assert out.shape == (batch, seq_len, hidden_dim), (
            f"Expected ({batch}, {seq_len}, {hidden_dim}), got {out.shape}"
        )

    def test_temporal_encoder_gate_range(self, temporal_sequence):
        """The internal gating values should fall in [0, 1] (sigmoid output)."""
        input_dim = temporal_sequence.shape[-1]
        hidden_dim = 16
        encoder = TemporalFusionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=1,
            num_transformer_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        encoder.eval()

        # Manually run sub-components to inspect the gate
        x = encoder.input_proj(temporal_sequence)
        lstm_out, _ = encoder.bilstm(x)
        lstm_out = encoder.lstm_norm(lstm_out)
        trans_in = encoder.pos_encoder(x)
        trans_out = encoder.transformer_encoder(trans_in)
        trans_out = encoder.transformer_norm(trans_out)
        combined = torch.cat([lstm_out, trans_out], dim=-1)
        gate = torch.sigmoid(encoder.gate_linear(combined))

        assert gate.min().item() >= 0.0 - 1e-6
        assert gate.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Tests -- HeterogeneousRelationFusion
# ---------------------------------------------------------------------------

@pytest.mark.skipif(HeterogeneousRelationFusion is None,
                    reason="HeterogeneousRelationFusion not importable")
class TestRelationFusion:
    """Tests for the heterogeneous relation-aware fusion layer."""

    def test_relation_fusion_forward(self, relation_graph_data):
        """Forward pass should produce fused embeddings of correct shape."""
        d = relation_graph_data
        model = HeterogeneousRelationFusion(
            hidden_dim=d["hidden_dim"],
            num_relation_types=d["num_rel_types"],
            num_heads=4,
        )
        model.eval()

        fused = model(d["node_emb"], d["edge_index"], d["edge_types"])
        assert fused.shape == (d["N"], d["hidden_dim"]), (
            f"Expected shape ({d['N']}, {d['hidden_dim']}), got {fused.shape}"
        )


# ---------------------------------------------------------------------------
# Tests -- CascadeRiskHead & DynamicHyperedgeWeightLearner
# ---------------------------------------------------------------------------

@pytest.mark.skipif(CascadeRiskHead is None,
                    reason="CascadeRiskHead not importable")
class TestCascadeRiskHead:
    """Tests for the differentiable cascade risk propagation head."""

    def test_cascade_risk_head_forward(self, small_hypergraph_data):
        """Forward pass should return per-node risk scores in [0, 1]."""
        d = small_hypergraph_data
        head = CascadeRiskHead(
            hidden_dim=d["in_ch"],
            num_cascade_steps=3,
            cascade_threshold=0.5,
        )
        head.eval()

        scores = head(d["x"], d["H"])
        assert scores.shape == (d["N"],), (
            f"Expected shape ({d['N']},), got {scores.shape}"
        )
        assert scores.min().item() >= 0.0 - 1e-6
        assert scores.max().item() <= 1.0 + 1e-6

    def test_dynamic_weight_learner(self, small_hypergraph_data):
        """DynamicHyperedgeWeightLearner should produce weights in [0, 1]."""
        d = small_hypergraph_data
        learner = DynamicHyperedgeWeightLearner(
            hidden_dim=d["in_ch"],
            time_embed_dim=8,
            mlp_hidden=16,
        )
        learner.eval()

        weights = learner(d["x"], d["H"], time_step=torch.tensor([3.0]))
        assert weights.shape == (d["M"],), (
            f"Expected shape ({d['M']},), got {weights.shape}"
        )
        assert weights.min().item() >= 0.0 - 1e-6, "Weights should be >= 0"
        assert weights.max().item() <= 1.0 + 1e-6, "Weights should be <= 1"
