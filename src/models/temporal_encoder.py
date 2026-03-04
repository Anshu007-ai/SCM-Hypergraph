"""
Temporal Fusion Encoder (v2.0) -- Bi-LSTM + Transformer with Learned Gating

Fuses two complementary temporal modelling strategies:

1. **Bi-LSTM** captures local sequential patterns and short-range
   dependencies in supply-chain time series (e.g., lead-time sequences,
   price fluctuations over recent weeks).

2. **Transformer** with sinusoidal positional encoding captures long-range
   temporal dependencies (e.g., seasonal patterns, delayed cascade effects).

The two streams are combined via a learned gating mechanism:

    Temporal_out = Gate * BiLSTM(X) + (1 - Gate) * Transformer(X)
    Gate = sigma(W_g [BiLSTM_out || Transformer_out] + b_g)

This allows the network to adaptively decide per-token whether local
(LSTM) or global (Transformer) context is more useful.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ======================================================================
# Positional Encoding
# ======================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Generates position-dependent signals that allow the Transformer to
    reason about sequential order without recurrence.

    Args:
        d_model:     Dimensionality of the model / embedding.
        max_len:     Maximum sequence length to pre-compute encodings for.
        dropout:     Dropout probability applied after adding the encoding.
    """

    def __init__(
        self, d_model: int, max_len: int = 5000, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the positional encoding table
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        # (1, max_len, d_model) -- ready for broadcasting over batch dim
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Tensor of same shape with positional information added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ======================================================================
# Temporal Fusion Encoder
# ======================================================================

class TemporalFusionEncoder(nn.Module):
    """Bi-LSTM + Transformer temporal encoder with learned gating.

    Args:
        input_dim:               Feature dimensionality at each time step.
        hidden_dim:              Internal hidden dimensionality (also the
                                 output dimensionality).
        num_lstm_layers:         Number of stacked Bi-LSTM layers.
        num_transformer_layers:  Number of Transformer encoder layers.
        num_heads:               Number of attention heads in the Transformer.
        dropout:                 Dropout probability used throughout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_lstm_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads

        # ---- Input projection ----------------------------------------
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # ---- Bi-LSTM stream ------------------------------------------
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # bidirectional doubles this
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # ---- Transformer stream --------------------------------------
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=hidden_dim, dropout=dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        self.transformer_norm = nn.LayerNorm(hidden_dim)

        # ---- Gating mechanism ----------------------------------------
        # Gate = sigma(W_g [lstm_out || transformer_out] + b_g)
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)

        # ---- Output projection ---------------------------------------
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialise all learnable parameters."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # LSTM parameters are initialised by PyTorch defaults (uniform).
        # Explicitly reset for reproducibility.
        for name, param in self.bilstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

        self.lstm_norm.reset_parameters()
        self.transformer_norm.reset_parameters()

        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.zeros_(self.gate_linear.bias)

        self.output_norm.reset_parameters()

    def forward(self, x_sequence: torch.Tensor) -> torch.Tensor:
        """Compute temporal embeddings from an input sequence.

        Args:
            x_sequence: Tensor of shape (batch, seq_len, input_dim).
                        Each entry along the seq_len axis is the feature
                        vector for one time step.

        Returns:
            temporal_embeddings: Tensor of shape (batch, seq_len, hidden_dim)
                                 containing the fused temporal representations.
        """
        # ---- Input projection ----------------------------------------
        x = self.input_proj(x_sequence)  # (B, T, hidden_dim)

        # ---- Bi-LSTM stream ------------------------------------------
        lstm_out, _ = self.bilstm(x)     # (B, T, hidden_dim)
        lstm_out = self.lstm_norm(lstm_out)

        # ---- Transformer stream --------------------------------------
        trans_in = self.pos_encoder(x)    # (B, T, hidden_dim)
        trans_out = self.transformer_encoder(trans_in)  # (B, T, hidden_dim)
        trans_out = self.transformer_norm(trans_out)

        # ---- Gating --------------------------------------------------
        # Concatenate the two streams along the feature dimension
        combined = torch.cat([lstm_out, trans_out], dim=-1)  # (B, T, 2*H)
        gate = torch.sigmoid(self.gate_linear(combined))     # (B, T, H)

        # Fused output
        fused = gate * lstm_out + (1.0 - gate) * trans_out   # (B, T, H)

        # ---- Output normalisation + dropout --------------------------
        temporal_embeddings = self.output_norm(fused)
        temporal_embeddings = self.dropout_layer(temporal_embeddings)

        return temporal_embeddings


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TemporalFusionEncoder -- Module Info & Smoke Test")
    print("=" * 60)

    input_dim = 64
    hidden_dim = 128
    batch_size = 4
    seq_len = 24  # e.g. 24 weekly snapshots

    model = TemporalFusionEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_lstm_layers=2,
        num_transformer_layers=2,
        num_heads=8,
        dropout=0.1,
    )
    print(f"\nModel:\n{model}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Synthetic sequence
    x_seq = torch.randn(batch_size, seq_len, input_dim)
    out = model(x_seq)

    print(f"\nInput shape:  {x_seq.shape}")
    print(f"Output shape: {out.shape}  (expected [{batch_size}, {seq_len}, {hidden_dim}])")

    # Quick gradient check
    loss = out.sum()
    loss.backward()
    grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"Gradient flow OK: {grad_ok}")

    # Positional encoding sanity
    pe_mod = SinusoidalPositionalEncoding(d_model=hidden_dim)
    dummy = torch.zeros(1, 50, hidden_dim)
    pe_out = pe_mod(dummy)
    print(f"\nPositional encoding output shape: {pe_out.shape}")

    print("\nSmoke test passed.")
