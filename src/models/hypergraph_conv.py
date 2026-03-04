"""
Spectral Hypergraph Convolution (v2.0)

Implements the spectral formulation from Zhou et al. (2006):
    X^{(k+1)} = sigma(D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X^{(k)} Theta^{(k)})

Where:
    D_v = diagonal vertex degree matrix (N x N)
    D_e = diagonal hyperedge degree matrix (M x M)
    H   = incidence matrix (N x M), H[v,e]=1 iff vertex v in hyperedge e
    W   = diagonal hyperedge weight matrix (M x M)
    Theta^{(k)} = learnable parameter matrix for layer k

Properties:
    - When all hyperedges have size 2, this reduces to standard spectral GCN
      (D_e^{-1} collapses to 1/2, and H W D_e^{-1} H^T becomes the
      normalised adjacency matrix up to a constant factor).
    - Supports optional graph attention over hyperedge weights.
    - Includes residual connections and LayerNorm for stable training.

Supply-chain context:
    Hyperedges model multi-party relationships such as a supplier providing
    components to multiple manufacturers simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SpectralHypergraphConv(nn.Module):
    """Spectral hypergraph convolution layer with optional attention.

    Implements one layer of the spectral hypergraph convolution operator
    described by Zhou et al. (2006).  Optionally learns attention-based
    hyperedge weights that modulate the information propagated through
    each hyperedge.

    Args:
        in_channels:   Dimensionality of input node features.
        out_channels:  Dimensionality of output node features.
        use_attention: If True, learn attention weights over hyperedges.
        dropout:       Dropout probability applied after activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        # Learnable parameter matrix Theta^{(k)}
        self.theta = nn.Linear(in_channels, out_channels, bias=False)

        # Residual projection when dimensions differ
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.residual_proj = None

        # Layer normalisation applied after the convolution
        self.layer_norm = nn.LayerNorm(out_channels)

        # Attention mechanism over hyperedge weights
        if use_attention:
            # Attention MLP: takes aggregated hyperedge features and produces
            # a scalar attention logit per hyperedge.
            self.attn_fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(in_channels // 2, 1),
            )

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialise all learnable parameters."""
        nn.init.xavier_uniform_(self.theta.weight)
        if self.residual_proj is not None:
            nn.init.xavier_uniform_(self.residual_proj.weight)
        self.layer_norm.reset_parameters()
        if self.use_attention:
            for module in self.attn_fc:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_degree_inv_sqrt(degree: torch.Tensor) -> torch.Tensor:
        """Compute D^{-1/2} in a numerically safe manner.

        Entries with degree zero are left as zero to avoid division by zero.
        """
        inv_sqrt = torch.zeros_like(degree)
        nonzero = degree > 0
        inv_sqrt[nonzero] = degree[nonzero].pow(-0.5)
        return inv_sqrt

    @staticmethod
    def _safe_degree_inv(degree: torch.Tensor) -> torch.Tensor:
        """Compute D^{-1} in a numerically safe manner."""
        inv = torch.zeros_like(degree)
        nonzero = degree > 0
        inv[nonzero] = 1.0 / degree[nonzero]
        return inv

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        H: torch.Tensor,
        W: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one spectral hypergraph convolution step.

        Args:
            x: Node feature matrix of shape (N, in_channels).
            H: Incidence matrix of shape (N, M) where H[v, e] = 1 iff
               vertex v belongs to hyperedge e.
            W: Optional hyperedge weight vector of shape (M,).  If None
               all hyperedges are assumed to have unit weight.

        Returns:
            node_out:          Updated node features (N, out_channels).
            hyperedge_out:     Intermediate hyperedge representations
                               (M, out_channels).
            attention_weights: Attention scores per hyperedge (M,).  If
                               ``use_attention`` is False these are simply
                               the (normalised) hyperedge weights.
        """
        N, M = H.shape  # N = num vertices, M = num hyperedges

        # ---- Degree matrices -----------------------------------------
        # D_v: vertex degree (N,) -- number of hyperedges each vertex is in
        d_v = H.sum(dim=1)  # (N,)
        # D_e: hyperedge degree (M,) -- number of vertices in each hyperedge
        d_e = H.sum(dim=0)  # (M,)

        d_v_inv_sqrt = self._safe_degree_inv_sqrt(d_v)  # (N,)
        d_e_inv = self._safe_degree_inv(d_e)             # (M,)

        # ---- Hyperedge weights W -------------------------------------
        if W is None:
            W_diag = torch.ones(M, device=x.device, dtype=x.dtype)
        else:
            W_diag = W  # (M,)

        # ---- Attention over hyperedge weights ------------------------
        if self.use_attention:
            # Aggregate node features per hyperedge for attention input
            # H^T x : (M, in_channels)
            he_agg = torch.matmul(H.t(), x)
            # Normalise by hyperedge degree to get mean representation
            he_agg = he_agg * d_e_inv.unsqueeze(1).clamp(min=1e-8)

            attn_logits = self.attn_fc(he_agg).squeeze(-1)  # (M,)
            attention_weights = torch.softmax(attn_logits, dim=0)  # (M,)

            # Modulate hyperedge weights
            W_diag = W_diag * attention_weights
        else:
            # Uniform attention -- normalise weights to sum to 1 for
            # consistent output scale.
            attention_weights = W_diag / (W_diag.sum() + 1e-8)

        # ---- Spectral convolution ------------------------------------
        # Step 1: D_v^{-1/2} X
        x_hat = d_v_inv_sqrt.unsqueeze(1) * x  # (N, in_channels)

        # Step 2: H^T (D_v^{-1/2} X)  -->  (M, in_channels)
        he_features = torch.matmul(H.t(), x_hat)

        # Step 3: W D_e^{-1} (H^T D_v^{-1/2} X)
        he_features = W_diag.unsqueeze(1) * d_e_inv.unsqueeze(1) * he_features

        # Step 4: H (W D_e^{-1} H^T D_v^{-1/2} X)  -->  (N, in_channels)
        node_signal = torch.matmul(H, he_features)

        # Step 5: D_v^{-1/2} (H W D_e^{-1} H^T D_v^{-1/2} X)
        node_signal = d_v_inv_sqrt.unsqueeze(1) * node_signal

        # Step 6: Theta^{(k)} -- learnable linear projection
        node_out = self.theta(node_signal)  # (N, out_channels)

        # ---- Hyperedge output representation -------------------------
        # Project aggregated hyperedge features through Theta as well
        hyperedge_out = self.theta(he_features)  # (M, out_channels)

        # ---- Residual connection ------------------------------------
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x if self.in_channels == self.out_channels else 0.0

        if isinstance(residual, torch.Tensor):
            node_out = node_out + residual

        # ---- LayerNorm + activation + dropout -----------------------
        node_out = self.layer_norm(node_out)
        node_out = F.elu(node_out)
        node_out = self.dropout(node_out)

        return node_out, hyperedge_out, attention_weights


# ======================================================================
# Quick smoke test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SpectralHypergraphConv -- Module Info & Smoke Test")
    print("=" * 60)

    in_ch, out_ch = 32, 64
    num_nodes, num_hyperedges = 20, 8

    model = SpectralHypergraphConv(in_ch, out_ch, use_attention=True, dropout=0.1)
    print(f"\nModel:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Synthetic data
    x = torch.randn(num_nodes, in_ch)
    # Random incidence matrix (each hyperedge contains 2-5 vertices)
    H = torch.zeros(num_nodes, num_hyperedges)
    for e in range(num_hyperedges):
        size = torch.randint(2, 6, (1,)).item()
        members = torch.randperm(num_nodes)[:size]
        H[members, e] = 1.0

    W = torch.rand(num_hyperedges)

    node_out, he_out, attn = model(x, H, W)
    print(f"\nInput  x shape:              {x.shape}")
    print(f"Incidence H shape:           {H.shape}")
    print(f"Output node_out shape:       {node_out.shape}")
    print(f"Output hyperedge_out shape:  {he_out.shape}")
    print(f"Attention weights shape:     {attn.shape}")
    print(f"Attention sum:               {attn.sum().item():.4f}")

    # Verify pairwise (size-2) reduction property
    print("\n--- Pairwise reduction check ---")
    H_pair = torch.zeros(4, 3)
    H_pair[0, 0] = 1; H_pair[1, 0] = 1  # edge {0,1}
    H_pair[1, 1] = 1; H_pair[2, 1] = 1  # edge {1,2}
    H_pair[2, 2] = 1; H_pair[3, 2] = 1  # edge {2,3}
    x_small = torch.randn(4, in_ch)
    model_small = SpectralHypergraphConv(in_ch, out_ch, use_attention=False)
    out_small, _, _ = model_small(x_small, H_pair)
    print(f"Pairwise graph output shape: {out_small.shape}  (should be [4, {out_ch}])")

    print("\nSmoke test passed.")
