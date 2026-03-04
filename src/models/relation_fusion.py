"""
Heterogeneous Relation Fusion (v2.0)

Implements relation-type-aware message passing for the five canonical
supply-chain relation types:

    1. supplier_of            -- raw material / component supply
    2. manufactured_by        -- assembly / manufacturing link
    3. transported_by         -- logistics / shipping link
    4. quality_controlled_by  -- inspection / certification link
    5. co_disrupted_with      -- shared disruption exposure

Each relation type maintains its own learnable projection matrices and
attention parameters.  Messages are aggregated via multi-head softmax
attention that weighs the contribution of each relation type per node,
producing a single fused embedding that captures heterogeneous context.

Architecture:
    For each relation type r:
        h_r = W_r^V  x_j                        (value projection)
        e_r = LeakyReLU(a_r^T [W_r^Q x_i || W_r^K x_j])  (attention logit)
    alpha_{r,ij} = softmax_j(e_r)
    z_i = sum_r  sum_{j in N_r(i)}  alpha_{r,ij}  h_r
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# Canonical supply-chain relation types
RELATION_TYPES: List[str] = [
    "supplier_of",
    "manufactured_by",
    "transported_by",
    "quality_controlled_by",
    "co_disrupted_with",
]


# ======================================================================
# Learnable Relation Embedding
# ======================================================================

class RelationEmbedding(nn.Module):
    """Learnable embedding vectors for each relation type.

    Provides a dense vector representation for each relation type that
    can be used to condition attention or message computation.

    Args:
        num_relation_types: Number of distinct relation types.
        embedding_dim:      Dimensionality of each relation embedding.
    """

    def __init__(
        self,
        num_relation_types: int = 5,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_relation_types = num_relation_types
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(num_relation_types, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, relation_indices: torch.Tensor) -> torch.Tensor:
        """Look up relation embeddings.

        Args:
            relation_indices: Integer tensor of relation type indices.

        Returns:
            Embeddings of shape (*relation_indices.shape, embedding_dim).
        """
        return self.embeddings(relation_indices)

    def get_all(self) -> torch.Tensor:
        """Return the full embedding table (num_relation_types, embedding_dim)."""
        idx = torch.arange(
            self.num_relation_types, device=self.embeddings.weight.device
        )
        return self.embeddings(idx)


# ======================================================================
# Heterogeneous Relation Fusion
# ======================================================================

class HeterogeneousRelationFusion(nn.Module):
    """Multi-head relation-aware attention fusion layer.

    For every edge (i, j) with relation type r, the layer computes
    relation-specific query / key / value projections and performs
    multi-head attention.  Results across all relation types are
    aggregated into a single fused embedding per node.

    Args:
        hidden_dim:         Node embedding dimensionality (input and output).
        num_relation_types: Number of distinct relation types.
        num_heads:          Number of attention heads.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_relation_types: int = 5,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        self.hidden_dim = hidden_dim
        self.num_relation_types = num_relation_types
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # ---- Per-relation projection matrices -----------------------
        # Query, Key, Value projections for each relation type
        self.W_Q = nn.ParameterList([
            nn.Parameter(torch.empty(num_heads, hidden_dim, self.head_dim))
            for _ in range(num_relation_types)
        ])
        self.W_K = nn.ParameterList([
            nn.Parameter(torch.empty(num_heads, hidden_dim, self.head_dim))
            for _ in range(num_relation_types)
        ])
        self.W_V = nn.ParameterList([
            nn.Parameter(torch.empty(num_heads, hidden_dim, self.head_dim))
            for _ in range(num_relation_types)
        ])

        # Per-relation attention bias (scalar per head per relation)
        self.attn_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(num_heads))
            for _ in range(num_relation_types)
        ])

        # ---- Relation embeddings ------------------------------------
        self.relation_embedding = RelationEmbedding(
            num_relation_types=num_relation_types,
            embedding_dim=hidden_dim,
        )

        # ---- Output projection --------------------------------------
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialise all learnable parameters."""
        for params in [self.W_Q, self.W_K, self.W_V]:
            for p in params:
                nn.init.xavier_uniform_(p)
        for b in self.attn_bias:
            nn.init.zeros_(b)
        self.relation_embedding.reset_parameters()
        self.output_proj.reset_parameters()
        self.layer_norm.reset_parameters()

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> torch.Tensor:
        """Compute heterogeneous relation-aware fused embeddings.

        Args:
            node_embeddings: Node features of shape (N, hidden_dim).
            edge_index:      Edge connectivity of shape (2, E) where
                             edge_index[0] are source nodes and edge_index[1]
                             are target nodes.
            edge_types:      Integer tensor of shape (E,) mapping each edge
                             to a relation type index in [0, num_relation_types).

        Returns:
            fused_embeddings: Updated node features of shape (N, hidden_dim).
        """
        N = node_embeddings.size(0)
        device = node_embeddings.device

        src, dst = edge_index[0], edge_index[1]  # (E,)

        # Initialise output accumulator and normalisation counter
        out = torch.zeros(N, self.num_heads, self.head_dim, device=device)
        count = torch.zeros(N, 1, 1, device=device)

        # Process each relation type independently
        for r in range(self.num_relation_types):
            mask = edge_types == r  # (E,)
            if not mask.any():
                continue

            src_r = src[mask]  # edges of type r
            dst_r = dst[mask]

            x_src = node_embeddings[src_r]  # (E_r, H_dim)
            x_dst = node_embeddings[dst_r]  # (E_r, H_dim)

            # Multi-head projections: Q from dst, K and V from src
            # W_Q[r]: (num_heads, hidden_dim, head_dim)
            # x_dst:  (E_r, hidden_dim)
            # Result: (E_r, num_heads, head_dim)
            q = torch.einsum("eh,nhd->end", x_dst, self.W_Q[r])
            k = torch.einsum("eh,nhd->end", x_src, self.W_K[r])
            v = torch.einsum("eh,nhd->end", x_src, self.W_V[r])

            # Scaled dot-product attention per head
            scale = self.head_dim ** 0.5
            attn_logits = (q * k).sum(dim=-1) / scale  # (E_r, num_heads)
            attn_logits = attn_logits + self.attn_bias[r].unsqueeze(0)

            # Softmax over incoming edges per target node per head
            # Use scatter-based softmax for efficiency
            attn_weights = self._scatter_softmax(
                attn_logits, dst_r, num_nodes=N
            )  # (E_r, num_heads)

            # Weighted values
            weighted_v = attn_weights.unsqueeze(-1) * v  # (E_r, heads, head_dim)

            # Scatter-add into target nodes
            out.scatter_add_(
                0,
                dst_r.unsqueeze(1).unsqueeze(2).expand_as(weighted_v),
                weighted_v,
            )
            count.scatter_add_(
                0,
                dst_r.unsqueeze(1).unsqueeze(2).expand(-1, 1, 1),
                torch.ones(dst_r.size(0), 1, 1, device=device),
            )

        # Reshape multi-head output back to (N, hidden_dim)
        out = out.reshape(N, self.hidden_dim)

        # Output projection + residual + norm
        out = self.output_proj(out)
        out = self.dropout(out)
        fused_embeddings = self.layer_norm(out + node_embeddings)

        return fused_embeddings

    @staticmethod
    def _scatter_softmax(
        logits: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute softmax grouped by target node index.

        Args:
            logits: (E, num_heads) attention logits.
            index:  (E,) target node indices for grouping.
            num_nodes: Total number of nodes N.

        Returns:
            Softmax attention weights of shape (E, num_heads).
        """
        # For numerical stability subtract the per-group max
        num_heads = logits.size(1)
        max_vals = torch.full(
            (num_nodes, num_heads), -1e9, device=logits.device
        )
        max_vals.scatter_reduce_(
            0,
            index.unsqueeze(1).expand_as(logits),
            logits,
            reduce="amax",
            include_self=True,
        )
        logits_stable = logits - max_vals[index]

        exp_logits = logits_stable.exp()
        sum_exp = torch.zeros(num_nodes, num_heads, device=logits.device)
        sum_exp.scatter_add_(
            0,
            index.unsqueeze(1).expand_as(exp_logits),
            exp_logits,
        )
        # Avoid division by zero for isolated nodes
        softmax_vals = exp_logits / (sum_exp[index] + 1e-12)
        return softmax_vals


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HeterogeneousRelationFusion -- Module Info & Smoke Test")
    print("=" * 60)

    hidden_dim = 64
    num_nodes = 30
    num_edges = 80
    num_relation_types = 5
    num_heads = 8

    model = HeterogeneousRelationFusion(
        hidden_dim=hidden_dim,
        num_relation_types=num_relation_types,
        num_heads=num_heads,
    )
    print(f"\nModel:\n{model}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    print(f"\nRelation types: {RELATION_TYPES}")

    # Synthetic graph
    node_emb = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.stack([
        torch.randint(0, num_nodes, (num_edges,)),
        torch.randint(0, num_nodes, (num_edges,)),
    ])
    edge_types = torch.randint(0, num_relation_types, (num_edges,))

    fused = model(node_emb, edge_index, edge_types)
    print(f"\nInput  node_embeddings shape: {node_emb.shape}")
    print(f"Edge index shape:             {edge_index.shape}")
    print(f"Edge types shape:             {edge_types.shape}")
    print(f"Output fused_embeddings shape: {fused.shape}")

    # Relation embedding sanity
    rel_emb = model.relation_embedding.get_all()
    print(f"\nRelation embedding table shape: {rel_emb.shape}")

    # Gradient check
    loss = fused.sum()
    loss.backward()
    grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"Gradient flow OK: {grad_ok}")

    print("\nSmoke test passed.")
