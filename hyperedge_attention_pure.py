"""
Pure PyTorch Hyperedge Aggregation Variants (No torch_scatter required)

This module implements the same 3 hyperedge aggregation approaches as hyperedge_attention.py
but uses only standard PyTorch operations. This avoids dependency issues with torch_scatter.

All variants follow the same interface:
    forward(node_feats, hyperedge_index, num_hyperedges) -> hyperedge_feats
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class UniformAggregationPure(nn.Module):
    """
    Uniform aggregation using pure PyTorch (no torch_scatter).
    Simple mean pooling without learnable parameters.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self,
                node_feats: torch.Tensor,
                hyperedge_index: torch.Tensor,
                num_hyperedges: int) -> torch.Tensor:
        """Aggregate node features using mean pooling (pure PyTorch)."""

        node_idx, hyperedge_idx = hyperedge_index[0], hyperedge_index[1]

        # Initialize hyperedge features
        hyperedge_feats = torch.zeros(num_hyperedges, self.hidden_dim,
                                    device=node_feats.device, dtype=node_feats.dtype)
        hyperedge_counts = torch.zeros(num_hyperedges, device=node_feats.device)

        # Aggregate features per hyperedge
        for i in range(len(node_idx)):
            n_idx = node_idx[i]
            he_idx = hyperedge_idx[i]
            hyperedge_feats[he_idx] += node_feats[n_idx]
            hyperedge_counts[he_idx] += 1

        # Compute mean (avoid division by zero)
        hyperedge_counts = hyperedge_counts.clamp(min=1)
        hyperedge_feats = hyperedge_feats / hyperedge_counts.unsqueeze(1)

        return hyperedge_feats


class LearnedScalarAttentionPure(nn.Module):
    """
    Learned scalar attention using pure PyTorch (no torch_scatter).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_linear = nn.Linear(hidden_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attention_linear.weight)

    def forward(self,
                node_feats: torch.Tensor,
                hyperedge_index: torch.Tensor,
                num_hyperedges: int) -> torch.Tensor:
        """Aggregate using learned scalar attention (pure PyTorch)."""

        node_idx, hyperedge_idx = hyperedge_index[0], hyperedge_index[1]

        # Compute attention scores for participating nodes
        attention_scores = self.attention_linear(node_feats[node_idx]).squeeze(-1)  # [num_memberships]

        # Initialize outputs
        hyperedge_feats = torch.zeros(num_hyperedges, self.hidden_dim,
                                    device=node_feats.device, dtype=node_feats.dtype)

        # Process each hyperedge separately for softmax
        for he in range(num_hyperedges):
            # Find nodes in this hyperedge
            mask = (hyperedge_idx == he)
            if not mask.any():
                continue

            nodes_in_he = node_idx[mask]
            scores_in_he = attention_scores[mask]

            # Apply softmax within this hyperedge
            attn_weights = torch.softmax(scores_in_he, dim=0)

            # Weighted aggregation
            weighted_feats = attn_weights.unsqueeze(1) * node_feats[nodes_in_he]
            hyperedge_feats[he] = weighted_feats.sum(dim=0)

        return hyperedge_feats


class StructuralImportanceAttentionPure(nn.Module):
    """
    Structural importance attention using pure PyTorch (no torch_scatter).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = hidden_dim // 4

        # Q/K/V projection layers
        self.query_proj = nn.Linear(hidden_dim, self.proj_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, self.proj_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, self.proj_dim, bias=False)

        # Scale factor
        self.scale = 1.0 / torch.sqrt(torch.tensor(self.proj_dim, dtype=torch.float))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)

    def forward(self,
                node_feats: torch.Tensor,
                hyperedge_index: torch.Tensor,
                num_hyperedges: int) -> torch.Tensor:
        """Aggregate using structural importance attention (pure PyTorch)."""

        node_idx, hyperedge_idx = hyperedge_index[0], hyperedge_index[1]

        # Project participating node features
        participating_nodes = node_feats[node_idx]  # [num_memberships, hidden_dim]
        node_keys = self.key_proj(participating_nodes)     # [num_memberships, proj_dim]
        node_values = self.value_proj(participating_nodes)  # [num_memberships, proj_dim]

        # Initialize outputs
        hyperedge_feats = torch.zeros(num_hyperedges, self.proj_dim,
                                    device=node_feats.device, dtype=node_feats.dtype)

        # Process each hyperedge separately
        for he in range(num_hyperedges):
            # Find nodes in this hyperedge
            mask = (hyperedge_idx == he)
            if not mask.any():
                continue

            he_keys = node_keys[mask]      # [nodes_in_he, proj_dim]
            he_values = node_values[mask]  # [nodes_in_he, proj_dim]

            # Compute centroid in K-space
            centroid = he_keys.mean(dim=0, keepdim=True)  # [1, proj_dim]

            # Compute L2 distances from centroid
            distances = torch.norm(he_keys - centroid, p=2, dim=-1)  # [nodes_in_he]

            # Use distances as attention scores (scaled)
            attention_scores = distances * self.scale

            # Apply softmax
            attn_weights = torch.softmax(attention_scores, dim=0)  # [nodes_in_he]

            # Weighted aggregation using values
            weighted_values = attn_weights.unsqueeze(1) * he_values  # [nodes_in_he, proj_dim]
            aggregated = weighted_values.sum(dim=0)  # [proj_dim]

            hyperedge_feats[he] = aggregated

        # Project back to original dimension
        hyperedge_feats = F.linear(
            hyperedge_feats,
            weight=self.value_proj.weight.t()  # [proj_dim, hidden_dim]
        )

        return hyperedge_feats


def create_hyperedge_aggregator_pure(aggregation_type: str, hidden_dim: int) -> nn.Module:
    """
    Factory function for pure PyTorch aggregation modules.

    Args:
        aggregation_type: One of 'uniform', 'scalar', 'structural'
        hidden_dim: Feature dimensionality

    Returns:
        Pure PyTorch aggregation module
    """
    if aggregation_type == 'uniform':
        return UniformAggregationPure(hidden_dim)
    elif aggregation_type == 'scalar':
        return LearnedScalarAttentionPure(hidden_dim)
    elif aggregation_type == 'structural':
        return StructuralImportanceAttentionPure(hidden_dim)
    else:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}")


if __name__ == "__main__":
    print("=" * 60)
    print("Pure PyTorch Hyperedge Aggregation - Test Suite")
    print("=" * 60)

    # Test configuration
    num_nodes = 10
    num_hyperedges = 4
    hidden_dim = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Synthetic data
    node_feats = torch.randn(num_nodes, hidden_dim, device=device)

    # Create hyperedge topology
    hyperedge_index = []
    for he_id in range(num_hyperedges):
        he_size = torch.randint(3, 5, (1,)).item()
        members = torch.randperm(num_nodes)[:he_size]
        for node_id in members:
            hyperedge_index.append([node_id.item(), he_id])

    hyperedge_index = torch.tensor(hyperedge_index, device=device).t()

    print(f"Test setup:")
    print(f"  Nodes: {num_nodes}, Hyperedges: {num_hyperedges}")
    print(f"  Feature dim: {hidden_dim}")
    print(f"  Device: {device}")
    print(f"  Memberships: {hyperedge_index.shape[1]}")

    # Test all variants
    variants = ['uniform', 'scalar', 'structural']

    for variant in variants:
        print(f"\n--- Testing {variant.upper()} aggregation (Pure PyTorch) ---")

        aggregator = create_hyperedge_aggregator_pure(variant, hidden_dim).to(device)
        total_params = sum(p.numel() for p in aggregator.parameters())

        # Forward pass
        hyperedge_feats = aggregator(node_feats, hyperedge_index, num_hyperedges)

        print(f"  Parameters: {total_params:,}")
        print(f"  Input shape: {node_feats.shape}")
        print(f"  Output shape: {hyperedge_feats.shape}")
        print(f"  Output range: [{hyperedge_feats.min():.3f}, {hyperedge_feats.max():.3f}]")

        # Check for NaN or inf
        if torch.isnan(hyperedge_feats).any():
            print(f"  [ERROR] NaN detected!")
        elif torch.isinf(hyperedge_feats).any():
            print(f"  [ERROR] Inf detected!")
        else:
            print(f"  [OK] Output is finite and valid")

    print(f"\n{'='*60}")
    print("[SUCCESS] Pure PyTorch hyperedge aggregation test complete!")
    print("No torch_scatter required - ready to run!")
    print(f"{'='*60}")