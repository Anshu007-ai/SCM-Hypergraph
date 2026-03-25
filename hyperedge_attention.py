# pip install torch_scatter
"""
Hyperedge Aggregation Variants for SpectralHypergraphConv

This module implements 3 different approaches to aggregate node features within hyperedges:

1. UniformAggregation: Simple mean pooling without learnable parameters
2. LearnedScalarAttention: Scalar attention scores per node with learned weights
3. StructuralImportanceAttention: Distance-based attention in projected embedding space

All variants follow the same interface:
    forward(node_feats, hyperedge_index, num_hyperedges) -> hyperedge_feats

Where:
    node_feats: [num_nodes, hidden_dim] node feature tensor
    hyperedge_index: [2, num_memberships] tensor (node_idx, hyperedge_idx)
    num_hyperedges: int, total number of hyperedges
    Returns: [num_hyperedges, hidden_dim] aggregated hyperedge features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import torch_scatter


class UniformAggregation(nn.Module):
    """
    Uniform aggregation using simple mean pooling.

    No learnable parameters - just averages node features within each hyperedge.
    This serves as the simplest baseline for comparison.
    """

    def __init__(self, hidden_dim: int):
        """
        Initialize uniform aggregation.

        Args:
            hidden_dim: Feature dimensionality (not used but kept for interface consistency)
        """
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self,
                node_feats: torch.Tensor,
                hyperedge_index: torch.Tensor,
                num_hyperedges: int) -> torch.Tensor:
        """
        Aggregate node features using mean pooling.

        Args:
            node_feats: [num_nodes, hidden_dim] node features
            hyperedge_index: [2, num_memberships] connectivity tensor
                           Row 0: node indices, Row 1: hyperedge indices
            num_hyperedges: Total number of hyperedges

        Returns:
            hyperedge_feats: [num_hyperedges, hidden_dim] aggregated features
        """
        node_idx, hyperedge_idx = hyperedge_index[0], hyperedge_index[1]

        # Mean pool node features within each hyperedge
        hyperedge_feats = torch_scatter.scatter_mean(
            src=node_feats[node_idx],      # [num_memberships, hidden_dim]
            index=hyperedge_idx,           # [num_memberships]
            dim=0,
            dim_size=num_hyperedges        # Output: [num_hyperedges, hidden_dim]
        )

        return hyperedge_feats


class LearnedScalarAttention(nn.Module):
    """
    Learned scalar attention mechanism.

    Computes a scalar attention score for each node, then performs
    softmax within each hyperedge and weighted aggregation.
    """

    def __init__(self, hidden_dim: int):
        """
        Initialize learned scalar attention.

        Args:
            hidden_dim: Feature dimensionality
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Single linear layer to compute scalar attention scores
        self.attention_linear = nn.Linear(hidden_dim, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.attention_linear.weight)

    def forward(self,
                node_feats: torch.Tensor,
                hyperedge_index: torch.Tensor,
                num_hyperedges: int) -> torch.Tensor:
        """
        Aggregate node features using learned scalar attention.

        Args:
            node_feats: [num_nodes, hidden_dim] node features
            hyperedge_index: [2, num_memberships] connectivity tensor
            num_hyperedges: Total number of hyperedges

        Returns:
            hyperedge_feats: [num_hyperedges, hidden_dim] aggregated features
        """
        node_idx, hyperedge_idx = hyperedge_index[0], hyperedge_index[1]

        # Compute scalar attention scores for each node
        attention_scores = self.attention_linear(node_feats[node_idx]).squeeze(-1)  # [num_memberships]

        # Apply softmax within each hyperedge to get attention weights
        attention_weights = torch_scatter.scatter_softmax(
            src=attention_scores,          # [num_memberships]
            index=hyperedge_idx,          # [num_memberships]
            dim=0
        )  # [num_memberships]

        # Weighted aggregation of node features
        weighted_node_feats = attention_weights.unsqueeze(-1) * node_feats[node_idx]  # [num_memberships, hidden_dim]

        hyperedge_feats = torch_scatter.scatter_add(
            src=weighted_node_feats,       # [num_memberships, hidden_dim]
            index=hyperedge_idx,          # [num_memberships]
            dim=0,
            dim_size=num_hyperedges       # Output: [num_hyperedges, hidden_dim]
        )

        return hyperedge_feats


class StructuralImportanceAttention(nn.Module):
    """
    Structural importance attention mechanism (our advanced design).

    Uses Q/K/V projections and computes attention based on L2 distance
    from hyperedge centroids in K-space. Nodes far from the centroid
    are considered structurally unusual and get higher importance.
    """

    def __init__(self, hidden_dim: int):
        """
        Initialize structural importance attention.

        Args:
            hidden_dim: Feature dimensionality
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = hidden_dim // 4  # Reduced dimensionality for Q/K/V

        # Q/K/V projection layers
        self.query_proj = nn.Linear(hidden_dim, self.proj_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, self.proj_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, self.proj_dim, bias=False)

        # Scale factor for attention scores
        self.scale = 1.0 / torch.sqrt(torch.tensor(self.proj_dim, dtype=torch.float))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)

    def forward(self,
                node_feats: torch.Tensor,
                hyperedge_index: torch.Tensor,
                num_hyperedges: int) -> torch.Tensor:
        """
        Aggregate node features using structural importance attention.

        Args:
            node_feats: [num_nodes, hidden_dim] node features
            hyperedge_index: [2, num_memberships] connectivity tensor
            num_hyperedges: Total number of hyperedges

        Returns:
            hyperedge_feats: [num_hyperedges, hidden_dim] aggregated features
        """
        node_idx, hyperedge_idx = hyperedge_index[0], hyperedge_idx[1]

        # Project node features to Q/K/V spaces
        node_keys = self.key_proj(node_feats[node_idx])      # [num_memberships, proj_dim]
        node_values = self.value_proj(node_feats[node_idx])  # [num_memberships, proj_dim]

        # Compute hyperedge centroids in K-space using mean pooling
        hyperedge_centroids = torch_scatter.scatter_mean(
            src=node_keys,                 # [num_memberships, proj_dim]
            index=hyperedge_idx,          # [num_memberships]
            dim=0,
            dim_size=num_hyperedges       # Output: [num_hyperedges, proj_dim]
        )

        # Expand centroids to match node dimensions for distance computation
        node_centroids = hyperedge_centroids[hyperedge_idx]  # [num_memberships, proj_dim]

        # Compute L2 distance from each node to its hyperedge centroid
        # Nodes far from centroid = high structural importance
        distances = torch.norm(node_keys - node_centroids, p=2, dim=-1)  # [num_memberships]

        # Scale distances and use as attention scores
        attention_scores = distances * self.scale

        # Apply softmax within each hyperedge to get attention weights
        attention_weights = torch_scatter.scatter_softmax(
            src=attention_scores,         # [num_memberships]
            index=hyperedge_idx,         # [num_memberships]
            dim=0
        )  # [num_memberships]

        # Weighted aggregation using V projections
        weighted_values = attention_weights.unsqueeze(-1) * node_values  # [num_memberships, proj_dim]

        hyperedge_feats = torch_scatter.scatter_add(
            src=weighted_values,          # [num_memberships, proj_dim]
            index=hyperedge_idx,         # [num_memberships]
            dim=0,
            dim_size=num_hyperedges      # Output: [num_hyperedges, proj_dim]
        )

        # Project back to original feature dimension
        # Note: This ensures output has same dimensionality as input for fair comparison
        hyperedge_feats = F.linear(
            hyperedge_feats,
            weight=self.value_proj.weight.t()  # Transpose to get [proj_dim, hidden_dim]
        )  # [num_hyperedges, hidden_dim]

        return hyperedge_feats


def create_hyperedge_aggregator(aggregation_type: str, hidden_dim: int) -> nn.Module:
    """
    Factory function to create hyperedge aggregation modules.

    Args:
        aggregation_type: One of 'uniform', 'scalar', 'structural'
        hidden_dim: Feature dimensionality

    Returns:
        Hyperedge aggregation module

    Raises:
        ValueError: If aggregation_type is not recognized
    """
    if aggregation_type == 'uniform':
        return UniformAggregation(hidden_dim)
    elif aggregation_type == 'scalar':
        return LearnedScalarAttention(hidden_dim)
    elif aggregation_type == 'structural':
        return StructuralImportanceAttention(hidden_dim)
    else:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}. "
                        f"Must be one of 'uniform', 'scalar', 'structural'.")


if __name__ == "__main__":
    print("=" * 60)
    print("Hyperedge Aggregation Variants - Test Suite")
    print("=" * 60)

    # Test configuration
    num_nodes = 10
    num_hyperedges = 4
    hidden_dim = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Synthetic data
    node_feats = torch.randn(num_nodes, hidden_dim, device=device)

    # Create hyperedge topology: each hyperedge connects 3-4 nodes
    hyperedge_index = []
    for he_id in range(num_hyperedges):
        # Random hyperedge size between 3 and 4
        he_size = torch.randint(3, 5, (1,)).item()
        # Random node members
        members = torch.randperm(num_nodes)[:he_size]

        for node_id in members:
            hyperedge_index.append([node_id.item(), he_id])

    hyperedge_index = torch.tensor(hyperedge_index, device=device).t()  # [2, num_memberships]

    print(f"Test setup:")
    print(f"  Nodes: {num_nodes}, Hyperedges: {num_hyperedges}")
    print(f"  Feature dim: {hidden_dim}")
    print(f"  Device: {device}")
    print(f"  Hyperedge memberships: {hyperedge_index.shape[1]}")

    # Test all 3 variants
    variants = ['uniform', 'scalar', 'structural']

    for variant in variants:
        print(f"\n--- Testing {variant.upper()} aggregation ---")

        # Create aggregator
        aggregator = create_hyperedge_aggregator(variant, hidden_dim).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in aggregator.parameters())

        # Forward pass
        hyperedge_feats = aggregator(node_feats, hyperedge_index, num_hyperedges)

        print(f"  Parameters: {total_params:,}")
        print(f"  Input shape: {node_feats.shape}")
        print(f"  Output shape: {hyperedge_feats.shape}")
        print(f"  Output range: [{hyperedge_feats.min():.3f}, {hyperedge_feats.max():.3f}]")

        # Check for NaN or inf
        if torch.isnan(hyperedge_feats).any():
            print(f"  [ERROR] NaN detected in output!")
        elif torch.isinf(hyperedge_feats).any():
            print(f"  [ERROR] Inf detected in output!")
        else:
            print(f"  [OK] Output is finite and valid")

    print(f"\n{'='*60}")
    print("Hyperedge aggregation test complete!")
    print(f"{'='*60}")