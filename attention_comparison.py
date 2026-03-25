"""
Attention Mechanism Comparison for HT-HGNN

This module compares 3 different hyperedge aggregation approaches:
1. Uniform aggregation (mean pooling baseline)
2. Learned scalar attention
3. Structural importance attention (our design)

Trains the full HT-HGNN model with each variant and compares:
- Criticality classification accuracy
- Macro F1 score
- Parameter count
- Training time

Results are saved to attention_comparison_results.csv for analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
import logging
from datetime import datetime
import sys
import warnings
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import attention variants (with pure PyTorch fallback)
try:
    from hyperedge_attention import (
        UniformAggregation,
        LearnedScalarAttention,
        StructuralImportanceAttention,
        create_hyperedge_aggregator
    )
    print("[INFO] Using torch_scatter implementation")
except ImportError:
    # Fallback to pure PyTorch implementation
    from hyperedge_attention_pure import (
        UniformAggregationPure as UniformAggregation,
        LearnedScalarAttentionPure as LearnedScalarAttention,
        StructuralImportanceAttentionPure as StructuralImportanceAttention,
        create_hyperedge_aggregator_pure as create_hyperedge_aggregator
    )
    print("[INFO] Using pure PyTorch implementation (torch_scatter not available)")

# Import model and training utilities
from src.models.ht_hgnn_model import HeterogeneousTemporalHypergraphNN
from losses import ContrastiveMultiTaskLoss
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attention_comparison.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Modified HypergraphConvolution to use different aggregation methods
class ParametrizedHypergraphConvolution(nn.Module):
    """
    Modified hypergraph convolution that supports different aggregation methods.

    This wraps the attention variants from hyperedge_attention.py and integrates
    them into the standard hypergraph convolution pipeline.
    """

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int,
                 num_hyperedges: int, aggregation_type: str = 'structural'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.num_hyperedges = num_hyperedges
        self.aggregation_type = aggregation_type

        # Node to hyperedge transformation
        self.node_to_edge = nn.Linear(in_channels, out_channels)
        # Hyperedge to node transformation
        self.edge_to_node = nn.Linear(out_channels, out_channels)

        # Create aggregation module
        self.aggregator = create_hyperedge_aggregator(aggregation_type, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.node_to_edge.reset_parameters()
        self.edge_to_node.reset_parameters()

    def incidence_to_hyperedge_index(self, incidence_matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert incidence matrix to hyperedge_index format for aggregation variants.

        Args:
            incidence_matrix: [num_hyperedges, num_nodes] binary matrix

        Returns:
            hyperedge_index: [2, num_memberships] tensor (node_idx, hyperedge_idx)
        """
        # Find non-zero entries
        hyperedge_indices, node_indices = torch.nonzero(incidence_matrix, as_tuple=True)

        # Stack to create [2, num_memberships] format
        hyperedge_index = torch.stack([node_indices, hyperedge_indices], dim=0)

        return hyperedge_index

    def forward(self, node_features: torch.Tensor,
                incidence_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass using the specified aggregation method.

        Args:
            node_features: [num_nodes, in_channels]
            incidence_matrix: [num_hyperedges, num_nodes] binary matrix

        Returns:
            node_out: Updated node embeddings
            hyperedge_out: Hyperedge embeddings
            attention_weights: Dummy attention (for interface compatibility)
        """
        # Convert incidence matrix to hyperedge_index format
        hyperedge_index = self.incidence_to_hyperedge_index(incidence_matrix)

        # Step 1: Aggregate nodes→hyperedges using the specified method
        hyperedge_features = self.aggregator(
            node_features, hyperedge_index, self.num_hyperedges
        )  # [num_hyperedges, in_channels]

        # Step 2: Transform hyperedge features
        hyperedge_features = self.node_to_edge(hyperedge_features)  # [num_hyperedges, out_channels]

        # Step 3: Propagate hyperedges→nodes
        node_out = torch.matmul(incidence_matrix.t(), hyperedge_features)  # [num_nodes, out_channels]
        node_out = self.edge_to_node(node_out)  # [num_nodes, out_channels]

        # Step 4: Residual connection
        if self.in_channels == self.out_channels:
            node_out = node_out + node_features

        # Dummy attention weights (for interface compatibility)
        attention_weights = torch.ones(self.num_hyperedges, device=node_features.device)

        return node_out, hyperedge_features, attention_weights


class AttentionVariantHTHGNN(HeterogeneousTemporalHypergraphNN):
    """
    HT-HGNN model variant that uses different hyperedge aggregation methods.
    """

    def __init__(self, aggregation_type: str = 'structural', **kwargs):
        # Initialize parent class
        super().__init__(**kwargs)

        self.aggregation_type = aggregation_type

        # Replace HGNN layers with parametrized versions
        self.hgnn_layers = nn.ModuleList([
            ParametrizedHypergraphConvolution(
                self.in_channels if i == 0 else self.hidden_channels,
                self.hidden_channels,
                self.num_nodes,
                self.num_hyperedges,
                aggregation_type=aggregation_type
            )
            for i in range(len(self.hgnn_layers))
        ])


class AttentionComparisonTrainer:
    """
    Trainer for comparing different attention mechanisms in HT-HGNN.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []

    def create_synthetic_data(self, num_nodes: int = 1206, num_hyperedges: int = 36,
                            hidden_dim: int = 64) -> Dict[str, torch.Tensor]:
        """Create synthetic data for comparison experiments."""

        # Node features
        node_features = torch.randn(num_nodes, 18, device=self.device)

        # Random incidence matrix (each hyperedge connects 3-8 nodes)
        incidence_matrix = torch.zeros(num_hyperedges, num_nodes, device=self.device)
        for he in range(num_hyperedges):
            size = torch.randint(3, 9, (1,)).item()
            members = torch.randperm(num_nodes)[:size]
            incidence_matrix[he, members] = 1.0

        # Node types
        nodes_per_type = num_nodes // 3
        node_types = (['supplier'] * nodes_per_type +
                     ['part'] * nodes_per_type +
                     ['transaction'] * (num_nodes - 2 * nodes_per_type))

        # Edge index for HGT (random pairwise connections)
        num_edges = min(5000, num_nodes * 10)  # Reasonable edge density
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=self.device)
        edge_types = ['supplies', 'uses', 'prices'] * (num_edges // 3 + 1)
        edge_types = edge_types[:num_edges]

        # Timestamps
        timestamps = torch.linspace(0, 10, num_nodes, device=self.device)

        # Targets for evaluation
        y_criticality = torch.randint(0, 2, (num_nodes,), device=self.device)  # Binary classification
        y_price = torch.randn(num_nodes, device=self.device)
        y_change = torch.randn(num_nodes, device=self.device)

        return {
            'node_features': node_features,
            'incidence_matrix': incidence_matrix,
            'node_types': node_types,
            'edge_index': edge_index,
            'edge_types': edge_types,
            'timestamps': timestamps,
            'y_criticality': y_criticality,
            'y_price': y_price,
            'y_change': y_change
        }

    def train_single_variant(self, aggregation_type: str, train_data: Dict,
                           val_data: Dict, test_data: Dict,
                           epochs: int = 50) -> Dict[str, Any]:
        """
        Train HT-HGNN with a specific aggregation type.

        Args:
            aggregation_type: 'uniform', 'scalar', or 'structural'
            train_data, val_data, test_data: Data splits
            epochs: Number of training epochs

        Returns:
            Training results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with {aggregation_type.upper()} aggregation")
        logger.info(f"{'='*60}")

        start_time = time.time()

        # Initialize model
        model = AttentionVariantHTHGNN(
            aggregation_type=aggregation_type,
            in_channels=18,
            hidden_channels=64,
            out_channels=32,
            num_nodes=1206,
            num_hyperedges=36,
            node_types=['supplier', 'part', 'transaction'],
            edge_types=['supplies', 'uses', 'prices'],
            num_hgnn_layers=2,
            num_hgt_heads=4,
            time_window=10
        ).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        aggregator_params = sum(p.numel() for layer in model.hgnn_layers
                              for p in layer.aggregator.parameters())

        logger.info(f"Model parameters: {total_params:,}")
        logger.info(f"Aggregator parameters: {aggregator_params:,}")

        # Loss and optimizer
        criterion = ContrastiveMultiTaskLoss(
            weight_price=1.0,
            weight_change=0.5,
            weight_criticality=0.3,
            ssl_weight=0.0,  # Disable SSL for fair comparison
            ssl_temperature=0.1
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Training loop
        model.train()
        train_losses = []

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in tqdm(range(epochs), desc=f"{aggregation_type} training"):
            optimizer.zero_grad()

            # Forward pass
            output = model(
                node_features=train_data['node_features'],
                incidence_matrix=train_data['incidence_matrix'],
                node_types=train_data['node_types'],
                edge_index=train_data['edge_index'],
                edge_types=train_data['edge_types'],
                timestamps=train_data['timestamps']
            )

            # Compute loss
            loss_dict = criterion(
                price_pred=output['price_pred'],
                price_target=train_data['y_price'].unsqueeze(-1),
                change_pred=output['change_pred'],
                change_target=train_data['y_change'].unsqueeze(-1),
                criticality_pred=output['criticality'],
                criticality_target=train_data['y_criticality'].unsqueeze(-1).float()
            )

            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss_dict['total_loss'].item())

            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss={loss_dict['total_loss'].item():.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Test set evaluation
            test_output = model(
                node_features=test_data['node_features'],
                incidence_matrix=test_data['incidence_matrix'],
                node_types=test_data['node_types'],
                edge_index=test_data['edge_index'],
                edge_types=test_data['edge_types'],
                timestamps=test_data['timestamps']
            )

            # Criticality classification metrics
            crit_pred = torch.sigmoid(test_output['criticality']).squeeze() > 0.5
            crit_true = test_data['y_criticality'].bool()

            crit_acc = accuracy_score(crit_true.cpu(), crit_pred.cpu())
            macro_f1 = f1_score(crit_true.cpu(), crit_pred.cpu(), average='macro')

        training_time = time.time() - start_time

        results = {
            'aggregation_type': aggregation_type,
            'crit_acc': crit_acc,
            'macro_f1': macro_f1,
            'param_count': total_params,
            'aggregator_params': aggregator_params,
            'training_time': training_time,
            'final_train_loss': train_losses[-1],
            'epochs': epochs
        }

        logger.info(f"\nResults for {aggregation_type}:")
        logger.info(f"  Accuracy: {crit_acc:.4f}")
        logger.info(f"  Macro F1: {macro_f1:.4f}")
        logger.info(f"  Parameters: {total_params:,}")
        logger.info(f"  Training time: {training_time:.1f}s")

        return results

    def run_attention_comparison(self, train_data: Dict, val_data: Dict, test_data: Dict,
                               epochs: int = 50) -> pd.DataFrame:
        """
        Run complete attention mechanism comparison.

        Args:
            train_data, val_data, test_data: Data splits
            epochs: Number of training epochs per variant

        Returns:
            DataFrame with comparison results
        """
        logger.info("Starting Hyperedge Attention Comparison Study")
        logger.info("=" * 80)

        variants = ['uniform', 'scalar', 'structural']
        results = []

        for variant in variants:
            try:
                result = self.train_single_variant(
                    aggregation_type=variant,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    epochs=epochs
                )
                results.append(result)

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed to train {variant} variant: {e}")
                # Add failed result for completeness
                results.append({
                    'aggregation_type': variant,
                    'crit_acc': 0.0,
                    'macro_f1': 0.0,
                    'param_count': 0,
                    'aggregator_params': 0,
                    'training_time': 0.0,
                    'final_train_loss': float('inf'),
                    'epochs': epochs,
                    'error': str(e)
                })

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Print summary table
        self._print_results_table(results_df)

        # Save results
        output_path = Path('attention_comparison_results.csv')
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to: {output_path}")

        return results_df

    def _print_results_table(self, results_df: pd.DataFrame):
        """Print formatted results table."""

        logger.info(f"\n{'='*80}")
        logger.info("ATTENTION MECHANISM COMPARISON RESULTS")
        logger.info(f"{'='*80}")

        print(f"\n{'Method':<15} {'Accuracy':<10} {'Macro F1':<10} {'Parameters':<12} {'Time (s)':<10}")
        print("-" * 65)

        for _, row in results_df.iterrows():
            print(f"{row['aggregation_type']:<15} "
                  f"{row['crit_acc']:<10.4f} "
                  f"{row['macro_f1']:<10.4f} "
                  f"{row['param_count']:<12,} "
                  f"{row['training_time']:<10.1f}")

        # Find best performing method
        best_idx = results_df['crit_acc'].idxmax()
        best_method = results_df.loc[best_idx]

        print(f"\nBest performing method: {best_method['aggregation_type'].upper()}")
        print(f"  - Accuracy: {best_method['crit_acc']:.4f}")
        print(f"  - Macro F1: {best_method['macro_f1']:.4f}")
        print(f"  - Parameter overhead: +{best_method['aggregator_params']:,} parameters")

        logger.info(f"{'='*80}")


def run_attention_comparison(train_data: Dict, val_data: Dict, test_data: Dict,
                           epochs: int = 50) -> pd.DataFrame:
    """
    Convenience function to run attention comparison study.

    Args:
        train_data, val_data, test_data: Data splits
        epochs: Number of training epochs per variant

    Returns:
        DataFrame with comparison results
    """
    trainer = AttentionComparisonTrainer()
    return trainer.run_attention_comparison(train_data, val_data, test_data, epochs)


def main():
    """Main execution for attention comparison study."""

    print(f"\n{'='*80}")
    print("HT-HGNN HYPEREDGE ATTENTION COMPARISON STUDY")
    print(f"{'='*80}")

    # Configuration
    epochs = 30  # Reduced for faster comparison
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}")
    logger.info(f"Epochs per variant: {epochs}")

    try:
        # Initialize trainer
        trainer = AttentionComparisonTrainer(device=device)

        # Create synthetic data splits
        logger.info("Creating synthetic data...")
        full_data = trainer.create_synthetic_data(num_nodes=1206, num_hyperedges=36)

        # Simple temporal split (70/15/15)
        n_nodes = 1206
        train_end = int(0.7 * n_nodes)
        val_end = int(0.85 * n_nodes)

        train_data = {k: v[:train_end] if v.ndim > 0 and len(v) == n_nodes else v
                     for k, v in full_data.items()}
        val_data = {k: v[train_end:val_end] if v.ndim > 0 and len(v) == n_nodes else v
                   for k, v in full_data.items()}
        test_data = {k: v[val_end:] if v.ndim > 0 and len(v) == n_nodes else v
                    for k, v in full_data.items()}

        logger.info(f"Data splits: Train={len(train_data['y_criticality'])}, "
                   f"Val={len(val_data['y_criticality'])}, "
                   f"Test={len(test_data['y_criticality'])}")

        # Run comparison study
        results_df = trainer.run_attention_comparison(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            epochs=epochs
        )

        print(f"\n{'='*80}")
        print("ATTENTION COMPARISON STUDY COMPLETE!")
        print(f"{'='*80}")
        print(f"Results saved to: attention_comparison_results.csv")
        print(f"Logs saved to: attention_comparison.log")
        print(f"{'='*80}")

    except Exception as e:
        logger.error(f"Attention comparison failed: {e}")
        raise


if __name__ == "__main__":
    main()