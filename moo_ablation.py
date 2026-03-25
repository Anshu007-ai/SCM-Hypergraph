"""
Multi-Objective Optimization (MOO) Ablation Study for HT-HGNN

This module provides infrastructure to test the contribution of each MOO transfer
mechanism in the 2-stage HT-HGNN pipeline:

Stage 1: MOO over aviation hypergraph
Stage 2: HT-HGNN neural network with MOO integration

MOO Transfer Mechanisms:
A. Feature-level: 4-dim MOO knee-point embedding → appended to 12-dim raw features (16-dim total)
B. Loss weights: MOO knee-point solution → initializes 4 task loss weights
C. HIC targets: MOO-calibrated HIC cascade simulation → KL divergence targets

Ablation Conditions:
1. CONFIG_FULL: All three mechanisms enabled (A+B+C)
2. CONFIG_NO_FEATURE: Only B+C (12-dim features + zeros padding)
3. CONFIG_NO_LOSS: Only A+C (fixed loss weights [1.0, 0.8, 1.2, 0.6])
4. CONFIG_NO_HIC: Only A+B (uniform cascade targets)
5. CONFIG_NONE: Pure neural baseline (no MOO integration)

Usage:
    from moo_ablation import MOOTransferConfig, run_moo_ablation_study

    # Single config
    config = CONFIG_FULL
    features = build_node_features(raw_features, moo_embedding, config)

    # Full study
    results = run_moo_ablation_study(train_data, val_data, test_data, HTHGNNModel)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class MOOTransferConfig:
    """
    Configuration for MOO transfer mechanisms in HT-HGNN.

    Controls which Multi-Objective Optimization outputs are used
    to enhance the neural network training.

    Attributes:
        use_moo_features: If True, append 4-dim MOO knee-point embedding to
                         12-dim raw node features (making 16-dim total).
        use_moo_loss_weights: If True, use MOO knee-point solution to initialize
                             the 4 task loss weights [lambda_delay, lambda_cancel,
                             lambda_crit, lambda_cascade].
        use_hic_targets: If True, use MOO-calibrated HIC cascade simulation
                        outputs as KL divergence targets for cascade head.
        name: Human-readable name for this configuration.
        description: Detailed description of what's enabled/disabled.
    """
    use_moo_features: bool = True
    use_moo_loss_weights: bool = True
    use_hic_targets: bool = True
    name: str = "Full MOO"
    description: str = "All MOO mechanisms enabled"

    def __post_init__(self):
        """Validate configuration and auto-generate name if needed."""
        if self.name == "Full MOO" and not all([self.use_moo_features, self.use_moo_loss_weights, self.use_hic_targets]):
            # Auto-generate name based on enabled mechanisms
            enabled = []
            if self.use_moo_features:
                enabled.append("Features")
            if self.use_moo_loss_weights:
                enabled.append("Loss")
            if self.use_hic_targets:
                enabled.append("HIC")

            if not enabled:
                self.name = "Pure Neural"
            else:
                self.name = f"MOO {'+'.join(enabled)}"

    @property
    def enabled_mechanisms(self) -> List[str]:
        """Return list of enabled mechanism names."""
        mechanisms = []
        if self.use_moo_features:
            mechanisms.append("Feature-level embedding")
        if self.use_moo_loss_weights:
            mechanisms.append("Loss weight initialization")
        if self.use_hic_targets:
            mechanisms.append("HIC cascade targets")
        return mechanisms

    @property
    def input_dim(self) -> int:
        """Return expected input feature dimension (always 16 for consistency)."""
        return 16  # Always 16: either 12+4 MOO or 12+4 zeros

    def summary(self) -> str:
        """Return a formatted summary of the configuration."""
        enabled = self.enabled_mechanisms
        if not enabled:
            return f"{self.name}: Pure neural baseline (no MOO integration)"
        return f"{self.name}: {', '.join(enabled)}"


# ============================================================================
# Pre-defined Ablation Configurations
# ============================================================================

CONFIG_FULL = MOOTransferConfig(
    use_moo_features=True,
    use_moo_loss_weights=True,
    use_hic_targets=True,
    name="Full MOO",
    description="All three MOO transfer mechanisms enabled (A+B+C)"
)

CONFIG_NO_FEATURE = MOOTransferConfig(
    use_moo_features=False,
    use_moo_loss_weights=True,
    use_hic_targets=True,
    name="No Feature",
    description="Loss weights + HIC targets only (B+C), zero-padded features"
)

CONFIG_NO_LOSS = MOOTransferConfig(
    use_moo_features=True,
    use_moo_loss_weights=False,
    use_hic_targets=True,
    name="No Loss",
    description="Feature embedding + HIC targets only (A+C), fixed loss weights"
)

CONFIG_NO_HIC = MOOTransferConfig(
    use_moo_features=True,
    use_moo_loss_weights=True,
    use_hic_targets=False,
    name="No HIC",
    description="Feature embedding + loss weights only (A+B), uniform cascade targets"
)

CONFIG_NONE = MOOTransferConfig(
    use_moo_features=False,
    use_moo_loss_weights=False,
    use_hic_targets=False,
    name="Pure Neural",
    description="Pure neural baseline with no MOO integration"
)

# All configs for iteration
ALL_CONFIGS = [CONFIG_FULL, CONFIG_NO_FEATURE, CONFIG_NO_LOSS, CONFIG_NO_HIC, CONFIG_NONE]

# Config lookup by name
CONFIG_MAP = {
    "full": CONFIG_FULL,
    "no_feature": CONFIG_NO_FEATURE,
    "no_loss": CONFIG_NO_LOSS,
    "no_hic": CONFIG_NO_HIC,
    "none": CONFIG_NONE
}


# ============================================================================
# MOO Transfer Functions
# ============================================================================

def build_node_features(
    raw_features: torch.Tensor,
    moo_embedding: torch.Tensor,
    config: MOOTransferConfig
) -> torch.Tensor:
    """
    Build node features according to MOO transfer configuration.

    Args:
        raw_features: Base node features tensor of shape (N, 12)
        moo_embedding: MOO knee-point embedding tensor of shape (N, 4) or (4,) for broadcast
        config: MOO transfer configuration

    Returns:
        Enhanced features tensor of shape (N, 16) - always 16-dim for consistency

    Note:
        Input dimension is always 16 to avoid model architecture changes.
        When MOO features are disabled, the last 4 dimensions are zero-padded.
    """
    if raw_features.dim() == 1:
        raw_features = raw_features.unsqueeze(0)
    if moo_embedding.dim() == 1:
        moo_embedding = moo_embedding.unsqueeze(0).expand(raw_features.size(0), -1)

    batch_size, raw_dim = raw_features.shape

    if raw_dim != 12:
        logger.warning(f"Expected 12-dim raw features, got {raw_dim}-dim. Adjusting...")

    if config.use_moo_features:
        # Concatenate MOO embedding: 12-dim raw + 4-dim MOO = 16-dim total
        enhanced_features = torch.cat([raw_features, moo_embedding], dim=1)
        logger.debug(f"Enhanced features with MOO embedding: {raw_features.shape} + {moo_embedding.shape} = {enhanced_features.shape}")
    else:
        # Zero-pad to maintain 16-dim input: 12-dim raw + 4-dim zeros = 16-dim total
        zero_padding = torch.zeros_like(moo_embedding)
        enhanced_features = torch.cat([raw_features, zero_padding], dim=1)
        logger.debug(f"Zero-padded features (MOO disabled): {raw_features.shape} + {zero_padding.shape} = {enhanced_features.shape}")

    return enhanced_features


def get_loss_weights(
    moo_knee_weights: torch.Tensor,
    config: MOOTransferConfig,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Get task loss weights according to MOO transfer configuration.

    Args:
        moo_knee_weights: MOO-derived loss weights tensor of shape (4,)
                         [lambda_delay, lambda_cancel, lambda_crit, lambda_cascade]
        config: MOO transfer configuration
        device: Target device for tensor (auto-detected if None)

    Returns:
        Loss weights tensor of shape (4,)
    """
    if device is None:
        device = moo_knee_weights.device if isinstance(moo_knee_weights, torch.Tensor) else torch.device('cpu')

    if config.use_moo_loss_weights:
        # Use MOO-optimized weights
        if not isinstance(moo_knee_weights, torch.Tensor):
            moo_knee_weights = torch.tensor(moo_knee_weights, dtype=torch.float32, device=device)

        logger.debug(f"Using MOO loss weights: {moo_knee_weights.tolist()}")
        return moo_knee_weights.to(device)
    else:
        # Use fixed default weights [delay, cancel, criticality, cascade]
        fixed_weights = torch.tensor([1.0, 0.8, 1.2, 0.6], dtype=torch.float32, device=device)

        logger.debug(f"Using fixed loss weights: {fixed_weights.tolist()}")
        return fixed_weights


def get_cascade_targets(
    hic_targets: torch.Tensor,
    config: MOOTransferConfig,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Get cascade targets according to MOO transfer configuration.

    Args:
        hic_targets: MOO-calibrated HIC cascade simulation targets
        config: MOO transfer configuration
        device: Target device for tensor (auto-detected if None)

    Returns:
        Cascade targets tensor (same shape as hic_targets if enabled,
                                uniform distribution if disabled)
    """
    if device is None:
        device = hic_targets.device if isinstance(hic_targets, torch.Tensor) else torch.device('cpu')

    if config.use_hic_targets:
        # Use MOO-calibrated HIC targets
        if not isinstance(hic_targets, torch.Tensor):
            hic_targets = torch.tensor(hic_targets, dtype=torch.float32, device=device)

        logger.debug(f"Using HIC cascade targets: shape {hic_targets.shape}")
        return hic_targets.to(device)
    else:
        # Use uniform probability distribution (uninformative baseline)
        uniform_targets = torch.ones_like(hic_targets, dtype=torch.float32, device=device)
        uniform_targets = uniform_targets / uniform_targets.sum(dim=-1, keepdim=True)  # Normalize to probability

        logger.debug(f"Using uniform cascade targets: shape {uniform_targets.shape}")
        return uniform_targets


# ============================================================================
# Ablation Study Runner
# ============================================================================

def run_moo_ablation_study(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    test_data: Dict[str, Any],
    base_model_class: type,
    moo_embedding: torch.Tensor,
    moo_knee_weights: torch.Tensor,
    hic_targets: torch.Tensor,
    epochs: int = 100,
    device: torch.device = torch.device('cpu'),
    save_models: bool = False,
    output_dir: str = "outputs/moo_ablation"
) -> pd.DataFrame:
    """
    Run complete MOO ablation study across all 5 configurations.

    Tests the contribution of each MOO transfer mechanism by training
    separate models with different combinations of mechanisms enabled.

    Args:
        train_data: Training dataset dictionary
        val_data: Validation dataset dictionary
        test_data: Test dataset dictionary
        base_model_class: HT-HGNN model class to instantiate
        moo_embedding: MOO knee-point embeddings (N, 4)
        moo_knee_weights: MOO-derived loss weights (4,)
        hic_targets: MOO-calibrated HIC cascade targets
        epochs: Number of training epochs per config
        device: Training device
        save_models: Whether to save trained models
        output_dir: Directory for saving results and models

    Returns:
        DataFrame with results for all configurations
    """
    print("="*80)
    print("MOO ABLATION STUDY - HT-HGNN")
    print("="*80)
    print(f"🎯 Testing {len(ALL_CONFIGS)} configurations:")
    for i, config in enumerate(ALL_CONFIGS, 1):
        print(f"   {i}. {config.summary()}")
    print()

    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for config_idx, config in enumerate(ALL_CONFIGS):
        print(f"[{config_idx+1}/{len(ALL_CONFIGS)}] Training: {config.name}")
        print("-" * 60)

        try:
            # Prepare data according to config
            train_features = build_node_features(
                train_data['features'], moo_embedding, config
            )
            val_features = build_node_features(
                val_data['features'], moo_embedding, config
            )
            test_features = build_node_features(
                test_data['features'], moo_embedding, config
            )

            loss_weights = get_loss_weights(moo_knee_weights, config, device)
            cascade_targets = get_cascade_targets(hic_targets, config, device)

            # Initialize model
            model = base_model_class(
                in_channels=config.input_dim,  # Always 16
                hidden_channels=64,
                out_channels=32,
                num_nodes=train_features.size(0),
                num_hyperedges=train_data.get('num_hyperedges', 36),
                node_types=['supplier', 'part', 'transaction'],
                edge_types=['supplies', 'uses', 'prices'],
                num_hgnn_layers=2,
                num_hgt_heads=4,
                time_window=10
            ).to(device)

            # Train model (simplified training loop - replace with your training function)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            best_val_acc = 0.0
            convergence_epoch = epochs
            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                # Training step
                model.train()
                optimizer.zero_grad()

                # Forward pass with config-specific data
                output = model(
                    node_features=train_features,
                    incidence_matrix=train_data['incidence_matrix'],
                    node_types=train_data['node_types'],
                    edge_index=train_data['edge_index'],
                    edge_types=train_data['edge_types'],
                    timestamps=train_data['timestamps']
                )

                # Compute loss with config-specific weights and targets
                loss = compute_multi_task_loss(
                    output, train_data['labels'], cascade_targets, loss_weights
                )

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                # Validation step
                if (epoch + 1) % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_output = model(
                            node_features=val_features,
                            incidence_matrix=val_data['incidence_matrix'],
                            node_types=val_data['node_types'],
                            edge_index=val_data['edge_index'],
                            edge_types=val_data['edge_types'],
                            timestamps=val_data['timestamps']
                        )

                        val_loss = compute_multi_task_loss(
                            val_output, val_data['labels'], cascade_targets, loss_weights
                        )
                        val_losses.append(val_loss.item())

                        # Compute validation accuracy
                        val_acc = compute_accuracy(val_output, val_data['labels'])

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            convergence_epoch = epoch + 1

                        print(f"   Epoch {epoch+1:3d}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}, Val Acc={val_acc:.3f}")

            # Final evaluation on test set
            model.eval()
            with torch.no_grad():
                test_output = model(
                    node_features=test_features,
                    incidence_matrix=test_data['incidence_matrix'],
                    node_types=test_data['node_types'],
                    edge_index=test_data['edge_index'],
                    edge_types=test_data['edge_types'],
                    timestamps=test_data['timestamps']
                )

                # Compute test metrics
                test_crit_acc = compute_accuracy(test_output, test_data['labels'])
                test_macro_f1 = compute_f1_score(test_output, test_data['labels'])
                test_cascade_mae = compute_cascade_mae(test_output, cascade_targets)

            # Save model if requested
            if save_models:
                model_path = output_path / f"model_{config.name.lower().replace(' ', '_')}.pt"
                torch.save({
                    'config': config,
                    'model_state_dict': model.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_acc': best_val_acc,
                    'convergence_epoch': convergence_epoch
                }, model_path)

            # Record results
            result = {
                'config_name': config.name,
                'use_moo_features': config.use_moo_features,
                'use_moo_loss_weights': config.use_moo_loss_weights,
                'use_hic_targets': config.use_hic_targets,
                'test_crit_acc': test_crit_acc,
                'test_macro_f1': test_macro_f1,
                'test_cascade_mae': test_cascade_mae,
                'convergence_epoch': convergence_epoch,
                'best_val_acc': best_val_acc,
                'final_train_loss': train_losses[-1] if train_losses else 0.0,
                'final_val_loss': val_losses[-1] if val_losses else 0.0
            }
            results.append(result)

            print(f"   ✓ Final: Crit Acc={test_crit_acc:.3f}, Macro F1={test_macro_f1:.3f}, Cascade MAE={test_cascade_mae:.3f}")
            print()

        except Exception as e:
            logger.error(f"Error training config {config.name}: {e}")
            print(f"   ✗ Training failed: {e}")
            print()
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = output_path / f"moo_ablation_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)

        # Print formatted results table
        print_ablation_results(results_df)

        print(f"\n💾 Results saved to: {results_path}")
    else:
        print("❌ No successful training runs completed")

    return results_df


def print_ablation_results(results_df: pd.DataFrame):
    """Print formatted ablation results table."""
    print("="*100)
    print("MOO ABLATION STUDY RESULTS")
    print("="*100)

    if results_df.empty:
        print("No results to display")
        return

    # Format table
    print(f"{'Configuration':<15} {'Features':<8} {'Loss Wts':<8} {'HIC':<5} {'Crit Acc':<9} {'Macro F1':<9} {'Casc MAE':<9} {'Conv Epoch':<10}")
    print("-" * 100)

    for _, row in results_df.iterrows():
        features_mark = "✓" if row['use_moo_features'] else "✗"
        loss_mark = "✓" if row['use_moo_loss_weights'] else "✗"
        hic_mark = "✓" if row['use_hic_targets'] else "✗"

        print(f"{row['config_name']:<15} {features_mark:<8} {loss_mark:<8} {hic_mark:<5} "
              f"{row['test_crit_acc']:<9.3f} {row['test_macro_f1']:<9.3f} {row['test_cascade_mae']:<9.3f} {row['convergence_epoch']:<10d}")

    print("="*100)

    # Analysis
    if len(results_df) >= 2:
        full_idx = results_df[results_df['config_name'] == 'Full MOO'].index
        none_idx = results_df[results_df['config_name'] == 'Pure Neural'].index

        if len(full_idx) > 0 and len(none_idx) > 0:
            full_acc = results_df.loc[full_idx[0], 'test_crit_acc']
            none_acc = results_df.loc[none_idx[0], 'test_crit_acc']
            improvement = full_acc - none_acc

            print(f"\n📈 MOO IMPACT ANALYSIS:")
            print(f"   Full MOO vs Pure Neural: {improvement:+.3f} accuracy improvement")

            if improvement > 0.05:
                print(f"   🎯 Strong positive impact (+{improvement:.3f}) - MOO mechanisms are highly effective")
            elif improvement > 0.02:
                print(f"   📊 Moderate positive impact (+{improvement:.3f}) - MOO mechanisms show clear benefit")
            elif improvement > 0:
                print(f"   📈 Slight positive impact (+{improvement:.3f}) - MOO mechanisms provide modest benefit")
            else:
                print(f"   ⚠️  No clear benefit ({improvement:+.3f}) - Consider MOO mechanism refinement")


# ============================================================================
# Helper Functions (Placeholder implementations)
# ============================================================================

def compute_multi_task_loss(output, labels, cascade_targets, loss_weights):
    """Placeholder multi-task loss computation."""
    # Replace with actual loss computation
    dummy_loss = torch.tensor(1.0, requires_grad=True)
    return dummy_loss

def compute_accuracy(output, labels):
    """Placeholder accuracy computation."""
    # Replace with actual accuracy computation
    return np.random.uniform(0.6, 0.95)

def compute_f1_score(output, labels):
    """Placeholder F1 score computation."""
    # Replace with actual F1 computation
    return np.random.uniform(0.5, 0.9)

def compute_cascade_mae(output, targets):
    """Placeholder cascade MAE computation."""
    # Replace with actual MAE computation
    return np.random.uniform(0.1, 0.5)


# ============================================================================
# Main execution for testing
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MOO ABLATION MODULE - HT-HGNN")
    print("="*80)

    # Display all configurations
    print("📋 Available Configurations:")
    for i, config in enumerate(ALL_CONFIGS, 1):
        print(f"\n{i}. {config.name}")
        print(f"   {config.description}")
        print(f"   Mechanisms: {', '.join(config.enabled_mechanisms) if config.enabled_mechanisms else 'None'}")

    print(f"\n🎯 This module enables systematic ablation study to isolate the")
    print(f"   contribution of each MOO transfer mechanism in HT-HGNN:")
    print(f"   A. Feature-level: MOO embedding appended to node features")
    print(f"   B. Loss weights: MOO-optimized task loss balancing")
    print(f"   C. HIC targets: MOO-calibrated cascade simulation targets")

    print(f"\n🚀 Usage:")
    print(f"   from moo_ablation import run_moo_ablation_study, CONFIG_FULL")
    print(f"   results = run_moo_ablation_study(train, val, test, HTHGNNModel)")

    # Test feature building
    print(f"\n🧪 Testing feature building:")
    raw_features = torch.randn(10, 12)  # 10 nodes, 12-dim features
    moo_embedding = torch.randn(10, 4)   # 10 nodes, 4-dim MOO embedding

    for config in ALL_CONFIGS:
        enhanced = build_node_features(raw_features, moo_embedding, config)
        print(f"   {config.name}: {raw_features.shape} → {enhanced.shape}")

    print(f"\n✅ MOO ablation module ready!")
    print("="*80)