"""
Transfer Learning Experiment for HT-HGNN

This experiment tests whether pretraining on structurally analogous BOM graphs
provides better initialization for IndiGo aviation fine-tuning compared to:
A) BOM pretraining (analogous source - our hypothesis)
B) M5 retail pretraining (non-analogous source)
C) No pretraining (Xavier initialization only)

The experiment uses identical model architectures with controlled random seeds
to isolate the effect of pretraining source on fine-tuning performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Tuple, Any, Optional
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import warnings
import copy
from datetime import datetime
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import model and utilities
from src.models.ht_hgnn_model import HeterogeneousTemporalHypergraphNN
from losses import ContrastiveMultiTaskLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transfer_learning.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility for fine-tuning."""

    def __init__(self, patience: int = 15, mode: str = 'max', min_delta: float = 0.001):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


def apply_xavier_init(model: nn.Module) -> None:
    """
    Apply Xavier uniform initialization to all Linear layers and zero bias.

    Args:
        model: The neural network model to initialize
    """
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    logger.info("[INIT] Applied Xavier uniform initialization to all Linear layers")


def create_synthetic_dataset(dataset_type: str, num_samples: int = 1000,
                           num_nodes: int = 1206, num_hyperedges: int = 36) -> Dict[str, torch.Tensor]:
    """
    Create synthetic datasets for pretraining experiments.

    Args:
        dataset_type: 'bom', 'm5', or 'indigo'
        num_samples: Number of samples to generate
        num_nodes: Number of nodes in hypergraph
        num_hyperedges: Number of hyperedges

    Returns:
        Dictionary with synthetic dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate base features with dataset-specific characteristics
    if dataset_type == 'bom':
        # BOM: hierarchical structure, discrete part relationships
        node_features = torch.randn(num_samples, num_nodes, 18, device=device) * 0.8
        # Add hierarchical patterns
        for i in range(0, num_nodes, 100):
            node_features[:, i:i+50, :5] += 1.0  # Supplier tier patterns

    elif dataset_type == 'm5':
        # M5 Retail: seasonal patterns, different structure
        node_features = torch.randn(num_samples, num_nodes, 18, device=device) * 1.2
        # Add seasonal/retail patterns
        seasonal_pattern = torch.sin(torch.arange(num_nodes, device=device).float() / 100)
        node_features[:, :, 0] += seasonal_pattern.unsqueeze(0).repeat(num_samples, 1)

    else:  # indigo
        # IndiGo: aviation-specific patterns
        node_features = torch.randn(num_samples, num_nodes, 18, device=device)
        # Add aviation-specific correlations
        node_features[:, :200, 10:15] *= 1.5  # Aircraft nodes have higher variance

    # Generate incidence matrix (same topology structure for fair comparison)
    incidence_matrix = torch.zeros(num_hyperedges, num_nodes, device=device)
    for he in range(num_hyperedges):
        size = torch.randint(5, 15, (1,)).item()
        members = torch.randperm(num_nodes)[:size]
        incidence_matrix[he, members] = 1.0

    # Generate targets based on dataset type
    if dataset_type == 'bom':
        # BOM targets: cost-focused
        y_price = torch.log1p(torch.abs(torch.randn(num_samples, num_nodes, device=device)))
        y_change = torch.randn(num_samples, num_nodes, device=device) * 0.1
        y_criticality = (torch.randn(num_samples, num_nodes, device=device) > 0).float()

    elif dataset_type == 'm5':
        # M5 targets: sales-focused
        y_price = torch.abs(torch.randn(num_samples, num_nodes, device=device)) * 2
        y_change = torch.randn(num_samples, num_nodes, device=device) * 0.3
        y_criticality = (torch.randn(num_samples, num_nodes, device=device) > 0.3).float()

    else:  # indigo
        # IndiGo targets: disruption-focused
        y_price = torch.abs(torch.randn(num_samples, num_nodes, device=device))
        y_change = torch.randn(num_samples, num_nodes, device=device) * 0.2
        y_criticality = (torch.randn(num_samples, num_nodes, device=device) > -0.2).float()

    # Node metadata
    nodes_per_type = num_nodes // 3
    node_types = (['supplier'] * nodes_per_type +
                 ['part'] * nodes_per_type +
                 ['transaction'] * (num_nodes - 2 * nodes_per_type))

    # Edge metadata
    num_edges = min(8000, num_nodes * 8)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_types = ['supplies', 'uses', 'prices'] * (num_edges // 3 + 1)
    edge_types = edge_types[:num_edges]

    # Timestamps
    timestamps = torch.linspace(0, 10, num_nodes, device=device)

    return {
        'node_features': node_features,
        'incidence_matrix': incidence_matrix,
        'node_types': node_types,
        'edge_index': edge_index,
        'edge_types': edge_types,
        'timestamps': timestamps,
        'y_price': y_price,
        'y_change': y_change,
        'y_criticality': y_criticality
    }


def pretrain(model: nn.Module, dataset_name: str, dataset: Dict[str, torch.Tensor],
            epochs: int = 50, lr: float = 0.001) -> nn.Module:
    """
    Pretrain the model on the given dataset.

    Args:
        model: HT-HGNN model to pretrain
        dataset_name: String label for logging
        dataset: Pretraining dataset
        epochs: Number of pretraining epochs
        lr: Learning rate

    Returns:
        Pretrained model
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PRETRAINING ON {dataset_name.upper()} DATASET")
    logger.info(f"{'='*60}")

    device = next(model.parameters()).device
    model.train()

    # Loss function and optimizer
    criterion = ContrastiveMultiTaskLoss(
        weight_price=1.0,
        weight_change=0.5,
        weight_criticality=0.3,
        ssl_weight=0.0,  # Disable SSL for pretraining
        ssl_temperature=0.1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    pretraining_losses = []
    best_loss = float('inf')

    logger.info(f"Starting pretraining for {epochs} epochs...")

    for epoch in tqdm(range(epochs), desc=f"Pretraining on {dataset_name}"):
        epoch_losses = []

        # Simple batch processing (use first sample for demonstration)
        batch_idx = epoch % dataset['node_features'].shape[0]

        optimizer.zero_grad()

        # Get batch data
        batch_features = dataset['node_features'][batch_idx:batch_idx+1].squeeze(0)
        batch_y_price = dataset['y_price'][batch_idx:batch_idx+1].squeeze(0).unsqueeze(-1)
        batch_y_change = dataset['y_change'][batch_idx:batch_idx+1].squeeze(0).unsqueeze(-1)
        batch_y_crit = dataset['y_criticality'][batch_idx:batch_idx+1].squeeze(0)

        # Forward pass
        output = model(
            node_features=batch_features,
            incidence_matrix=dataset['incidence_matrix'],
            node_types=dataset['node_types'],
            edge_index=dataset['edge_index'],
            edge_types=dataset['edge_types'],
            timestamps=dataset['timestamps']
        )

        # Compute loss
        loss_dict = criterion(
            price_pred=output['price_pred'],
            price_target=batch_y_price,
            change_pred=output['change_pred'],
            change_target=batch_y_change,
            criticality_pred=output['criticality'],
            criticality_target=batch_y_crit
        )

        # Backward pass
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss = loss_dict['total_loss'].item()
        epoch_losses.append(epoch_loss)

        # Update scheduler
        scheduler.step(epoch_loss)

        # Track best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss

        # Logging
        if (epoch + 1) % 10 == 0:
            avg_loss = np.mean(epoch_losses[-10:])
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, LR={current_lr:.6f}")

        pretraining_losses.append(epoch_loss)

    logger.info(f"Pretraining complete - Best loss: {best_loss:.4f}")
    logger.info(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    return model


def finetune(model: nn.Module, indigo_train: Dict[str, torch.Tensor],
            indigo_val: Dict[str, torch.Tensor], epochs: int = 100,
            lr: float = 0.0001) -> Tuple[nn.Module, int, Dict[str, list]]:
    """
    Fine-tune the model on IndiGo dataset with early stopping.

    Args:
        model: Pretrained (or initialized) model
        indigo_train: IndiGo training data
        indigo_val: IndiGo validation data
        epochs: Maximum fine-tuning epochs
        lr: Fine-tuning learning rate

    Returns:
        Tuple of (fine-tuned model, convergence_epoch, training_history)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FINE-TUNING ON INDIGO DATASET")
    logger.info(f"{'='*60}")

    device = next(model.parameters()).device
    model.train()

    # Loss function and optimizer for fine-tuning
    criterion = ContrastiveMultiTaskLoss(
        weight_price=1.0,
        weight_change=0.5,
        weight_criticality=0.3,
        ssl_weight=0.0,  # Focus on supervised learning for fine-tuning
        ssl_temperature=0.1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    early_stopping = EarlyStopping(patience=15, mode='max', min_delta=0.001)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_accuracy': []
    }

    convergence_epoch = None
    best_f1 = 0.0

    logger.info(f"Starting fine-tuning for up to {epochs} epochs...")

    for epoch in tqdm(range(epochs), desc="Fine-tuning on IndiGo"):

        # Training phase
        model.train()
        train_losses = []

        # Simple batch processing for training
        batch_idx = epoch % indigo_train['node_features'].shape[0]

        optimizer.zero_grad()

        # Get training batch
        train_features = indigo_train['node_features'][batch_idx:batch_idx+1].squeeze(0)
        train_y_price = indigo_train['y_price'][batch_idx:batch_idx+1].squeeze(0).unsqueeze(-1)
        train_y_change = indigo_train['y_change'][batch_idx:batch_idx+1].squeeze(0).unsqueeze(-1)
        train_y_crit = indigo_train['y_criticality'][batch_idx:batch_idx+1].squeeze(0)

        # Forward pass
        train_output = model(
            node_features=train_features,
            incidence_matrix=indigo_train['incidence_matrix'],
            node_types=indigo_train['node_types'],
            edge_index=indigo_train['edge_index'],
            edge_types=indigo_train['edge_types'],
            timestamps=indigo_train['timestamps']
        )

        # Compute training loss
        train_loss_dict = criterion(
            price_pred=train_output['price_pred'],
            price_target=train_y_price,
            change_pred=train_output['change_pred'],
            change_target=train_y_change,
            criticality_pred=train_output['criticality'],
            criticality_target=train_y_crit
        )

        # Backward pass
        train_loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss = train_loss_dict['total_loss'].item()
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            # Get validation batch
            val_batch_idx = epoch % indigo_val['node_features'].shape[0]
            val_features = indigo_val['node_features'][val_batch_idx:val_batch_idx+1].squeeze(0)
            val_y_price = indigo_val['y_price'][val_batch_idx:val_batch_idx+1].squeeze(0).unsqueeze(-1)
            val_y_change = indigo_val['y_change'][val_batch_idx:val_batch_idx+1].squeeze(0).unsqueeze(-1)
            val_y_crit = indigo_val['y_criticality'][val_batch_idx:val_batch_idx+1].squeeze(0)

            # Forward pass
            val_output = model(
                node_features=val_features,
                incidence_matrix=indigo_val['incidence_matrix'],
                node_types=indigo_val['node_types'],
                edge_index=indigo_val['edge_index'],
                edge_types=indigo_val['edge_types'],
                timestamps=indigo_val['timestamps']
            )

            # Compute validation loss
            val_loss_dict = criterion(
                price_pred=val_output['price_pred'],
                price_target=val_y_price,
                change_pred=val_output['change_pred'],
                change_target=val_y_change,
                criticality_pred=val_output['criticality'],
                criticality_target=val_y_crit
            )

            val_loss = val_loss_dict['total_loss'].item()

            # Compute validation metrics
            val_crit_pred = torch.sigmoid(val_output['criticality']).squeeze() > 0.5
            val_crit_true = val_y_crit.bool()

            val_accuracy = accuracy_score(val_crit_true.cpu(), val_crit_pred.cpu())
            val_f1 = f1_score(val_crit_true.cpu(), val_crit_pred.cpu(), average='macro')

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_accuracy'].append(val_accuracy)

        # Check for convergence (first epoch with val_f1 > 0.85)
        if convergence_epoch is None and val_f1 > 0.85:
            convergence_epoch = epoch + 1
            logger.info(f"[CONVERGENCE] Reached val_f1 > 0.85 at epoch {convergence_epoch}")

        # Track best F1
        if val_f1 > best_f1:
            best_f1 = val_f1

        # Early stopping check
        if early_stopping(val_f1):
            logger.info(f"[EARLY STOP] Stopping at epoch {epoch+1} (val_f1={val_f1:.4f})")
            break

        # Periodic logging
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Train_Loss={train_loss:.4f}, "
                       f"Val_Loss={val_loss:.4f}, Val_F1={val_f1:.4f}, Val_Acc={val_accuracy:.4f}")

    # Set convergence epoch to total epochs if never reached 0.85
    if convergence_epoch is None:
        convergence_epoch = epochs
        logger.info(f"[NO CONVERGENCE] Never reached val_f1 > 0.85 (best: {best_f1:.4f})")

    logger.info(f"Fine-tuning complete - Best val_f1: {best_f1:.4f}")

    return model, convergence_epoch, history


def evaluate(model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Evaluate model on test data.

    Args:
        model: Trained model
        test_data: Test dataset

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Use first test sample
        test_features = test_data['node_features'][0]
        test_y_crit = test_data['y_criticality'][0]
        test_y_change = test_data['y_change'][0]

        # Forward pass
        output = model(
            node_features=test_features,
            incidence_matrix=test_data['incidence_matrix'],
            node_types=test_data['node_types'],
            edge_index=test_data['edge_index'],
            edge_types=test_data['edge_types'],
            timestamps=test_data['timestamps']
        )

        # Compute metrics
        crit_pred = torch.sigmoid(output['criticality']).squeeze() > 0.5
        crit_true = test_y_crit.bool()

        crit_acc = accuracy_score(crit_true.cpu(), crit_pred.cpu())
        macro_f1 = f1_score(crit_true.cpu(), crit_pred.cpu(), average='macro')

        # Cascade MAE (using change prediction as proxy)
        change_pred = output['change_pred'].squeeze()
        cascade_mae = mean_absolute_error(test_y_change.cpu(), change_pred.cpu())

    return {
        'crit_acc': crit_acc,
        'macro_f1': macro_f1,
        'cascade_mae': cascade_mae
    }


def run_transfer_experiment(bom_dataset: Dict[str, torch.Tensor],
                          m5_dataset: Dict[str, torch.Tensor],
                          indigo_train: Dict[str, torch.Tensor],
                          indigo_val: Dict[str, torch.Tensor],
                          indigo_test: Dict[str, torch.Tensor]) -> pd.DataFrame:
    """
    Run the complete transfer learning experiment.

    Args:
        bom_dataset: BOM pretraining dataset
        m5_dataset: M5 retail pretraining dataset
        indigo_train: IndiGo training data
        indigo_val: IndiGo validation data
        indigo_test: IndiGo test data

    Returns:
        DataFrame with experimental results
    """
    logger.info(f"\n{'='*80}")
    logger.info("TRANSFER LEARNING EXPERIMENT - HT-HGNN")
    logger.info(f"{'='*80}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Model configuration
    model_config = {
        'in_channels': 18,
        'hidden_channels': 64,
        'out_channels': 32,
        'num_nodes': 1206,
        'num_hyperedges': 36,
        'node_types': ['supplier', 'part', 'transaction'],
        'edge_types': ['supplies', 'uses', 'prices'],
        'num_hgnn_layers': 2,
        'num_hgt_heads': 4,
        'time_window': 10
    }

    results = []
    convergence_histories = {}

    # Condition A: BOM Pretraining (analogous source)
    logger.info(f"\n[CONDITION A] BOM Pretraining -> IndiGo Fine-tuning")
    torch.manual_seed(42)  # Reset seed
    model_a = HeterogeneousTemporalHypergraphNN(**model_config).to(device)

    # Pretrain on BOM
    model_a = pretrain(model_a, "BOM", bom_dataset, epochs=50, lr=0.001)

    # Fine-tune on IndiGo
    model_a, conv_a, hist_a = finetune(model_a, indigo_train, indigo_val, epochs=100, lr=0.0001)
    convergence_histories['BOM'] = hist_a

    # Evaluate
    metrics_a = evaluate(model_a, indigo_test)
    results.append({
        'pretraining_source': 'BOM',
        'crit_acc': metrics_a['crit_acc'],
        'macro_f1': metrics_a['macro_f1'],
        'cascade_mae': metrics_a['cascade_mae'],
        'convergence_epoch': conv_a
    })

    # Condition B: M5 Pretraining (non-analogous source)
    logger.info(f"\n[CONDITION B] M5 Retail Pretraining -> IndiGo Fine-tuning")
    torch.manual_seed(42)  # Reset seed
    model_b = HeterogeneousTemporalHypergraphNN(**model_config).to(device)

    # Pretrain on M5
    model_b = pretrain(model_b, "M5_Retail", m5_dataset, epochs=50, lr=0.001)

    # Fine-tune on IndiGo
    model_b, conv_b, hist_b = finetune(model_b, indigo_train, indigo_val, epochs=100, lr=0.0001)
    convergence_histories['M5_Retail'] = hist_b

    # Evaluate
    metrics_b = evaluate(model_b, indigo_test)
    results.append({
        'pretraining_source': 'M5_Retail',
        'crit_acc': metrics_b['crit_acc'],
        'macro_f1': metrics_b['macro_f1'],
        'cascade_mae': metrics_b['cascade_mae'],
        'convergence_epoch': conv_b
    })

    # Condition C: No pretraining (Xavier initialization)
    logger.info(f"\n[CONDITION C] No Pretraining (Xavier Init) -> IndiGo Fine-tuning")
    torch.manual_seed(42)  # Reset seed
    model_c = HeterogeneousTemporalHypergraphNN(**model_config).to(device)

    # Apply Xavier initialization
    apply_xavier_init(model_c)

    # Fine-tune on IndiGo (no pretraining)
    model_c, conv_c, hist_c = finetune(model_c, indigo_train, indigo_val, epochs=100, lr=0.0001)
    convergence_histories['Xavier_Init'] = hist_c

    # Evaluate
    metrics_c = evaluate(model_c, indigo_test)
    results.append({
        'pretraining_source': 'Xavier_Init',
        'crit_acc': metrics_c['crit_acc'],
        'macro_f1': metrics_c['macro_f1'],
        'cascade_mae': metrics_c['cascade_mae'],
        'convergence_epoch': conv_c
    })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Print comparison table
    print_results_table(results_df)

    # Plot convergence curves
    plot_convergence_curves(convergence_histories)

    # Save results
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    results_df.to_csv(output_dir / 'transfer_learning_results.csv', index=False)
    logger.info(f"Results saved to: {output_dir / 'transfer_learning_results.csv'}")

    return results_df


def print_results_table(results_df: pd.DataFrame):
    """Print formatted results comparison table."""

    logger.info(f"\n{'='*80}")
    logger.info("TRANSFER LEARNING RESULTS COMPARISON")
    logger.info(f"{'='*80}")

    print(f"\n{'Source':<15} {'Crit Acc':<10} {'Macro F1':<10} {'Cascade MAE':<12} {'Convergence':<12}")
    print("-" * 70)

    for _, row in results_df.iterrows():
        source = row['pretraining_source']
        crit_acc = row['crit_acc']
        macro_f1 = row['macro_f1']
        cascade_mae = row['cascade_mae']
        conv_epoch = row['convergence_epoch']

        print(f"{source:<15} {crit_acc:<10.4f} {macro_f1:<10.4f} {cascade_mae:<12.4f} {conv_epoch:<12d}")

    # Find best performing condition
    best_idx = results_df['macro_f1'].idxmax()
    best_condition = results_df.loc[best_idx]

    print(f"\n[BEST] {best_condition['pretraining_source']}: "
          f"F1={best_condition['macro_f1']:.4f}, "
          f"Convergence Epoch={best_condition['convergence_epoch']}")

    # Analyze BOM vs others
    bom_f1 = results_df[results_df['pretraining_source'] == 'BOM']['macro_f1'].iloc[0]
    m5_f1 = results_df[results_df['pretraining_source'] == 'M5_Retail']['macro_f1'].iloc[0]
    xavier_f1 = results_df[results_df['pretraining_source'] == 'Xavier_Init']['macro_f1'].iloc[0]

    print(f"\n[ANALYSIS]")
    print(f"  BOM vs M5 Retail: {bom_f1:.4f} vs {m5_f1:.4f} ({(bom_f1-m5_f1)*100:+.2f}% diff)")
    print(f"  BOM vs Xavier Init: {bom_f1:.4f} vs {xavier_f1:.4f} ({(bom_f1-xavier_f1)*100:+.2f}% diff)")

    if bom_f1 > max(m5_f1, xavier_f1):
        print(f"  [OK] BOM pretraining provides BEST transfer learning performance")
    else:
        print(f"  [FAIL] BOM pretraining does NOT provide best performance")

    logger.info(f"{'='*80}")


def plot_convergence_curves(histories: Dict[str, Dict[str, list]]):
    """Plot validation F1 convergence curves for all conditions."""

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    colors = {'BOM': 'blue', 'M5_Retail': 'orange', 'Xavier_Init': 'green'}

    for condition, history in histories.items():
        epochs = list(range(1, len(history['val_f1']) + 1))
        val_f1 = history['val_f1']

        ax.plot(epochs, val_f1,
               color=colors.get(condition, 'gray'),
               linewidth=2.5,
               label=f"{condition.replace('_', ' ')}")

        # Mark convergence point (val_f1 > 0.85)
        for i, f1 in enumerate(val_f1):
            if f1 > 0.85:
                ax.scatter(i+1, f1, color=colors.get(condition, 'gray'),
                          s=100, zorder=5, marker='o', edgecolor='white', linewidth=2)
                break

    # Add convergence threshold line
    ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='Convergence Threshold (0.85)')

    ax.set_xlabel('Fine-tuning Epoch', fontsize=12)
    ax.set_ylabel('Validation Macro F1 Score', fontsize=12)
    ax.set_title('Transfer Learning Convergence Comparison\n'
                 'Validation F1 vs Fine-tuning Epoch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    # Save plot
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'transfer_learning_convergence.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"Convergence plot saved to: {plot_path}")


def main():
    """Main execution for transfer learning experiment."""

    print(f"\n{'='*80}")
    print("HT-HGNN TRANSFER LEARNING EXPERIMENT")
    print(f"{'='*80}")

    try:
        # Create synthetic datasets
        logger.info("Creating synthetic datasets...")

        bom_dataset = create_synthetic_dataset('bom', num_samples=200)
        m5_dataset = create_synthetic_dataset('m5', num_samples=200)

        # IndiGo datasets (smaller for fine-tuning)
        indigo_train = create_synthetic_dataset('indigo', num_samples=100)
        indigo_val = create_synthetic_dataset('indigo', num_samples=50)
        indigo_test = create_synthetic_dataset('indigo', num_samples=50)

        logger.info("Datasets created successfully")
        logger.info(f"  BOM: {bom_dataset['node_features'].shape[0]} samples")
        logger.info(f"  M5: {m5_dataset['node_features'].shape[0]} samples")
        logger.info(f"  IndiGo Train: {indigo_train['node_features'].shape[0]} samples")
        logger.info(f"  IndiGo Val: {indigo_val['node_features'].shape[0]} samples")
        logger.info(f"  IndiGo Test: {indigo_test['node_features'].shape[0]} samples")

        # Run experiment
        results_df = run_transfer_experiment(
            bom_dataset=bom_dataset,
            m5_dataset=m5_dataset,
            indigo_train=indigo_train,
            indigo_val=indigo_val,
            indigo_test=indigo_test
        )

        print(f"\n{'='*80}")
        print("TRANSFER LEARNING EXPERIMENT COMPLETE!")
        print(f"{'='*80}")
        print(f"Results saved to:")
        print(f"  - outputs/transfer_learning_results.csv")
        print(f"  - outputs/transfer_learning_convergence.png")
        print(f"  - transfer_learning.log")
        print(f"{'='*80}")

    except Exception as e:
        logger.error(f"Transfer learning experiment failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()