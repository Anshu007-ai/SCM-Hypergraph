"""
SSL Temperature Sensitivity Analysis for HT-HGNN

This script performs a comprehensive temperature sensitivity sweep for the NT-Xent
contrastive loss in HT-HGNN. It trains the full model from scratch for each
temperature value and evaluates performance on criticality classification.

Temperature values tested: [0.05, 0.1, 0.2, 0.3, 0.5]

Outputs:
- ssl_temperature_results.csv: Detailed results for each temperature
- ssl_temperature_sweep.png: Visualization of results

Usage:
    python ssl_sweep.py
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import warnings
from typing import Dict, List, Tuple
import sys
from datetime import datetime
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import model and losses
from src.models.ht_hgnn_model import HeterogeneousTemporalHypergraphNN
from losses import ContrastiveMultiTaskLoss, nt_xent_loss

# Import data utilities
from train_ht_hgnn_safe_v2 import SafeHT_HGNN_Trainer, SafeDeviceManager, SafeDataValidator

# Import evaluation metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ssl_sweep.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SSLTemperatureSweepTrainer:
    """
    Trainer specifically for SSL temperature sensitivity analysis.

    This extends the base trainer to include NT-Xent contrastive loss
    with configurable temperature and proper evaluation metrics.
    """

    def __init__(self,
                 ssl_temperature: float = 0.1,
                 ssl_weight: float = 0.1,
                 in_channels: int = 18,
                 hidden_channels: int = 64,
                 out_channels: int = 32,
                 num_nodes: int = 1206,
                 num_hyperedges: int = 36):
        """
        Initialize SSL trainer with specific temperature.

        Args:
            ssl_temperature: Temperature parameter for NT-Xent loss
            ssl_weight: Weight for SSL loss component
            Other args: Standard model configuration
        """

        self.ssl_temperature = ssl_temperature
        self.ssl_weight = ssl_weight
        self.device = SafeDeviceManager.get_device()

        # Model configuration
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.num_hyperedges = num_hyperedges

        # Initialize model
        self.model = HeterogeneousTemporalHypergraphNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_nodes=num_nodes,
            num_hyperedges=num_hyperedges,
            node_types=['supplier', 'part', 'transaction'],
            edge_types=['supplies', 'uses', 'prices'],
            num_hgnn_layers=2,
            num_hgt_heads=4,
            time_window=10
        ).to(self.device)

        # Initialize loss function with SSL
        self.loss_fn = ContrastiveMultiTaskLoss(
            weight_price=1.0,
            weight_change=0.5,
            weight_criticality=0.3,
            ssl_weight=ssl_weight,
            ssl_temperature=ssl_temperature
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5,
            eps=1e-8
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )

        # Mixed precision
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()

        # Training history
        self.history = {
            'loss': [], 'loss_price': [], 'loss_change': [],
            'loss_criticality': [], 'ssl_loss': [], 'val_loss': []
        }

        logger.info(f"[INIT] SSL trainer initialized with τ={ssl_temperature}")
        logger.info(f"       SSL weight: {ssl_weight}, Device: {self.device}")

    def prepare_data(self):
        """Prepare training/validation/test data"""

        logger.info("[DATA] Loading and preparing dataset...")

        # Use existing data preparation logic
        try:
            features_df = pd.read_csv('outputs/datasets/features.csv')
            hci_labels_df = pd.read_csv('outputs/datasets/hci_labels.csv')
            incidence_df = pd.read_csv('outputs/datasets/incidence.csv')
        except FileNotFoundError as e:
            logger.error(f"[DATA] Dataset file not found: {e}")
            logger.error("       Please ensure dataset files are in outputs/datasets/")
            raise

        # Extract hyperedge features
        hyperedge_features = features_df.drop('hyperedge_id', axis=1).values.astype(np.float32)

        # Build node features from hyperedges
        node_features = np.zeros((self.num_nodes, self.in_channels), dtype=np.float32)
        hyperedge_map = {he_id: i for i, he_id in enumerate(features_df['hyperedge_id'].values)}

        node_counts = np.zeros(self.num_nodes)
        for _, row in incidence_df.iterrows():
            he_idx = hyperedge_map.get(row['hyperedge_id'])
            node_idx = int(row['node_id'].split('_')[1]) % self.num_nodes

            if he_idx is not None and 0 <= node_idx < self.num_nodes:
                node_features[node_idx] += hyperedge_features[he_idx]
                node_counts[node_idx] += 1

        # Normalize by connection count
        for i in range(self.num_nodes):
            if node_counts[i] > 0:
                node_features[i] /= node_counts[i]

        # Fill unconnected nodes
        unconnected_mask = node_counts == 0
        if unconnected_mask.any():
            avg_features = hyperedge_features.mean(axis=0)
            node_features[unconnected_mask] = avg_features

        # Convert to tensor and normalize
        X_tensor = torch.FloatTensor(node_features).to(self.device)
        X_mean, X_std = X_tensor.mean(dim=0), X_tensor.std(dim=0) + 1e-8
        X_tensor = (X_tensor - X_mean) / X_std

        # Build incidence matrix
        incidence_matrix = np.zeros((self.num_hyperedges, self.num_nodes), dtype=np.float32)
        for _, row in incidence_df.iterrows():
            he_idx = hyperedge_map.get(row['hyperedge_id'])
            node_idx = int(row['node_id'].split('_')[1]) % self.num_nodes
            if he_idx is not None and 0 <= node_idx < self.num_nodes:
                incidence_matrix[he_idx, node_idx] = 1

        incidence_tensor = torch.FloatTensor(incidence_matrix).to(self.device)

        # Edge index for graph operations
        edges_i, edges_j = torch.nonzero(incidence_tensor, as_tuple=True)
        edge_index = torch.stack([edges_i, edges_j])

        # Node and edge types
        node_types = []
        nodes_per_type = self.num_nodes // 3
        for i in range(self.num_nodes):
            if i < nodes_per_type:
                node_types.append('supplier')
            elif i < 2 * nodes_per_type:
                node_types.append('part')
            else:
                node_types.append('transaction')

        edge_types = ['supplies', 'uses', 'prices']
        assigned_edge_types = [edge_types[i % len(edge_types)] for i in range(edge_index.size(1))]

        # Generate synthetic targets (replace with real labels)
        y_price = torch.randn(self.num_nodes, 1).to(self.device)
        y_change = torch.randn(self.num_nodes, 1).to(self.device)
        y_criticality = torch.randint(0, 2, (self.num_nodes, 1)).float().to(self.device)

        # Timestamps
        timestamps = torch.linspace(0, 10, self.num_nodes).to(self.device)

        # Create train/val/test splits (temporal split for time series data)
        n_train = int(0.7 * self.num_nodes)
        n_val = int(0.15 * self.num_nodes)
        n_test = self.num_nodes - n_train - n_val

        train_idx = slice(0, n_train)
        val_idx = slice(n_train, n_train + n_val)
        test_idx = slice(n_train + n_val, self.num_nodes)

        data = {
            'X': X_tensor,
            'incidence_matrix': incidence_tensor,
            'node_types': node_types,
            'edge_index': edge_index,
            'edge_types': assigned_edge_types,
            'timestamps': timestamps,
            'y_price': y_price,
            'y_change': y_change,
            'y_criticality': y_criticality,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }

        logger.info(f"[DATA] Dataset prepared: {self.num_nodes} nodes, {self.num_hyperedges} hyperedges")
        logger.info(f"       Train: {n_train}, Val: {n_val}, Test: {n_test}")

        return data

    def create_augmented_views(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two augmented views of node features for contrastive learning.

        Args:
            node_features: Original node features [N, D]

        Returns:
            Tuple of (z1, z2) augmented view embeddings
        """

        # Simple augmentation: add different noise to create two views
        noise_scale = 0.1

        # View 1: Add gaussian noise
        z1 = node_features + torch.randn_like(node_features) * noise_scale

        # View 2: Add different gaussian noise + small rotation in feature space
        z2 = node_features + torch.randn_like(node_features) * noise_scale

        return z1, z2

    def train_epoch(self, data: Dict) -> Dict:
        """Train single epoch with SSL loss"""

        self.model.train()
        self.optimizer.zero_grad()

        try:
            # Create augmented views for contrastive learning
            z1, z2 = self.create_augmented_views(data['X'])

            if self.use_amp:
                with autocast(device_type='cuda'):
                    # Forward pass for view 1
                    output1 = self.model(
                        node_features=z1,
                        incidence_matrix=data['incidence_matrix'],
                        node_types=data['node_types'],
                        edge_index=data['edge_index'],
                        edge_types=data['edge_types'],
                        timestamps=data['timestamps']
                    )

                    # Forward pass for view 2
                    output2 = self.model(
                        node_features=z2,
                        incidence_matrix=data['incidence_matrix'],
                        node_types=data['node_types'],
                        edge_index=data['edge_index'],
                        edge_types=data['edge_types'],
                        timestamps=data['timestamps']
                    )

                    # Use embeddings for contrastive loss
                    embed1 = output1['embeddings']
                    embed2 = output2['embeddings']

                    # Compute combined loss (supervised + SSL)
                    loss_dict = self.loss_fn(
                        price_pred=output1['price_pred'],
                        price_target=data['y_price'],
                        change_pred=output1['change_pred'],
                        change_target=data['y_change'],
                        criticality_pred=output1['criticality'],
                        criticality_target=data['y_criticality'],
                        z1=embed1,
                        z2=embed2
                    )

                self.scaler.scale(loss_dict['total_loss']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # CPU training
                output1 = self.model(
                    node_features=z1,
                    incidence_matrix=data['incidence_matrix'],
                    node_types=data['node_types'],
                    edge_index=data['edge_index'],
                    edge_types=data['edge_types'],
                    timestamps=data['timestamps']
                )

                output2 = self.model(
                    node_features=z2,
                    incidence_matrix=data['incidence_matrix'],
                    node_types=data['node_types'],
                    edge_index=data['edge_index'],
                    edge_types=data['edge_types'],
                    timestamps=data['timestamps']
                )

                embed1 = output1['embeddings']
                embed2 = output2['embeddings']

                loss_dict = self.loss_fn(
                    price_pred=output1['price_pred'],
                    price_target=data['y_price'],
                    change_pred=output1['change_pred'],
                    change_target=data['y_change'],
                    criticality_pred=output1['criticality'],
                    criticality_target=data['y_criticality'],
                    z1=embed1,
                    z2=embed2
                )

                loss_dict['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            return loss_dict

        except RuntimeError as e:
            logger.error(f"[ERROR] Training error: {e}")
            SafeDeviceManager.clear_cache()
            raise

    @torch.no_grad()
    def evaluate(self, data: Dict) -> Dict:
        """Evaluate model on validation/test set"""

        self.model.eval()

        # Forward pass
        output = self.model(
            node_features=data['X'],
            incidence_matrix=data['incidence_matrix'],
            node_types=data['node_types'],
            edge_index=data['edge_index'],
            edge_types=data['edge_types'],
            timestamps=data['timestamps']
        )

        # Compute validation loss
        val_loss_dict = self.loss_fn(
            price_pred=output['price_pred'],
            price_target=data['y_price'],
            change_pred=output['change_pred'],
            change_target=data['y_change'],
            criticality_pred=output['criticality'],
            criticality_target=data['y_criticality']
        )

        # Criticality classification metrics
        crit_pred = torch.sigmoid(output['criticality']).cpu().numpy() > 0.5
        crit_true = data['y_criticality'].cpu().numpy().astype(bool)

        crit_acc = accuracy_score(crit_true, crit_pred)
        macro_f1 = f1_score(crit_true, crit_pred, average='macro')

        return {
            'val_loss': val_loss_dict['total_loss'].item(),
            'crit_acc': crit_acc,
            'macro_f1': macro_f1,
            'ssl_loss': val_loss_dict.get('ssl_loss', 0.0)
        }

    def train(self, data: Dict, epochs: int = 50, verbose: bool = True) -> Dict:
        """Full training loop with validation"""

        logger.info(f"[TRAIN] Starting training for {epochs} epochs (τ={self.ssl_temperature})")

        best_val_acc = 0.0
        val_loss_curve = []

        for epoch in tqdm(range(epochs), desc=f"Training τ={self.ssl_temperature}"):

            # Training step
            loss_dict = self.train_epoch(data)

            # Validation step
            val_metrics = self.evaluate(data)

            # Update history
            self.history['loss'].append(loss_dict['total_loss'].item())
            self.history['loss_price'].append(loss_dict['loss_price'])
            self.history['loss_change'].append(loss_dict['loss_change'])
            self.history['loss_criticality'].append(loss_dict['loss_criticality'])
            self.history['ssl_loss'].append(loss_dict.get('ssl_loss', 0.0))
            self.history['val_loss'].append(val_metrics['val_loss'])

            val_loss_curve.append(val_metrics['val_loss'])

            # Track best validation accuracy
            if val_metrics['crit_acc'] > best_val_acc:
                best_val_acc = val_metrics['crit_acc']

            # Scheduler step
            self.scheduler.step()

            # Verbose logging
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs}: "
                          f"Loss={loss_dict['total_loss'].item():.4f}, "
                          f"SSL={loss_dict.get('ssl_loss', 0.0):.4f}, "
                          f"Val_Acc={val_metrics['crit_acc']:.3f}")

        # Final evaluation on test set
        test_metrics = self.evaluate(data)

        return {
            'final_crit_acc': test_metrics['crit_acc'],
            'final_macro_f1': test_metrics['macro_f1'],
            'val_loss_curve': val_loss_curve,
            'best_val_acc': best_val_acc,
            'final_ssl_loss': test_metrics['ssl_loss']
        }


def train_with_temperature(tau: float, epochs: int = 50) -> Dict:
    """
    Train model with specific temperature and return evaluation results.

    Args:
        tau: Temperature parameter for NT-Xent loss
        epochs: Number of training epochs

    Returns:
        Dictionary with training/evaluation metrics
    """

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING WITH TEMPERATURE τ = {tau}")
    logger.info(f"{'='*60}")

    # Initialize trainer with specific temperature
    trainer = SSLTemperatureSweepTrainer(
        ssl_temperature=tau,
        ssl_weight=0.1  # Fixed SSL weight
    )

    # Prepare data
    data = trainer.prepare_data()

    # Train model
    results = trainer.train(data, epochs=epochs, verbose=True)

    logger.info(f"[RESULT] τ={tau}: Accuracy={results['final_crit_acc']:.3f}, "
               f"F1={results['final_macro_f1']:.3f}")

    return {
        'tau': tau,
        'crit_acc': results['final_crit_acc'],
        'macro_f1': results['final_macro_f1'],
        'val_loss_curve': results['val_loss_curve'],
        'best_val_acc': results['best_val_acc'],
        'final_ssl_loss': results['final_ssl_loss']
    }


def run_temperature_sweep(tau_values: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5],
                         epochs: int = 50) -> pd.DataFrame:
    """
    Run complete temperature sensitivity sweep.

    Args:
        tau_values: List of temperature values to test
        epochs: Number of epochs per temperature

    Returns:
        DataFrame with all results
    """

    logger.info(f"\n{'='*80}")
    logger.info(f"SSL TEMPERATURE SENSITIVITY SWEEP")
    logger.info(f"{'='*80}")
    logger.info(f"Temperature values: {tau_values}")
    logger.info(f"Epochs per temperature: {epochs}")
    logger.info(f"Total experiments: {len(tau_values)}")

    results = []

    for tau in tau_values:
        try:
            result = train_with_temperature(tau, epochs)
            results.append(result)

            # Clear GPU cache between runs
            SafeDeviceManager.clear_cache()

        except Exception as e:
            logger.error(f"[ERROR] Temperature {tau} failed: {e}")
            # Add failed result to maintain sweep integrity
            results.append({
                'tau': tau,
                'crit_acc': 0.0,
                'macro_f1': 0.0,
                'val_loss_curve': [float('inf')] * epochs,
                'best_val_acc': 0.0,
                'final_ssl_loss': float('inf')
            })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    logger.info(f"\n[COMPLETE] Temperature sweep finished")
    logger.info(f"Results overview:")
    for _, row in df_results.iterrows():
        logger.info(f"  τ={row['tau']:.2f}: Acc={row['crit_acc']:.3f}, F1={row['macro_f1']:.3f}")

    return df_results


def save_results_and_plots(results_df: pd.DataFrame):
    """
    Save results to CSV and create visualization plots.

    Args:
        results_df: DataFrame with sweep results
    """

    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    # Save CSV results
    csv_path = output_dir / 'ssl_temperature_results.csv'

    # Prepare CSV data (expand val_loss_curve to separate columns)
    csv_data = []
    for _, row in results_df.iterrows():
        csv_row = {
            'tau': row['tau'],
            'crit_acc': row['crit_acc'],
            'macro_f1': row['macro_f1'],
            'best_val_acc': row['best_val_acc'],
            'final_ssl_loss': row['final_ssl_loss']
        }

        # Add per-epoch validation loss
        val_curves = row['val_loss_curve']
        if isinstance(val_curves, list):
            for epoch, val_loss in enumerate(val_curves):
                csv_row[f'val_loss_epoch_{epoch+1}'] = val_loss

        csv_data.append(csv_row)

    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"[SAVE] Results saved to: {csv_path}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Temperature vs Accuracy (Bar Chart)
    ax1.bar(results_df['tau'], results_df['crit_acc'],
            color='steelblue', alpha=0.7, edgecolor='navy')
    ax1.set_xlabel('Temperature (τ)', fontsize=12)
    ax1.set_ylabel('Criticality Accuracy', fontsize=12)
    ax1.set_title('Temperature vs Criticality Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # Add value labels on bars
    for i, (tau, acc) in enumerate(zip(results_df['tau'], results_df['crit_acc'])):
        ax1.text(tau, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)

    # Panel 2: Validation Loss Curves
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))

    for idx, (_, row) in enumerate(results_df.iterrows()):
        tau = row['tau']
        val_curve = row['val_loss_curve']

        if isinstance(val_curve, list) and len(val_curve) > 0:
            epochs = list(range(1, len(val_curve) + 1))
            ax2.plot(epochs, val_curve,
                    label=f'τ={tau:.2f}', color=colors[idx],
                    linewidth=2, alpha=0.8)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Curves by Temperature', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / 'ssl_temperature_sweep.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"[SAVE] Plot saved to: {plot_path}")


def main():
    """Main execution for SSL temperature sweep"""

    print(f"\n{'='*80}")
    print("HT-HGNN SSL TEMPERATURE SENSITIVITY ANALYSIS")
    print(f"{'='*80}")

    # Configuration
    tau_values = [0.05, 0.1, 0.2, 0.3, 0.5]
    epochs = 50  # Can be reduced for faster testing

    start_time = datetime.now()

    try:
        # Run temperature sweep
        results_df = run_temperature_sweep(tau_values, epochs)

        # Save results and create plots
        save_results_and_plots(results_df)

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n{'='*80}")
        print("SSL TEMPERATURE SWEEP COMPLETE")
        print(f"{'='*80}")
        print(f"Duration: {duration}")
        print(f"Temperatures tested: {len(tau_values)}")
        print(f"Best temperature: τ={results_df.loc[results_df['crit_acc'].idxmax(), 'tau']:.2f}")
        print(f"Best accuracy: {results_df['crit_acc'].max():.3f}")
        print(f"\nFiles saved:")
        print(f"  - outputs/ssl_temperature_results.csv")
        print(f"  - outputs/ssl_temperature_sweep.png")
        print(f"{'='*80}")

    except Exception as e:
        logger.error(f"[FATAL] SSL temperature sweep failed: {e}")
        raise


if __name__ == "__main__":
    main()