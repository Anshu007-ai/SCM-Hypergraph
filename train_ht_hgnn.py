"""
Training script for Heterogeneous Temporal Hypergraph Neural Network

Demonstrates:
1. Model initialization
2. Data preparation
3. Multi-task learning training loop
4. Entropy and sensitivity analysis
5. Result interpretation
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.models.ht_hgnn_model import (
    HeterogeneousTemporalHypergraphNN,
    MultiTaskLoss,
    EntropyAnalyzer
)


class HT_HGNN_Trainer:
    """
    Trainer for HT-HGNN model
    
    Handles:
    - Data preparation
    - Model training
    - Entropy analysis
    - Result visualization
    """
    
    def __init__(self, 
                 in_channels: int = 18,
                 hidden_channels: int = 64,
                 out_channels: int = 32,
                 num_nodes: int = 1206,
                 num_hyperedges: int = 36,
                 device: str = 'cpu',
                 learning_rate: float = 0.001):
        
        self.device = torch.device(device)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.num_hyperedges = num_hyperedges
        
        print("\n" + "="*70)
        print("HETEROGENEOUS TEMPORAL HYPERGRAPH NEURAL NETWORK")
        print("="*70)
        print(f"\nModel Configuration:")
        print(f"  Input channels: {in_channels}")
        print(f"  Hidden channels: {hidden_channels}")
        print(f"  Output channels: {out_channels}")
        print(f"  Nodes: {num_nodes}")
        print(f"  Hyperedges: {num_hyperedges}")
        print(f"  Device: {device}")
        
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
        
        # Loss function
        self.loss_fn = MultiTaskLoss(
            weight_price=1.0,
            weight_change=0.5,
            weight_criticality=0.3
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        # Training history
        self.history = {
            'loss': [],
            'loss_price': [],
            'loss_change': [],
            'loss_criticality': []
        }
        
        print(f"\n✓ Model initialized with {self._count_parameters()} parameters")
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def prepare_data(self, 
                     features_df: pd.DataFrame,
                     hci_labels_df: pd.DataFrame,
                     incidence_csv: str = 'outputs/datasets/incidence.csv',
                     test_split: float = 0.2) -> Dict:
        """
        Prepare data for training
        
        Creates:
        - Node features tensor
        - Incidence matrix
        - Node/edge type assignments
        - Target labels
        """
        
        print("\n" + "-"*70)
        print("DATA PREPARATION")
        print("-"*70)
        
        # 1. Create node features (from aggregated hyperedge features)
        # features_df has 36 hyperedges, we need to expand to 1206 nodes
        # Each node gets features of its connected hyperedges
        hyperedge_features = features_df.drop('hyperedge_id', axis=1).values.astype(np.float32)  # (36, 18)
        
        # Load incidence to map nodes to hyperedges
        incidence_df = pd.read_csv(incidence_csv)
        
        # Build node features by aggregating connected hyperedges
        node_features = np.zeros((self.num_nodes, self.in_channels), dtype=np.float32)
        hyperedge_map = {he_id: i for i, he_id in enumerate(
            features_df['hyperedge_id'].values
        )}
        
        # Count connections per node
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
        
        # Fill unconnected nodes with average features
        unconnected_mask = node_counts == 0
        if unconnected_mask.any():
            avg_features = hyperedge_features.mean(axis=0)
            node_features[unconnected_mask] = avg_features
        
        X_tensor = torch.FloatTensor(node_features).to(self.device)
        
        # Normalize features
        X_mean = X_tensor.mean(dim=0)
        X_std = X_tensor.std(dim=0) + 1e-8
        X_tensor = (X_tensor - X_mean) / X_std
        
        print(f"✓ Node features: {X_tensor.shape}")
        
        # 2. Load incidence matrix
        incidence_matrix = np.zeros((self.num_hyperedges, self.num_nodes))
        
        # Build incidence using the hyperedge_map created above
        for _, row in incidence_df.iterrows():
            he_idx = hyperedge_map.get(row['hyperedge_id'])
            node_idx = int(row['node_id'].split('_')[1]) % self.num_nodes
            if he_idx is not None and 0 <= node_idx < self.num_nodes:
                incidence_matrix[he_idx, node_idx] = 1
        
        incidence_tensor = torch.FloatTensor(incidence_matrix).to(self.device)
        print(f"✓ Incidence matrix: {incidence_tensor.shape}")
        print(f"  Sparsity: {(incidence_tensor.sum() / incidence_tensor.numel() * 100):.1f}%")
        
        # 3. Create edge index for heterogeneous graph
        # Sample edges from incidence matrix
        edges_i, edges_j = torch.nonzero(incidence_tensor, as_tuple=True)
        edge_index = torch.stack([edges_i, edges_j]).to(self.device)
        
        # Assign edge types (cycling through: supplies, uses, prices)
        edge_types = ['supplies', 'uses', 'prices']
        assigned_edge_types = [
            edge_types[i % len(edge_types)] 
            for i in range(edge_index.size(1))
        ]
        
        print(f"✓ Edge index: {edge_index.shape}")
        
        # 4. Node types
        # Assign based on position (suppliers, parts, transactions)
        node_types = []
        nodes_per_type = self.num_nodes // 3
        for i in range(self.num_nodes):
            if i < nodes_per_type:
                node_types.append('supplier')
            elif i < 2 * nodes_per_type:
                node_types.append('part')
            else:
                node_types.append('transaction')
        
        print(f"✓ Node types: {len(set(node_types))} types")
        print(f"  Suppliers: {node_types.count('supplier')}")
        print(f"  Parts: {node_types.count('part')}")
        print(f"  Transactions: {node_types.count('transaction')}")
        
        # 5. Target labels
        y_price = np.random.normal(100, 20, self.num_nodes).astype(np.float32)
        y_change = np.random.uniform(-0.1, 0.1, self.num_nodes).astype(np.float32)
        y_criticality = (hci_labels_df['HCI'].values / 
                        hci_labels_df['HCI'].max()).astype(np.float32)
        
        # Pad to match num_nodes if needed
        if len(y_criticality) < self.num_nodes:
            y_criticality = np.pad(
                y_criticality,
                (0, self.num_nodes - len(y_criticality)),
                mode='edge'
            )
        
        y_price_tensor = torch.FloatTensor(y_price).to(self.device)
        y_change_tensor = torch.FloatTensor(y_change).to(self.device)
        y_criticality_tensor = torch.FloatTensor(
            y_criticality[:self.num_nodes]
        ).to(self.device)
        
        print(f"✓ Target labels:")
        print(f"  Price: {y_price_tensor.shape} (mean: {y_price_tensor.mean():.2f})")
        print(f"  Change: {y_change_tensor.shape} (mean: {y_change_tensor.mean():.4f})")
        print(f"  Criticality: {y_criticality_tensor.shape} (mean: {y_criticality_tensor.mean():.4f})")
        
        # 6. Timestamps
        timestamps = torch.linspace(0, 10, self.num_nodes).to(self.device)
        
        return {
            'X': X_tensor,
            'incidence_matrix': incidence_tensor,
            'node_types': node_types,
            'edge_index': edge_index,
            'edge_types': assigned_edge_types,
            'timestamps': timestamps,
            'y_price': y_price_tensor,
            'y_change': y_change_tensor,
            'y_criticality': y_criticality_tensor,
            'test_split': test_split
        }
    
    def train_epoch(self, data: Dict) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        # Forward pass
        output = self.model(
            node_features=data['X'],
            incidence_matrix=data['incidence_matrix'],
            node_types=data['node_types'],
            edge_index=data['edge_index'],
            edge_types=data['edge_types'],
            timestamps=data['timestamps']
        )
        
        # Compute loss
        loss_dict = self.loss_fn(
            price_pred=output['price_pred'],
            price_target=data['y_price'],
            change_pred=output['change_pred'],
            change_target=data['y_change'],
            criticality_pred=output['criticality'],
            criticality_target=data['y_criticality']
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss_dict
    
    def train(self, data: Dict, epochs: int = 50, verbose: bool = True) -> Dict:
        """
        Full training loop with entropy analysis
        """
        print("\n" + "-"*70)
        print("TRAINING")
        print("-"*70)
        
        for epoch in range(epochs):
            loss_dict = self.train_epoch(data)
            
            # Update history
            self.history['loss'].append(loss_dict['total_loss'].item())
            self.history['loss_price'].append(loss_dict['loss_price'])
            self.history['loss_change'].append(loss_dict['loss_change'])
            self.history['loss_criticality'].append(loss_dict['loss_criticality'])
            
            # Scheduler step
            self.scheduler.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  Total Loss: {loss_dict['total_loss'].item():.6f}")
                print(f"  Price Loss: {loss_dict['loss_price']:.6f}")
                print(f"  Change Loss: {loss_dict['loss_change']:.6f}")
                print(f"  Criticality Loss: {loss_dict['loss_criticality']:.6f}")
        
        print(f"\n✓ Training completed")
        print(f"  Final loss: {self.history['loss'][-1]:.6f}")
        
        return self.history
    
    def analyze_entropy_and_sensitivity(self, data: Dict) -> Dict:
        """
        Perform entropy and sensitivity analysis
        """
        print("\n" + "-"*70)
        print("ENTROPY AND SENSITIVITY ANALYSIS")
        print("-"*70)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                node_features=data['X'],
                incidence_matrix=data['incidence_matrix'],
                node_types=data['node_types'],
                edge_index=data['edge_index'],
                edge_types=data['edge_types'],
                timestamps=data['timestamps']
            )
        
        # Extract entropy metrics
        entropy = output['entropy'].item() if isinstance(output['entropy'], torch.Tensor) \
            else output['entropy']
        sensitivity = output['sensitivity']
        information_flow = output['information_flow']
        cascade_scores = output['cascade_scores']
        
        print(f"\n✓ Entropy Metrics:")
        print(f"  Information Flow Entropy: {entropy:.4f}")
        print(f"    (Higher = more distributed, Lower = more focused)")
        
        print(f"\n✓ Sensitivity Analysis:")
        print(f"  Mean sensitivity: {sensitivity.mean():.4f}")
        print(f"  Max sensitivity: {sensitivity.max():.4f}")
        print(f"  Std sensitivity: {sensitivity.std():.4f}")
        
        # Top sensitive nodes
        top_k = 10
        top_indices = torch.topk(sensitivity, k=min(top_k, len(sensitivity))).indices
        print(f"\n  Top {top_k} most sensitive nodes:")
        for i, idx in enumerate(top_indices):
            print(f"    {i+1}. Node {idx.item()}: {sensitivity[idx].item():.4f}")
        
        print(f"\n✓ Information Flow:")
        print(f"  Mean flow: {information_flow.mean():.4f}")
        print(f"  Max flow: {information_flow.max():.4f}")
        
        print(f"\n✓ Cascade Scores (Propagation Importance):")
        print(f"  Mean cascade: {cascade_scores.mean():.4f}")
        print(f"  Max cascade: {cascade_scores.max():.4f}")
        
        # Critical nodes
        print(f"\n  Top {top_k} critical nodes:")
        top_cascade = torch.topk(cascade_scores, k=min(top_k, len(cascade_scores))).indices
        for i, idx in enumerate(top_cascade):
            print(f"    {i+1}. Node {idx.item()}: {cascade_scores[idx].item():.4f}")
        
        return {
            'entropy': entropy,
            'sensitivity': sensitivity.cpu().numpy(),
            'information_flow': information_flow.cpu().numpy(),
            'cascade_scores': cascade_scores.cpu().numpy(),
            'top_sensitive_nodes': top_indices.cpu().numpy(),
            'top_critical_nodes': top_cascade.cpu().numpy()
        }
    
    def save_model(self, path: str = 'outputs/models/ht_hgnn_model.pt'):
        """Save trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'in_channels': self.in_channels,
                'hidden_channels': self.hidden_channels,
                'out_channels': self.out_channels,
                'num_nodes': self.num_nodes,
                'num_hyperedges': self.num_hyperedges
            },
            'history': self.history
        }, path)
        print(f"✓ Model saved to {path}")


def main():
    """
    Main execution
    Demonstrates the full HT-HGNN pipeline
    """
    
    # Load data
    features_df = pd.read_csv('outputs/datasets/features.csv')
    hci_labels_df = pd.read_csv('outputs/datasets/hci_labels.csv')
    
    # Initialize trainer
    trainer = HT_HGNN_Trainer(
        in_channels=18,
        hidden_channels=64,
        out_channels=32,
        num_nodes=1206,
        num_hyperedges=36,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.001
    )
    
    # Prepare data
    data = trainer.prepare_data(features_df, hci_labels_df)
    
    # Train model
    history = trainer.train(data, epochs=50, verbose=True)
    
    # Analyze entropy and sensitivity
    analysis = trainer.analyze_entropy_and_sensitivity(data)
    
    # Save model
    trainer.save_model()
    
    # Save analysis results
    analysis_results = {
        'entropy': float(analysis['entropy']),
        'mean_sensitivity': float(analysis['sensitivity'].mean()),
        'mean_information_flow': float(analysis['information_flow'].mean()),
        'mean_cascade_score': float(analysis['cascade_scores'].mean()),
        'top_sensitive_nodes': analysis['top_sensitive_nodes'].tolist(),
        'top_critical_nodes': analysis['top_critical_nodes'].tolist()
    }
    
    with open('outputs/ht_hgnn_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ HT-HGNN TRAINING COMPLETE")
    print("="*70)
    print(f"\nResults saved:")
    print(f"  Model: outputs/models/ht_hgnn_model.pt")
    print(f"  Analysis: outputs/ht_hgnn_analysis.json")


if __name__ == "__main__":
    main()
