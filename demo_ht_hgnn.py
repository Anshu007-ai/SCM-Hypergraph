"""
Quick demonstration of HT-HGNN architecture
Shows model instantiation, data loading, and entropy analysis without full training
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.models.ht_hgnn_model import (
    HeterogeneousTemporalHypergraphNN,
    MultiTaskLoss,
    EntropyAnalyzer
)


def demo_ht_hgnn():
    """
    Demonstrate HT-HGNN model capabilities
    """
    
    print("\n" + "="*70)
    print("HT-HGNN MODEL DEMONSTRATION")
    print("="*70)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Using device: {device}")
    
    # Load data
    print("\n" + "-"*70)
    print("LOADING DATA")
    print("-"*70)
    
    features_df = pd.read_csv('outputs/datasets/features.csv')
    hci_labels_df = pd.read_csv('outputs/datasets/hci_labels.csv')
    incidence_df = pd.read_csv('outputs/datasets/incidence.csv')
    
    num_nodes = 1206
    num_hyperedges = 36
    in_channels = 18
    
    print(f"✓ Features shape: {features_df.shape}")
    print(f"✓ HCI labels shape: {hci_labels_df.shape}")
    print(f"✓ Incidence shape: {incidence_df.shape}")
    
    # Prepare tensors
    print("\n" + "-"*70)
    print("PREPARING DATA TENSORS")
    print("-"*70)
    
    # Node features from hyperedge aggregation
    hyperedge_features = features_df.drop('hyperedge_id', axis=1).values.astype(np.float32)
    node_features = np.zeros((num_nodes, in_channels), dtype=np.float32)
    
    hyperedge_map = {he_id: i for i, he_id in enumerate(features_df['hyperedge_id'].values)}
    node_counts = np.zeros(num_nodes)
    
    for _, row in incidence_df.iterrows():
        he_idx = hyperedge_map.get(row['hyperedge_id'])
        node_idx = int(row['node_id'].split('_')[1]) % num_nodes
        if he_idx is not None and 0 <= node_idx < num_nodes:
            node_features[node_idx] += hyperedge_features[he_idx]
            node_counts[node_idx] += 1
    
    for i in range(num_nodes):
        if node_counts[i] > 0:
            node_features[i] /= node_counts[i]
    
    unconnected_mask = node_counts == 0
    if unconnected_mask.any():
        avg_features = hyperedge_features.mean(axis=0)
        node_features[unconnected_mask] = avg_features
    
    X_tensor = torch.FloatTensor(node_features).to(device)
    X_mean = X_tensor.mean(dim=0)
    X_std = X_tensor.std(dim=0) + 1e-8
    X_tensor = (X_tensor - X_mean) / X_std
    
    print(f"✓ Node features tensor: {X_tensor.shape}")
    
    # Incidence matrix
    incidence_matrix = np.zeros((num_hyperedges, num_nodes))
    for _, row in incidence_df.iterrows():
        he_idx = hyperedge_map.get(row['hyperedge_id'])
        node_idx = int(row['node_id'].split('_')[1]) % num_nodes
        if he_idx is not None and 0 <= node_idx < num_nodes:
            incidence_matrix[he_idx, node_idx] = 1
    
    incidence_tensor = torch.FloatTensor(incidence_matrix).to(device)
    print(f"✓ Incidence matrix: {incidence_tensor.shape}")
    
    # Edge index
    edges_i, edges_j = torch.nonzero(incidence_tensor, as_tuple=True)
    edge_index = torch.stack([edges_i, edges_j]).to(device)
    edge_types = ['supplies', 'uses', 'prices']
    assigned_edge_types = [edge_types[i % len(edge_types)] for i in range(edge_index.size(1))]
    print(f"✓ Edge index: {edge_index.shape}")
    
    # Node types
    node_types = []
    nodes_per_type = num_nodes // 3
    for i in range(num_nodes):
        if i < nodes_per_type:
            node_types.append('supplier')
        elif i < 2 * nodes_per_type:
            node_types.append('part')
        else:
            node_types.append('transaction')
    
    # Timestamps
    timestamps = torch.linspace(0, 10, num_nodes).to(device)
    
    print(f"✓ Node types: {len(set(node_types))} types")
    print(f"✓ Timestamps: {timestamps.shape}")
    
    # Initialize model
    print("\n" + "-"*70)
    print("INITIALIZING HT-HGNN MODEL")
    print("-"*70)
    
    model = HeterogeneousTemporalHypergraphNN(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=32,
        num_nodes=num_nodes,
        num_hyperedges=num_hyperedges,
        node_types=['supplier', 'part', 'transaction'],
        edge_types=['supplies', 'uses', 'prices'],
        num_hgnn_layers=2,
        num_hgt_heads=4,
        time_window=10
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized with {total_params:,} parameters")
    
    # Forward pass
    print("\n" + "-"*70)
    print("FORWARD PASS & ENTROPY ANALYSIS")
    print("-"*70)
    
    model.eval()
    with torch.no_grad():
        output = model(
            node_features=X_tensor,
            incidence_matrix=incidence_tensor,
            node_types=node_types,
            edge_index=edge_index,
            edge_types=assigned_edge_types,
            timestamps=timestamps
        )
    
    # Extract outputs
    print(f"\n✓ Model outputs:")
    print(f"  Price predictions: {output['price_pred'].shape}")
    print(f"    Range: [{output['price_pred'].min().item():.2f}, {output['price_pred'].max().item():.2f}]")
    print(f"  Change forecast: {output['change_pred'].shape}")
    print(f"    Range: [{output['change_pred'].min().item():.4f}, {output['change_pred'].max().item():.4f}]")
    print(f"  Criticality: {output['criticality'].shape}")
    print(f"    Range: [{output['criticality'].min().item():.4f}, {output['criticality'].max().item():.4f}]")
    
    # Entropy metrics
    print(f"\n✓ Entropy Metrics:")
    entropy = output['entropy'].item() if isinstance(output['entropy'], torch.Tensor) else output['entropy']
    print(f"  Information Flow Entropy: {entropy:.6f}")
    
    print(f"\n✓ Sensitivity Analysis:")
    sensitivity = output['sensitivity']
    print(f"  Shape: {sensitivity.shape}")
    print(f"  Mean: {sensitivity.mean().item():.6f}")
    print(f"  Max: {sensitivity.max().item():.6f}")
    print(f"  Std: {sensitivity.std().item():.6f}")
    
    # Top sensitive nodes
    top_k = 10
    top_indices = torch.topk(sensitivity, k=min(top_k, len(sensitivity))).indices
    print(f"\n  Top {top_k} most sensitive nodes:")
    for i, idx in enumerate(top_indices):
        print(f"    {i+1}. Node {idx.item():4d}: sensitivity = {sensitivity[idx].item():.6f}")
    
    # Information flow
    info_flow = output['information_flow']
    print(f"\n✓ Information Flow:")
    print(f"  Mean: {info_flow.mean().item():.6f}")
    print(f"  Max: {info_flow.max().item():.6f}")
    print(f"  Std: {info_flow.std().item():.6f}")
    
    # Cascade scores
    cascade = output['cascade_scores']
    print(f"\n✓ Cascade/Propagation Scores:")
    print(f"  Mean: {cascade.mean().item():.6f}")
    print(f"  Max: {cascade.max().item():.6f}")
    
    top_cascade = torch.topk(cascade, k=min(top_k, len(cascade))).indices
    print(f"\n  Top {top_k} critical nodes:")
    for i, idx in enumerate(top_cascade):
        print(f"    {i+1}. Node {idx.item():4d}: cascade = {cascade[idx].item():.6f}")
    
    # Save results
    print("\n" + "-"*70)
    print("SAVING RESULTS")
    print("-"*70)
    
    results = {
        'model_config': {
            'in_channels': in_channels,
            'hidden_channels': 64,
            'out_channels': 32,
            'num_nodes': num_nodes,
            'num_hyperedges': num_hyperedges,
            'total_parameters': int(total_params)
        },
        'metrics': {
            'entropy': float(entropy),
            'sensitivity_mean': float(sensitivity.mean().item()),
            'sensitivity_max': float(sensitivity.max().item()),
            'sensitivity_std': float(sensitivity.std().item()),
            'information_flow_mean': float(info_flow.mean().item()),
            'cascade_mean': float(cascade.mean().item()),
            'cascade_max': float(cascade.max().item())
        },
        'top_sensitive_nodes': [int(i) for i in top_indices.cpu().numpy()],
        'top_critical_nodes': [int(i) for i in top_cascade.cpu().numpy()]
    }
    
    output_path = Path('outputs/ht_hgnn_demo.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("HT-HGNN DEMONSTRATION SUMMARY")
    print("="*70)
    print(f"""
Architecture Components:
  ✓ HGNN+ Layer (2×): Hypergraph message passing with attention
  ✓ HGT Layer: Heterogeneous transformer with 4 heads
  ✓ TGN Layer: Temporal graph network with GRU cells
  ✓ 3 Output Heads: Price, Change, Criticality

Entropy Analysis Capabilities:
  ✓ Attention-based Information Flow: {entropy:.6f}
  ✓ Perturbation-based Sensitivity: Mean = {sensitivity.mean().item():.6f}
  ✓ Cascade Propagation Scoring: Mean = {cascade.mean().item():.6f}
  ✓ Information Flow Metrics: Mean = {info_flow.mean().item():.6f}

Data Configuration:
  ✓ Nodes: {num_nodes} (402 suppliers, 402 parts, 402 transactions)
  ✓ Hyperedges: {num_hyperedges}
  ✓ Features per node: {in_channels}
  ✓ Total edges: {edge_index.size(1)}
  ✓ Incidence sparsity: {(incidence_tensor.sum() / incidence_tensor.numel() * 100):.1f}%

Next Steps:
  → Run `train_ht_hgnn.py` for full training (50 epochs)
  → Compare against baseline Gradient Boosting (R²=0.8969)
  → Evaluate multi-task learning benefits
  → Visualize attention weights and cascade importance
""")


if __name__ == "__main__":
    demo_ht_hgnn()
