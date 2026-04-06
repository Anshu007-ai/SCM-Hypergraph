"""
Heterogeneous Temporal Hypergraph Neural Network (HT-HGNN)

Core architecture combining:
1. Hypergraph Neural Network (HGNN+) - multi-way relationships
2. Heterogeneous Graph Transformer (HGT) - entity type distinction
3. Temporal Graph Network (TGN) - time-aware cascades

For entropy analysis via:
- Attention mechanism weights (information flow)
- Graph perturbation analysis (sensitivity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, TransformerConv
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass


@dataclass
class NodeFeatures:
    """Node representation with type and temporal info"""
    node_id: str
    node_type: str  # 'supplier', 'part', 'transaction'
    embedding: torch.Tensor
    timestamp: Optional[int] = None
    features: Optional[torch.Tensor] = None


class HypergraphConvolution(MessagePassing):
    """
    Hypergraph convolution layer (HGNN+)
    Handles multi-way relationships (not just pairwise)
    
    Flow:
    node → hyperedge (aggregation)
    hyperedge → node (propagation)
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_nodes: int, 
                 num_hyperedges: int, use_attention: bool = True):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.num_hyperedges = num_hyperedges
        self.use_attention = use_attention
        
        # Node to hyperedge transformation
        self.node_to_edge = nn.Linear(in_channels, out_channels)
        # Hyperedge to node transformation
        self.edge_to_node = nn.Linear(out_channels, out_channels)
        
        # Attention for hyperedges
        if use_attention:
            self.edge_attention = nn.Parameter(torch.ones(num_hyperedges))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.node_to_edge.reset_parameters()
        self.edge_to_node.reset_parameters()
        if self.use_attention:
            nn.init.ones_(self.edge_attention)
    
    def forward(self, node_features: torch.Tensor, 
                incidence_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (num_nodes, in_channels)
            incidence_matrix: (num_hyperedges, num_nodes) binary matrix
        
        Returns:
            node_out: Updated node embeddings
            hyperedge_out: Hyperedge embeddings
            attention_weights: Attention scores (for entropy analysis)
        """
        # Step 1: Node → Hyperedge (aggregation)
        # Use sparse matrix multiplication: H.t() @ X
        # H is (N, E), so H.t() is (E, N). X is (N, D). Result is (E, D).
        hyperedge_features = torch.sparse.mm(incidence_matrix.t(), node_features)
        hyperedge_features = self.node_to_edge(hyperedge_features)  # (E, out_D)
        
        # Step 2: Apply attention to hyperedges
        if self.use_attention:
            attention = torch.softmax(self.edge_attention, dim=0)  # (E,)
            hyperedge_features = hyperedge_features * attention.unsqueeze(1)  # (E, out_D)
        else:
            attention = torch.ones(self.num_hyperedges, device=node_features.device)
        
        # Step 3: Hyperedge → Node (propagation)
        # Use sparse matrix multiplication: H @ HE
        # H is (N, E), HE is (E, out_D). Result is (N, out_D).
        node_out = torch.sparse.mm(incidence_matrix, hyperedge_features)
        node_out = self.edge_to_node(node_out)  # (N, out_D)
        
        # Residual connection
        if self.in_channels == self.out_channels:
            node_out = node_out + node_features
        
        return node_out, hyperedge_features, attention


class HeterogenousGraphTransformer(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT)
    
    Distinguishes between node types:
    - suppliers
    - parts
    - transactions
    
    And different edge types:
    - supplies_to_part
    - uses_in_transaction
    - prices_in
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4,
                 node_types: List[str] = None, edge_types: List[str] = None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        
        # Default node and edge types
        self.node_types = node_types or ['supplier', 'part', 'transaction']
        self.edge_types = edge_types or ['supplies', 'uses', 'prices']
        
        # Type-specific embedding projections
        self.type_embedding = nn.ModuleDict({
            nt: nn.Linear(in_channels, out_channels) 
            for nt in self.node_types
        })
        
        # Transformer layers for each edge type
        self.transformer_layers = nn.ModuleDict({
            et: TransformerConv(out_channels, out_channels // num_heads, 
                               heads=num_heads, concat=True, dropout=0.1)
            for et in self.edge_types
        })
        
        # Attention mechanism for multi-head fusion
        # Each transformer outputs: (num_nodes, num_heads * (out_channels // num_heads)) = (num_nodes, out_channels)
        # So concatenating 3 edge types gives (num_nodes, 3 * out_channels)
        self.attention_fusion = nn.Linear(
            out_channels * len(edge_types), out_channels
        ) if len(edge_types) > 1 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.type_embedding.values():
            emb.reset_parameters()
        for conv in self.transformer_layers.values():
            conv.reset_parameters()
        if isinstance(self.attention_fusion, nn.Linear):
            self.attention_fusion.reset_parameters()

    def forward(self, node_features: torch.Tensor, 
                edge_index: torch.Tensor,
                node_types: List[str],
                edge_types: List[str]) -> torch.Tensor:
        """
        Args:
            node_features: (num_nodes, in_channels)
            edge_index: (2, num_edges)
            edge_types: Type label for each edge
        
        Returns:
            node_embeddings: (num_nodes, out_channels) with type-aware semantics
            attention_scores: Dict of attention by edge type
        """
        # Type-specific projections
        type_embeddings = {}
        for nt in self.node_types:
            mask = torch.tensor([n == nt for n in node_types])
            if mask.any():
                type_embeddings[nt] = self.type_embedding[nt](
                    node_features[mask]
                )
        
        # Apply transformers for each edge type
        edge_type_outputs = {}
        attention_scores = {}
        
        for et in self.edge_types:
            # Filter edges of this type
            edge_mask = torch.tensor([e == et for e in edge_types])
            if edge_mask.any():
                edge_index_et = edge_index[:, edge_mask]
                
                # Get output and attention
                output = self.transformer_layers[et](
                    node_features, edge_index_et
                )
                edge_type_outputs[et] = output
                
                # Store attention (for entropy analysis)
                attention_scores[et] = torch.mean(
                    self.transformer_layers[et].att if hasattr(
                        self.transformer_layers[et], 'att'
                    ) else torch.ones(node_features.size(0))
                )
        
        # Fuse outputs from all edge types
        if edge_type_outputs:
            combined = torch.cat(list(edge_type_outputs.values()), dim=-1)
            node_embeddings = self.attention_fusion(combined)
        else:
            node_embeddings = node_features
        
        return node_embeddings, attention_scores


class TemporalGraphNetwork(nn.Module):
    """
    Temporal Graph Network (TGN)
    
    Tracks how changes propagate through time
    Uses memory modules to maintain temporal state
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 time_window: int = 10):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_window = time_window
        
        # Memory module for each node
        # Stores temporal information about past interactions
        self.memory = nn.Parameter(
            torch.randn(1, out_channels)  # Will be expanded per node
        )
        
        # GRU cell for temporal state updates
        self.gru_cell = nn.GRUCell(in_channels, out_channels)
        
        # Temporal attention
        self.temporal_attention = nn.Linear(out_channels + 1, 1)  # +1 for time
        
        # Embedding projection
        self.projection = nn.Linear(in_channels, out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.gru_cell.reset_parameters()
        nn.init.xavier_uniform_(self.temporal_attention.weight)
        self.projection.reset_parameters()
    
    def forward(self, node_features: torch.Tensor, 
                timestamps: torch.Tensor,
                batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (num_nodes, in_channels)
            timestamps: (num_nodes,) temporal information
            batch_size: Number of time steps in batch
        
        Returns:
            temporal_embeddings: (num_nodes, out_channels)
            cascade_scores: (num_nodes,) propagation importance
        """
        num_nodes = node_features.size(0)
        device = node_features.device
        
        # Initialize memory
        node_memory = self.memory.expand(num_nodes, -1).clone()
        
        # Compute temporal attention weights
        time_encoded = torch.cat([
            node_features,
            timestamps.unsqueeze(1) / self.time_window
        ], dim=1)
        temporal_attention = torch.sigmoid(self.temporal_attention(time_encoded))
        
        # Update memory with GRU
        projected = self.projection(node_features)
        node_memory = self.gru_cell(projected, node_memory)
        
        # Apply temporal attention
        temporal_embeddings = node_memory * temporal_attention
        
        # Compute cascade scores (how much each node affects downstream)
        cascade_scores = torch.sum(
            temporal_attention * torch.abs(projected), dim=1
        )
        cascade_scores = cascade_scores / (cascade_scores.max() + 1e-8)
        
        return temporal_embeddings, cascade_scores


class SpectralHypergraphConv(nn.Module):
    """
    Spectral Hypergraph Convolution
    """
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        # Hypergraph adjacency matrix
        self.A = nn.Parameter(torch.randn(num_layers, in_channels, out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        pass
    
    def forward(self, x, incidence_matrix):
        """
        Args:
            x: (num_nodes, in_channels)
            incidence_matrix: (num_hyperedges, num_nodes) binary matrix
        
        Returns:
            output: (num_nodes, out_channels)
        """
        # Step 1: Hypergraph convolution
        # Use spectral decomposition
        # A is (num_layers, in_channels, out_channels)
        # We want to compute A @ X
        # X is (num_nodes, in_channels)
        # Result is (num_nodes, out_channels)
        output = torch.zeros(x.size(0), self.out_channels, device=x.device)
        
        for i in range(self.num_layers):
            # Compute X @ A
            output += torch.mm(x, self.A[i])
        
        return output


class HeterogeneousTemporalHypergraphNN(nn.Module):
    """
    Complete HT-HGNN Model combining all three components
    
    Architecture:
    Input → HGNN+ → HGT → TGN → Output heads
                          ↓
                    Entropy Analysis
    
    Outputs:
    1. Price predictions
    2. Change forecasts
    3. Critical node identification
    4. Entropy/sensitivity metrics
    """
    
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_nodes: int,
                 num_hyperedges: int,
                 node_types: list,
                 edge_types: list,
                 num_hgnn_layers: int = 2,
                 num_hgt_heads: int = 4,
                 time_window: int = 10,
                 use_spectral_conv: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.num_hyperedges = num_hyperedges
        self.num_hgt_heads = num_hgt_heads
        self.time_window = time_window
        self.use_spectral_conv = use_spectral_conv

        # 1. Hypergraph Convolution Layers
        self.hgnn_layers = None
        self.hypergraph_conv = None
        if use_spectral_conv:
            # v2.0 Spectral Convolution
            self.hypergraph_conv = SpectralHypergraphConv(
                in_channels, hidden_channels, num_hgnn_layers
            )
        else:
            # v1.0 HGNN+ Convolution
            self.hgnn_layers = nn.ModuleList([
                HypergraphConvolution(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    num_nodes=num_nodes,
                    num_hyperedges=num_hyperedges
                ) for i in range(num_hgnn_layers)
            ])
        
        # 2. Heterogeneous Graph Transformer
        self.hgt = HeterogenousGraphTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_heads=num_hgt_heads,
            node_types=node_types,
            edge_types=edge_types
        )
        
        # 3. Temporal Graph Network
        self.tgn = TemporalGraphNetwork(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            time_window=time_window
        )
        
        # 4. Output Heads
        self.price_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.change_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.criticality_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 4) # 4 classes: Low, Medium, High, Critical
        )
        
        # 5. Entropy Analyzer
        self.entropy_analyzer = EntropyAnalyzer(num_nodes, num_hyperedges)

        self.reset_parameters()

    def reset_parameters(self):
        if self.hgnn_layers:
            for layer in self.hgnn_layers:
                layer.reset_parameters()
        if self.hypergraph_conv:
            self.hypergraph_conv.reset_parameters()
        
        self.hgt.reset_parameters()
        self.tgn.reset_parameters()
        for head in [self.price_head, self.change_head, self.criticality_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
        self.entropy_analyzer.reset_parameters()
    
    def forward(self, 
                node_features: torch.Tensor,
                incidence_matrix: torch.Tensor, # Sparse (N, E) tensor
                node_types: List[str],
                edge_index: torch.Tensor,
                edge_types: List[str],
                timestamps: torch.Tensor) -> Dict:
        """
        Full forward pass through all layers
        """
        x = node_features
        attention_weights_hgnn = {}

        # 1. Apply hypergraph convolution
        if self.use_spectral_conv and self.hypergraph_conv:
            # Spectral conv expects (E, N)
            x = self.hypergraph_conv(x, incidence_matrix.t())
            attention_weights_hgnn['layer_0'] = torch.ones(self.num_hyperedges, device=x.device)
        elif self.hgnn_layers:
            # v1.0 HGNN+ layers expect (E, N)
            for i, hgnn_layer in enumerate(self.hgnn_layers):
                x, _, attn = hgnn_layer(x, incidence_matrix.t()) 
                attention_weights_hgnn[f'layer_{i}'] = attn
        
        # 2. HGT - entity type distinction
        x_hgt, attention_weights_hgt = self.hgt(
            x, edge_index, node_types, edge_types
        )
        
        # 3. TGN - temporal cascade tracking
        x_tgn, cascade_scores = self.tgn(x_hgt, timestamps, batch_size=1)
        
        # 4. Multi-task output heads
        price_pred = self.price_head(x_tgn)
        change_pred = self.change_head(x_tgn)
        criticality_logits = self.criticality_head(x_tgn)
        criticality_prob = F.softmax(criticality_logits, dim=-1)
        criticality_pred = torch.argmax(criticality_prob, dim=-1)
        
        # 5. Entropy analysis
        entropy_metrics = self.entropy_analyzer(
            node_features=x_tgn,
            attention_weights_hgnn=attention_weights_hgnn,
            attention_weights_hgt=attention_weights_hgt,
            cascade_scores=cascade_scores,
            incidence_matrix=incidence_matrix
        )
        
        return {
            "price_pred": price_pred,
            "change_pred": change_pred,
            "criticality_logits": criticality_logits,
            "criticality_prob": criticality_prob,
            "criticality_pred": criticality_pred,
            "cascade_scores": cascade_scores,
            "entropy": entropy_metrics["entropy"],
            "sensitivity": entropy_metrics["sensitivity"],
            "information_flow": entropy_metrics["information_flow"],
            "attention_hgnn": attention_weights_hgnn,
            "attention_hgt": attention_weights_hgt,
        }


class EntropyAnalyzer(nn.Module):
    """
    Computes entropy and sensitivity metrics
    
    Entropy Analysis via:
    1. Attention mechanism weights (information flow)
    2. Graph perturbation (sensitivity)
    """
    
    def __init__(self, hidden_channels: int, num_nodes: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes
        
        self.reset_parameters()
    
    def reset_parameters(self):
        pass
    
    def forward(self,
                node_features: torch.Tensor,
                attention_weights_hgnn: Dict,
                attention_weights_hgt: Dict,
                cascade_scores: torch.Tensor,
                incidence_matrix: torch.Tensor) -> Dict:
        """
        Compute entropy and sensitivity metrics
        
        Returns:
        - entropy: Shannon entropy of information flow
        - sensitivity: Sensitivity to node perturbations
        - information_flow: Which nodes carry most info
        """
        
        # 1. Entropy from attention weights
        entropy_hgnn = self._compute_attention_entropy(attention_weights_hgnn)
        entropy_hgt = sum(attention_weights_hgt.values()) / len(attention_weights_hgt)
        
        # 2. Sensitivity via perturbation analysis
        sensitivity = self._compute_perturbation_sensitivity(
            node_features, incidence_matrix
        )
        
        # 3. Information flow from cascade scores
        information_flow = self._compute_information_flow(
            cascade_scores, attention_weights_hgnn
        )
        
        return {
            'entropy': (entropy_hgnn + entropy_hgt) / 2,
            'sensitivity': sensitivity,
            'information_flow': information_flow
        }
    
    def _compute_attention_entropy(self, attention_weights: Dict) -> torch.Tensor:
        """
        Shannon entropy of attention weights
        High entropy = distributed attention (many paths)
        Low entropy = focused attention (few critical paths)
        """
        entropies = []
        for layer_attn in attention_weights.values():
            # Normalize attention
            p = layer_attn / (layer_attn.sum() + 1e-8)
            # Shannon entropy
            entropy = -(p * torch.log(p + 1e-8)).sum()
            entropies.append(entropy)
        
        return torch.stack(entropies).mean() if entropies else torch.tensor(0.0)
    
    def _compute_perturbation_sensitivity(self,
                                        node_features: torch.Tensor,
                                        incidence_matrix: torch.Tensor) -> torch.Tensor:
        """
        Simplified graph perturbation analysis
        
        For each node (sampled for efficiency):
        1. Measure its feature importance
        2. Compute propagation reach via incidence
        3. Aggregate as sensitivity score
        """
        num_nodes = node_features.size(0)
        
        # Sample nodes for efficiency (use 100 nodes max)
        sample_size = min(100, num_nodes)
        sample_indices = torch.randperm(num_nodes)[:sample_size]
        
        # Precompute hyperedge sums for efficiency
        max_hyperedges = incidence_matrix.sum(dim=1).max() + 1e-8
        
        sensitivities = []
        for idx in sample_indices:
            i = idx.item()
            # Feature magnitude as importance proxy
            feature_magnitude = torch.norm(node_features[i])
            
            # Propagation impact via incidence matrix
            affected_hyperedges = incidence_matrix[:, i].sum()
            propagation = affected_hyperedges / max_hyperedges
            
            # Sensitivity = feature importance × propagation reach
            sensitivity = feature_magnitude * propagation
            sensitivities.append(sensitivity)
        
        # Expand sampled sensitivities to all nodes
        all_sensitivities = torch.ones(num_nodes, device=node_features.device)
        sampled_sensitivities = torch.stack(sensitivities)
        
        # Scale based on feature norms for unsampled nodes
        for i in range(num_nodes):
            if i not in sample_indices:
                feature_magnitude = torch.norm(node_features[i])
                affected_hyperedges = incidence_matrix[:, i].sum()
                propagation = affected_hyperedges / max_hyperedges
                all_sensitivities[i] = feature_magnitude * propagation
            else:
                sample_pos = (sample_indices == i).nonzero(as_tuple=True)[0].item()
                all_sensitivities[i] = sampled_sensitivities[sample_pos]
        
        # Normalize
        all_sensitivities = all_sensitivities / (all_sensitivities.max() + 1e-8)
        
        return all_sensitivities
    
    def _compute_information_flow(self,
                                 cascade_scores: torch.Tensor,
                                 attention_weights: Dict) -> torch.Tensor:
        """
        Combined information flow metric
        
        Combines:
        - Cascade propagation score (node-level, 1206 nodes)
        - Attention-based importance (hyperedge-level, 36 hyperedges)
        """
        # For hyperedges: average the attention weights
        if attention_weights:
            avg_attention_hyperedge = sum(w.mean() if isinstance(w, torch.Tensor) else torch.tensor(w) 
                                         for w in attention_weights.values()) / len(attention_weights)
        else:
            avg_attention_hyperedge = torch.tensor(1.0, device=cascade_scores.device)
        
        # Scale cascade scores by average attention
        information_flow = cascade_scores * avg_attention_hyperedge
        
        return information_flow


class MultiTaskLoss(nn.Module):
    """
    Multi-task learning loss
    
    Combines three tasks:
    1. Price prediction (MSE)
    2. Change forecast (MSE)
    3. Critical node identification (Cross-entropy for multi-class)
    """
    
    def __init__(self, weight_price: float = 1.0, weight_change: float = 0.5,
                 weight_criticality: float = 1.2, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight_price = weight_price
        self.weight_change = weight_change
        self.weight_criticality = weight_criticality
        
        self.mse_loss = nn.MSELoss()
        # Use CrossEntropyLoss for multi-class classification
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self,
                price_pred: torch.Tensor,
                price_target: torch.Tensor,
                change_pred: torch.Tensor,
                change_target: torch.Tensor,
                criticality_pred: torch.Tensor,
                criticality_target: torch.Tensor) -> Dict:
        """
        Compute weighted multi-task loss
        """
        
        loss_price = self.mse_loss(price_pred, price_target)
        loss_change = self.mse_loss(change_pred, change_target)
        # Ensure target is long type for CrossEntropyLoss
        loss_criticality = self.ce_loss(criticality_pred, criticality_target.long())
        
        total_loss = (self.weight_price * loss_price +
                     self.weight_change * loss_change +
                     self.weight_criticality * loss_criticality)
        
        return {
            'total_loss': total_loss,
            'loss_price': loss_price.item(),
            'loss_change': loss_change.item(),
            'loss_criticality': loss_criticality.item()
        }


if __name__ == "__main__":
    print("HT-HGNN Model Definition Complete")
    print("\nArchitecture:")
    print("1. HGNN+ - Hypergraph convolution (multi-way relationships)")
    print("2. HGT - Heterogeneous transformer (entity type distinction)")
    print("3. TGN - Temporal graph network (cascade tracking)")
    print("\nOutputs:")
    print("- Price predictions")
    print("- Change forecasts")
    print("- Critical node scores")
    print("- Entropy/sensitivity metrics")
    print("\nFor entropy analysis:")
    print("- Attention weights as information flow")
    print("- Perturbation analysis for sensitivity")
