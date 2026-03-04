"""
Hyperedge Importance Analysis for Hypergraph Neural Networks

Provides multiple methods for computing hyperedge-level importance scores:
1. Gradient-based attribution - backpropagates prediction signal to hyperedges
2. Leave-one-out removal - measures prediction change when each hyperedge is masked
3. Attention-based extraction - uses learned attention weights from the model

These importance scores identify which supply chain subassemblies (hyperedges)
are most critical for model predictions.

Author: HT-HGNN v2.0 Project
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class HyperedgeImportanceAnalyzer:
    """
    Analyzes and ranks hyperedge importance in a hypergraph neural network.

    Supports three attribution methods:
        - 'gradient': Gradient-based importance using backpropagation.
        - 'removal': Leave-one-out importance by masking each hyperedge.
        - 'attention': Extracts learned attention weights from the model.

    Attributes:
        model: Trained HT-HGNN model instance.
        hypergraph: Hypergraph object with incidence structure.
        device: Torch device for computation.
        incidence_matrix: Tensor form of the incidence matrix.
        hyperedge_ids: Ordered list of hyperedge identifiers.
        node_ids: Ordered list of node identifiers.
    """

    def __init__(self, model: nn.Module, hypergraph: Any):
        """
        Initialize the hyperedge importance analyzer.

        Args:
            model: A trained HT-HGNN model (or compatible nn.Module).
            hypergraph: Hypergraph object providing get_incidence_matrix() method.
        """
        self.model = model
        self.model.eval()
        self.hypergraph = hypergraph
        self.device = next(model.parameters()).device

        # Extract incidence matrix from the hypergraph
        H_np, self.hyperedge_ids, self.node_ids = hypergraph.get_incidence_matrix()
        self.incidence_matrix = torch.tensor(H_np, dtype=torch.float32).to(self.device)
        self.num_hyperedges = self.incidence_matrix.shape[0]
        self.num_nodes = self.incidence_matrix.shape[1]

        # Cache for importance scores
        self._importance_cache: Dict[str, Dict[str, float]] = {}

    def compute_importance(
        self,
        node_features: torch.Tensor,
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        method: str = 'gradient',
        prediction_type: str = 'criticality',
    ) -> Dict[str, float]:
        """
        Compute importance scores for all hyperedges.

        Args:
            node_features: Node feature tensor (num_nodes, num_features).
            node_types: Type label for each node.
            edge_index: Edge index tensor (2, num_edges).
            edge_types: Type label for each edge.
            timestamps: Timestamp tensor for temporal component.
            method: Attribution method - 'gradient', 'removal', or 'attention'.
            prediction_type: Which output head to analyze
                ('criticality', 'price', 'change').

        Returns:
            Dictionary mapping hyperedge_id (str) to importance score (float).

        Raises:
            ValueError: If an unsupported method is specified.
        """
        method_dispatch = {
            'gradient': self._gradient_importance,
            'removal': self._removal_importance,
            'attention': self._attention_importance,
        }

        if method not in method_dispatch:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Supported: {list(method_dispatch.keys())}"
            )

        importance_fn = method_dispatch[method]
        scores = importance_fn(
            node_features, node_types, edge_index,
            edge_types, timestamps, prediction_type
        )

        # Cache the result
        cache_key = f"{method}_{prediction_type}"
        self._importance_cache[cache_key] = scores

        return scores

    def _gradient_importance(
        self,
        node_features: torch.Tensor,
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        prediction_type: str,
    ) -> Dict[str, float]:
        """
        Compute gradient-based hyperedge importance.

        Creates a differentiable mask over the incidence matrix and
        backpropagates the prediction loss to compute gradients with
        respect to each hyperedge mask entry.

        Args:
            node_features: Node feature tensor.
            node_types: Type labels.
            edge_index: Edge connectivity.
            edge_types: Edge type labels.
            timestamps: Temporal information.
            prediction_type: Output head to analyze.

        Returns:
            Dictionary mapping hyperedge_id to gradient-based importance.
        """
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        timestamps = timestamps.to(self.device)

        # Create differentiable hyperedge mask
        hyperedge_mask = torch.ones(
            self.num_hyperedges, device=self.device, requires_grad=True
        )
        masked_incidence = self.incidence_matrix * hyperedge_mask.unsqueeze(1)

        # Forward pass
        output = self.model(
            node_features, masked_incidence, node_types,
            edge_index, edge_types, timestamps
        )

        pred_key_map = {
            'criticality': 'criticality',
            'price': 'price_pred',
            'change': 'change_pred',
        }
        pred_key = pred_key_map.get(prediction_type, 'criticality')
        prediction = output[pred_key]

        # Backpropagate summed prediction to get gradient w.r.t. mask
        loss = prediction.sum()
        loss.backward()

        # Extract gradient magnitudes as importance
        grad = hyperedge_mask.grad
        if grad is None:
            grad = torch.zeros(self.num_hyperedges, device=self.device)

        importance_values = torch.abs(grad).detach().cpu().numpy()

        # Normalize to [0, 1]
        max_val = importance_values.max()
        if max_val > 0:
            importance_values = importance_values / max_val

        # Map to hyperedge ids
        scores = {}
        for i, hid in enumerate(self.hyperedge_ids):
            scores[hid] = float(importance_values[i])

        return scores

    @torch.no_grad()
    def _removal_importance(
        self,
        node_features: torch.Tensor,
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        prediction_type: str,
    ) -> Dict[str, float]:
        """
        Compute leave-one-out hyperedge importance.

        For each hyperedge, compute the model output with that hyperedge
        removed and measure the change from the full prediction.

        Args:
            node_features: Node feature tensor.
            node_types: Type labels.
            edge_index: Edge connectivity.
            edge_types: Edge type labels.
            timestamps: Temporal information.
            prediction_type: Output head to analyze.

        Returns:
            Dictionary mapping hyperedge_id to removal-based importance.
        """
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        timestamps = timestamps.to(self.device)

        pred_key_map = {
            'criticality': 'criticality',
            'price': 'price_pred',
            'change': 'change_pred',
        }
        pred_key = pred_key_map.get(prediction_type, 'criticality')

        # Full prediction (baseline)
        full_output = self.model(
            node_features, self.incidence_matrix, node_types,
            edge_index, edge_types, timestamps
        )
        full_pred = full_output[pred_key].detach()
        if full_pred.dim() == 0:
            full_pred_value = full_pred.item()
        else:
            full_pred_value = full_pred.mean().item()

        scores = {}
        for i, hid in enumerate(self.hyperedge_ids):
            # Create mask with hyperedge i removed
            mask = torch.ones(self.num_hyperedges, device=self.device)
            mask[i] = 0.0
            masked_incidence = self.incidence_matrix * mask.unsqueeze(1)

            # Predict without this hyperedge
            output = self.model(
                node_features, masked_incidence, node_types,
                edge_index, edge_types, timestamps
            )
            pred = output[pred_key].detach()
            if pred.dim() == 0:
                pred_value = pred.item()
            else:
                pred_value = pred.mean().item()

            # Importance = absolute change when removed
            scores[hid] = abs(full_pred_value - pred_value)

        # Normalize to [0, 1]
        max_val = max(scores.values()) if scores else 1.0
        if max_val > 0:
            scores = {k: v / max_val for k, v in scores.items()}

        return scores

    @torch.no_grad()
    def _attention_importance(
        self,
        node_features: torch.Tensor,
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        prediction_type: str,
    ) -> Dict[str, float]:
        """
        Extract importance from model attention weights.

        Uses the attention weights learned by the HGNN+ layers to
        directly measure the importance assigned by the model to
        each hyperedge.

        Args:
            node_features: Node feature tensor.
            node_types: Type labels.
            edge_index: Edge connectivity.
            edge_types: Edge type labels.
            timestamps: Temporal information.
            prediction_type: Output head (used for context, attention is shared).

        Returns:
            Dictionary mapping hyperedge_id to attention-based importance.
        """
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        timestamps = timestamps.to(self.device)

        # Forward pass to collect attention weights
        output = self.model(
            node_features, self.incidence_matrix, node_types,
            edge_index, edge_types, timestamps
        )

        # Extract attention weights from HGNN layers
        attn_weights = output.get('attention_weights_hgnn', {})

        if not attn_weights:
            # Fallback: uniform importance
            scores = {hid: 1.0 / self.num_hyperedges for hid in self.hyperedge_ids}
            return scores

        # Average attention across layers
        accumulated = torch.zeros(self.num_hyperedges, device=self.device)
        n_layers = 0
        for layer_key, attn in attn_weights.items():
            if isinstance(attn, torch.Tensor) and attn.shape[0] == self.num_hyperedges:
                accumulated += attn.detach()
                n_layers += 1

        if n_layers > 0:
            accumulated = accumulated / n_layers
        else:
            accumulated = torch.ones(self.num_hyperedges, device=self.device)

        # Normalize to [0, 1]
        importance_values = accumulated.cpu().numpy()
        max_val = importance_values.max()
        if max_val > 0:
            importance_values = importance_values / max_val

        scores = {}
        for i, hid in enumerate(self.hyperedge_ids):
            scores[hid] = float(importance_values[i])

        return scores

    def rank_hyperedges(
        self,
        scores: Optional[Dict[str, float]] = None,
        method: str = 'gradient',
    ) -> List[Tuple[str, float]]:
        """
        Rank hyperedges by importance score in descending order.

        Args:
            scores: Pre-computed importance scores. If None, uses the
                last cached scores for the given method.
            method: Method key for cache lookup if scores is None.

        Returns:
            Sorted list of (hyperedge_id, importance_score) tuples,
            ordered from most to least important.
        """
        if scores is None:
            # Look up most recent cached result for this method
            matching_keys = [k for k in self._importance_cache if k.startswith(method)]
            if matching_keys:
                scores = self._importance_cache[matching_keys[-1]]
            else:
                raise ValueError(
                    f"No cached scores for method '{method}'. "
                    "Call compute_importance() first or provide scores directly."
                )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def get_critical_hyperedges(
        self,
        scores: Optional[Dict[str, float]] = None,
        threshold: float = 0.8,
        method: str = 'gradient',
    ) -> List[Tuple[str, float]]:
        """
        Retrieve hyperedges with importance above the given threshold.

        Args:
            scores: Pre-computed importance scores. If None, uses cached scores.
            threshold: Minimum importance score to be considered critical.
                Scores are expected in [0, 1].
            method: Method key for cache lookup if scores is None.

        Returns:
            List of (hyperedge_id, importance_score) tuples for high-importance
            hyperedges, sorted in descending order.
        """
        ranked = self.rank_hyperedges(scores=scores, method=method)
        critical = [(hid, score) for hid, score in ranked if score >= threshold]
        return critical

    def compare_methods(
        self,
        node_features: torch.Tensor,
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        prediction_type: str = 'criticality',
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute and compare importance scores across all three methods.

        Args:
            node_features: Node feature tensor.
            node_types: Type labels.
            edge_index: Edge connectivity.
            edge_types: Edge type labels.
            timestamps: Temporal information.
            prediction_type: Output head to analyze.

        Returns:
            Dictionary with method names as keys, each mapping to
            hyperedge importance dictionaries.
        """
        results = {}
        for method in ['gradient', 'removal', 'attention']:
            results[method] = self.compute_importance(
                node_features, node_types, edge_index,
                edge_types, timestamps,
                method=method,
                prediction_type=prediction_type,
            )
        return results

    def summary(self, scores: Dict[str, float], top_k: int = 10) -> str:
        """
        Generate a text summary of hyperedge importance scores.

        Args:
            scores: Dictionary of hyperedge_id to importance score.
            top_k: Number of top hyperedges to include in summary.

        Returns:
            Formatted string summary.
        """
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        lines = [
            "=== Hyperedge Importance Analysis ===",
            f"Total hyperedges analyzed: {len(scores)}",
            f"Top {min(top_k, len(ranked))} hyperedges:",
            "",
        ]

        for i, (hid, score) in enumerate(ranked[:top_k]):
            bar = "#" * int(score * 30)
            lines.append(f"  {i+1:3d}. {hid:<20s} {score:.4f} |{bar}")

        # Statistics
        values = list(scores.values())
        lines.extend([
            "",
            f"Mean importance:   {np.mean(values):.4f}",
            f"Std importance:    {np.std(values):.4f}",
            f"Max importance:    {np.max(values):.4f}",
            f"Min importance:    {np.min(values):.4f}",
            f"Critical (>0.8):   {sum(1 for v in values if v >= 0.8)}",
        ])

        return "\n".join(lines)


if __name__ == "__main__":
    print("Hyperedge Importance Analyzer Module")
    print("=" * 50)
    print()
    print("Supported attribution methods:")
    print("  1. gradient  - Gradient-based backpropagation importance")
    print("  2. removal   - Leave-one-out removal analysis")
    print("  3. attention - Learned attention weight extraction")
    print()
    print("Usage example:")
    print("  analyzer = HyperedgeImportanceAnalyzer(model, hypergraph)")
    print("  scores = analyzer.compute_importance(features, ..., method='gradient')")
    print("  critical = analyzer.get_critical_hyperedges(scores, threshold=0.8)")
    print("  ranked = analyzer.rank_hyperedges(scores)")
    print()
    print("Module ready for integration with HT-HGNN v2.0.")
