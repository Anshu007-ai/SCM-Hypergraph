"""
Feature Attribution Analysis for Hypergraph Neural Networks

Provides node-level feature attribution using integrated gradients as the
primary method. Computes how much each input feature contributes to the
model's prediction for individual nodes or batches of nodes.

Integrated Gradients formula:
    IG_i(x) = (x_i - x'_i) * integral_{alpha=0}^{1}
              (partial F(x' + alpha*(x - x')) / partial x_i) d_alpha

Where x' is the baseline input (typically zero), and the integral is
approximated via Riemann summation.

Author: HT-HGNN v2.0 Project
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


class FeatureAttributionAnalyzer:
    """
    Computes per-feature attribution scores for node predictions.

    Uses integrated gradients as the primary attribution method,
    providing a principled way to decompose model predictions into
    per-feature contributions.

    Attributes:
        model: Trained HT-HGNN model instance.
        feature_names: List of human-readable feature names.
        device: Torch device for computation.
        n_steps: Number of interpolation steps for integrated gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        n_steps: int = 50,
    ):
        """
        Initialize the feature attribution analyzer.

        Args:
            model: A trained HT-HGNN model (or compatible nn.Module).
            feature_names: Ordered list of feature names corresponding to
                columns of the node feature matrix.
            n_steps: Number of interpolation steps for integrated gradients.
                Higher values give more accurate attributions at the cost
                of computation time.
        """
        self.model = model
        self.model.eval()
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.n_steps = n_steps
        self.device = next(model.parameters()).device

        # Cache for batch computation results
        self._batch_cache: Optional[pd.DataFrame] = None

    def _get_prediction_key(self, prediction_type: str) -> str:
        """
        Map user-facing prediction type to model output key.

        Args:
            prediction_type: One of 'criticality', 'price', 'change'.

        Returns:
            Corresponding key in the model output dictionary.
        """
        key_map = {
            'criticality': 'criticality',
            'price': 'price_pred',
            'change': 'change_pred',
        }
        return key_map.get(prediction_type, 'criticality')

    def compute_attributions(
        self,
        node_features: torch.Tensor,
        node_id: int,
        incidence_matrix: torch.Tensor,
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        prediction_type: str = 'criticality',
        baseline: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute feature attributions for a single node using integrated gradients.

        Interpolates between a baseline (default: zero) and the actual input,
        accumulating gradients at each step to compute the integrated gradient
        for each feature.

        Args:
            node_features: Full node feature tensor (num_nodes, num_features).
            node_id: Index of the target node.
            incidence_matrix: Hypergraph incidence matrix tensor.
            node_types: Type label for each node.
            edge_index: Edge index tensor (2, num_edges).
            edge_types: Type label for each edge.
            timestamps: Timestamp tensor for temporal component.
            prediction_type: Which output head to explain.
            baseline: Optional custom baseline tensor. If None, uses zeros.

        Returns:
            Dictionary mapping feature_name to SHAP-like attribution value.
        """
        node_features = node_features.to(self.device)
        incidence_matrix = incidence_matrix.to(self.device)
        edge_index = edge_index.to(self.device)
        timestamps = timestamps.to(self.device)

        pred_key = self._get_prediction_key(prediction_type)

        # Baseline: zero features
        if baseline is None:
            baseline = torch.zeros_like(node_features).to(self.device)
        else:
            baseline = baseline.to(self.device)

        # Compute integrated gradients
        accumulated_grads = torch.zeros(self.num_features, device=self.device)

        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            interpolated = baseline + alpha * (node_features - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)

            output = self.model(
                interpolated, incidence_matrix, node_types,
                edge_index, edge_types, timestamps
            )

            prediction = output[pred_key]
            if prediction.dim() > 0:
                target_pred = prediction[node_id]
            else:
                target_pred = prediction

            target_pred.backward(retain_graph=True)

            if interpolated.grad is not None:
                accumulated_grads += interpolated.grad[node_id].detach()
                interpolated.grad = None

        # IG = (input - baseline) * mean_gradient
        avg_grads = accumulated_grads / (self.n_steps + 1)
        delta = node_features[node_id] - baseline[node_id]
        attributions = (delta.to(self.device) * avg_grads).detach().cpu().numpy()

        # Build result dictionary
        result = {}
        for i, name in enumerate(self.feature_names):
            result[name] = float(attributions[i])

        return result

    def compute_batch_attributions(
        self,
        features_batch: torch.Tensor,
        node_ids: List[int],
        incidence_matrix: torch.Tensor,
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        prediction_type: str = 'criticality',
    ) -> pd.DataFrame:
        """
        Compute feature attributions for a batch of nodes.

        Runs integrated gradients for each node in the batch and
        assembles results into a pandas DataFrame.

        Args:
            features_batch: Node feature tensor (num_nodes, num_features).
            node_ids: List of node indices to compute attributions for.
            incidence_matrix: Hypergraph incidence matrix tensor.
            node_types: Type label for each node.
            edge_index: Edge index tensor.
            edge_types: Edge type labels.
            timestamps: Temporal information.
            prediction_type: Which output head to explain.

        Returns:
            DataFrame with node_ids as index and feature_names as columns.
            Each cell contains the attribution value for that feature-node pair.
        """
        all_attributions = []

        for nid in node_ids:
            attr = self.compute_attributions(
                node_features=features_batch,
                node_id=nid,
                incidence_matrix=incidence_matrix,
                node_types=node_types,
                edge_index=edge_index,
                edge_types=edge_types,
                timestamps=timestamps,
                prediction_type=prediction_type,
            )
            all_attributions.append(attr)

        df = pd.DataFrame(all_attributions, index=node_ids)
        df.index.name = 'node_id'

        # Cache the result
        self._batch_cache = df

        return df

    def get_top_features(
        self,
        node_id: int,
        attributions: Optional[Dict[str, float]] = None,
        k: int = 5,
        node_features: Optional[torch.Tensor] = None,
        incidence_matrix: Optional[torch.Tensor] = None,
        node_types: Optional[List[str]] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_types: Optional[List[str]] = None,
        timestamps: Optional[torch.Tensor] = None,
        prediction_type: str = 'criticality',
    ) -> List[Tuple[str, float]]:
        """
        Get the top-k most important features for a given node.

        If pre-computed attributions are provided, uses those directly.
        Otherwise checks the batch cache, and falls back to computing
        attributions on the fly.

        Args:
            node_id: Index of the target node.
            attributions: Optional pre-computed attribution dict.
            k: Number of top features to return.
            node_features: Node feature tensor (required if attributions is None).
            incidence_matrix: Incidence matrix (required if attributions is None).
            node_types: Type labels (required if attributions is None).
            edge_index: Edge connectivity (required if attributions is None).
            edge_types: Edge type labels (required if attributions is None).
            timestamps: Temporal info (required if attributions is None).
            prediction_type: Output head to explain.

        Returns:
            List of (feature_name, importance_score) tuples sorted by
            absolute importance in descending order, limited to top k.
        """
        if attributions is None:
            # Check batch cache first
            if self._batch_cache is not None and node_id in self._batch_cache.index:
                attributions = self._batch_cache.loc[node_id].to_dict()
            elif node_features is not None:
                # Compute on the fly
                attributions = self.compute_attributions(
                    node_features=node_features,
                    node_id=node_id,
                    incidence_matrix=incidence_matrix,
                    node_types=node_types,
                    edge_index=edge_index,
                    edge_types=edge_types,
                    timestamps=timestamps,
                    prediction_type=prediction_type,
                )
            else:
                raise ValueError(
                    "Either provide pre-computed attributions or pass "
                    "node_features and model inputs for on-the-fly computation."
                )

        sorted_features = sorted(
            attributions.items(), key=lambda x: abs(x[1]), reverse=True
        )
        return sorted_features[:k]

    def feature_importance_summary(
        self,
        batch_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute aggregate feature importance statistics across all nodes.

        Args:
            batch_df: DataFrame from compute_batch_attributions(). If None,
                uses the cached batch result.

        Returns:
            DataFrame with features as index and columns:
                - mean_attribution: Mean attribution across nodes.
                - std_attribution: Standard deviation.
                - abs_mean: Mean of absolute attributions.
                - positive_count: Number of nodes with positive attribution.
                - negative_count: Number of nodes with negative attribution.
        """
        if batch_df is None:
            batch_df = self._batch_cache

        if batch_df is None:
            raise ValueError(
                "No batch attribution data available. "
                "Call compute_batch_attributions() first."
            )

        summary = pd.DataFrame({
            'mean_attribution': batch_df.mean(),
            'std_attribution': batch_df.std(),
            'abs_mean': batch_df.abs().mean(),
            'positive_count': (batch_df > 0).sum(),
            'negative_count': (batch_df < 0).sum(),
        })
        summary = summary.sort_values('abs_mean', ascending=False)
        summary.index.name = 'feature'

        return summary

    def convergence_check(
        self,
        node_features: torch.Tensor,
        node_id: int,
        incidence_matrix: torch.Tensor,
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        prediction_type: str = 'criticality',
    ) -> Dict[str, Any]:
        """
        Check whether integrated gradients have converged.

        The completeness axiom states that attributions should sum to
        the difference between the prediction and baseline prediction.
        This method checks how closely this holds.

        Args:
            node_features: Node feature tensor.
            node_id: Target node index.
            incidence_matrix: Incidence matrix tensor.
            node_types: Type labels.
            edge_index: Edge connectivity.
            edge_types: Edge type labels.
            timestamps: Temporal information.
            prediction_type: Output head to explain.

        Returns:
            Dictionary with convergence metrics:
                - attribution_sum: Sum of all feature attributions.
                - prediction_delta: f(x) - f(baseline).
                - convergence_error: Absolute difference.
                - is_converged: True if error < 0.01.
        """
        pred_key = self._get_prediction_key(prediction_type)

        node_features = node_features.to(self.device)
        incidence_matrix = incidence_matrix.to(self.device)
        edge_index = edge_index.to(self.device)
        timestamps = timestamps.to(self.device)

        # f(x) prediction
        with torch.no_grad():
            output_full = self.model(
                node_features, incidence_matrix, node_types,
                edge_index, edge_types, timestamps
            )
            pred_full = output_full[pred_key]
            if pred_full.dim() > 0:
                f_x = pred_full[node_id].item()
            else:
                f_x = pred_full.item()

        # f(baseline) prediction
        baseline = torch.zeros_like(node_features).to(self.device)
        with torch.no_grad():
            output_base = self.model(
                baseline, incidence_matrix, node_types,
                edge_index, edge_types, timestamps
            )
            pred_base = output_base[pred_key]
            if pred_base.dim() > 0:
                f_baseline = pred_base[node_id].item()
            else:
                f_baseline = pred_base.item()

        # Compute attributions
        attributions = self.compute_attributions(
            node_features=node_features,
            node_id=node_id,
            incidence_matrix=incidence_matrix,
            node_types=node_types,
            edge_index=edge_index,
            edge_types=edge_types,
            timestamps=timestamps,
            prediction_type=prediction_type,
        )

        attr_sum = sum(attributions.values())
        pred_delta = f_x - f_baseline
        conv_error = abs(attr_sum - pred_delta)

        return {
            'attribution_sum': attr_sum,
            'prediction_delta': pred_delta,
            'convergence_error': conv_error,
            'is_converged': conv_error < 0.01,
        }

    def summary_text(
        self,
        attributions: Dict[str, float],
        node_id: int,
        top_k: int = 5,
    ) -> str:
        """
        Generate a human-readable text summary of feature attributions.

        Args:
            attributions: Feature attribution dictionary.
            node_id: Node index for the summary header.
            top_k: Number of top features to display.

        Returns:
            Formatted multi-line string.
        """
        sorted_features = sorted(
            attributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        lines = [
            f"=== Feature Attribution for Node {node_id} ===",
            f"Total features: {len(attributions)}",
            f"Attribution sum: {sum(attributions.values()):.4f}",
            "",
            f"Top {min(top_k, len(sorted_features))} Features:",
        ]

        for i, (fname, score) in enumerate(sorted_features[:top_k]):
            direction = "+" if score >= 0 else "-"
            bar_len = int(min(abs(score) * 30, 30))
            bar = "#" * bar_len
            lines.append(f"  {i+1}. {fname:<25s} {direction}{abs(score):.4f} |{bar}")

        # Positive vs negative contributors
        positive = {k: v for k, v in attributions.items() if v > 0}
        negative = {k: v for k, v in attributions.items() if v < 0}

        lines.extend([
            "",
            f"Positive contributors: {len(positive)} features "
            f"(sum={sum(positive.values()):.4f})",
            f"Negative contributors: {len(negative)} features "
            f"(sum={sum(negative.values()):.4f})",
        ])

        return "\n".join(lines)


if __name__ == "__main__":
    print("Feature Attribution Analyzer Module")
    print("=" * 50)
    print()
    print("Primary method: Integrated Gradients")
    print()
    print("Formula:")
    print("  IG_i(x) = (x_i - x'_i) * integral_{alpha=0}^{1}")
    print("            (dF(x' + alpha*(x - x')) / dx_i) d_alpha")
    print()
    print("Features:")
    print("  - Per-feature attribution via integrated gradients")
    print("  - Batch computation with DataFrame output")
    print("  - Top-k feature extraction")
    print("  - Convergence checking (completeness axiom)")
    print("  - Aggregate feature importance summaries")
    print()
    print("Usage example:")
    print("  analyzer = FeatureAttributionAnalyzer(model, feature_names)")
    print("  attr = analyzer.compute_attributions(features, node_id=42, ...)")
    print("  top = analyzer.get_top_features(node_id=42, attributions=attr, k=5)")
    print()
    print("Module ready for integration with HT-HGNN v2.0.")
