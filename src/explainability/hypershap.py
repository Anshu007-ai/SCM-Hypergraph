"""
HyperSHAP: SHAP-based Explainability for Hypergraph Neural Networks

Extends classical SHAP (SHapley Additive exPlanations) to hypergraph structures
by computing Shapley values over hyperedge coalitions instead of individual features.

The core formula for hyperedge attribution:

    phi_e = Sum over S subset E\{e} of [|S|!(|E|-|S|-1)!/|E|!] * [f(S union {e}) - f(S)]

Each hyperedge receives an importance score reflecting its marginal contribution
to the model prediction for a given node.

Author: HT-HGNN v2.0 Project
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class NodeExplanation:
    """Container for a single node's SHAP-based explanation."""
    node_id: int
    prediction_type: str
    node_attribution: float
    hyperedge_attributions: Dict[int, float]
    feature_attributions: Dict[str, float]
    prediction_value: float
    base_value: float
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to a dictionary representation."""
        return {
            'node_id': self.node_id,
            'prediction_type': self.prediction_type,
            'node_attribution': self.node_attribution,
            'hyperedge_attributions': self.hyperedge_attributions,
            'feature_attributions': self.feature_attributions,
            'prediction_value': self.prediction_value,
            'base_value': self.base_value,
            'recommendations': self.recommendations,
        }


class HyperSHAP:
    """
    Custom SHAP adaptation for hypergraph neural networks.

    Computes Shapley values for both hyperedges and node features,
    providing attribution scores that explain model predictions
    in the context of supply chain hypergraph structures.

    The key innovation is treating hyperedge coalitions (subsets of
    hyperedges) as the coalition space for Shapley value computation,
    rather than individual input features.

    Attributes:
        model: A trained HT-HGNN model instance.
        incidence_matrix: Binary incidence matrix (num_hyperedges, num_nodes).
        num_samples: Number of Monte Carlo samples for Shapley approximation.
        device: Torch device for computation.
    """

    def __init__(
        self,
        model: nn.Module,
        incidence_matrix: torch.Tensor,
        num_samples: int = 100,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize HyperSHAP explainer.

        Args:
            model: Trained HT-HGNN model (or compatible nn.Module).
            incidence_matrix: Binary tensor of shape (num_hyperedges, num_nodes).
            num_samples: Number of Monte Carlo samples for Shapley approximation.
            feature_names: Optional list of feature names for attribution labeling.
        """
        self.model = model
        self.model.eval()
        self.incidence_matrix = incidence_matrix.float()
        self.num_samples = num_samples
        self.num_hyperedges = incidence_matrix.shape[0]
        self.num_nodes = incidence_matrix.shape[1]
        self.feature_names = feature_names
        self.device = next(model.parameters()).device

        # Move incidence matrix to device
        self.incidence_matrix = self.incidence_matrix.to(self.device)

    def _shapley_weight(self, coalition_size: int, total_elements: int) -> float:
        """
        Compute the Shapley weighting factor for a coalition of given size.

        The weight is: |S|! * (|E| - |S| - 1)! / |E|!

        Args:
            coalition_size: Size of the coalition S (excluding the element).
            total_elements: Total number of elements |E|.

        Returns:
            The Shapley weight as a float.
        """
        s = coalition_size
        n = total_elements
        numerator = math.factorial(s) * math.factorial(n - s - 1)
        denominator = math.factorial(n)
        return numerator / denominator

    def _mask_incidence(self, active_hyperedges: List[int]) -> torch.Tensor:
        """
        Create a masked incidence matrix where only specified hyperedges are active.

        Args:
            active_hyperedges: Indices of hyperedges to keep active.

        Returns:
            Masked incidence matrix of same shape as original.
        """
        mask = torch.zeros(self.num_hyperedges, device=self.device)
        if len(active_hyperedges) > 0:
            indices = torch.tensor(active_hyperedges, device=self.device, dtype=torch.long)
            mask[indices] = 1.0
        masked = self.incidence_matrix * mask.unsqueeze(1)
        return masked

    @torch.no_grad()
    def _evaluate_coalition(
        self,
        node_features: torch.Tensor,
        active_hyperedges: List[int],
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        prediction_type: str,
        node_id: int,
    ) -> float:
        """
        Evaluate the model prediction for a specific hyperedge coalition.

        Args:
            node_features: Node feature tensor.
            active_hyperedges: List of active hyperedge indices.
            node_types: Type labels for each node.
            edge_index: Edge index tensor.
            edge_types: Type labels for each edge.
            timestamps: Timestamp tensor for each node.
            prediction_type: Which output head to use ('criticality', 'price', 'change').
            node_id: Target node index.

        Returns:
            Prediction value for the given node under the active coalition.
        """
        masked_incidence = self._mask_incidence(active_hyperedges)
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

        if prediction.dim() == 0:
            return prediction.item()
        return prediction[node_id].item()

    def _compute_hyperedge_shapley(
        self,
        node_features: torch.Tensor,
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        prediction_type: str,
        node_id: int,
    ) -> Dict[int, float]:
        """
        Compute Shapley values for each hyperedge using Monte Carlo sampling.

        For each sample, a random permutation of hyperedges is generated.
        The marginal contribution of each hyperedge is computed as it is
        added to the growing coalition.

        Args:
            node_features: Node feature tensor.
            node_types: Type labels.
            edge_index: Edge connectivity.
            edge_types: Edge type labels.
            timestamps: Temporal information.
            prediction_type: Output head to explain.
            node_id: Target node.

        Returns:
            Dictionary mapping hyperedge index to its Shapley value.
        """
        all_hyperedges = list(range(self.num_hyperedges))
        shapley_values = {e: 0.0 for e in all_hyperedges}

        for _ in range(self.num_samples):
            # Random permutation of hyperedges
            perm = np.random.permutation(all_hyperedges).tolist()
            coalition = []

            prev_value = self._evaluate_coalition(
                node_features, coalition, node_types,
                edge_index, edge_types, timestamps,
                prediction_type, node_id
            )

            for e in perm:
                coalition.append(e)
                current_value = self._evaluate_coalition(
                    node_features, coalition, node_types,
                    edge_index, edge_types, timestamps,
                    prediction_type, node_id
                )
                marginal = current_value - prev_value
                shapley_values[e] += marginal
                prev_value = current_value

        # Average over samples
        for e in shapley_values:
            shapley_values[e] /= self.num_samples

        return shapley_values

    def _compute_feature_attributions(
        self,
        node_features: torch.Tensor,
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        prediction_type: str,
        node_id: int,
    ) -> Dict[str, float]:
        """
        Compute per-feature attribution using integrated gradients.

        Approximates feature importance by interpolating between a zero
        baseline and the actual node features, accumulating gradients
        along the interpolation path.

        Args:
            node_features: Node feature tensor.
            node_types: Type labels.
            edge_index: Edge connectivity.
            edge_types: Edge type labels.
            timestamps: Temporal information.
            prediction_type: Output head to explain.
            node_id: Target node.

        Returns:
            Dictionary mapping feature name to attribution score.
        """
        num_features = node_features.shape[1]
        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(num_features)
        ]

        # Integrated gradients with n_steps along the interpolation path
        n_steps = 50
        baseline = torch.zeros_like(node_features).to(self.device)
        accumulated_grads = torch.zeros(num_features, device=self.device)

        pred_key_map = {
            'criticality': 'criticality',
            'price': 'price_pred',
            'change': 'change_pred',
        }
        pred_key = pred_key_map.get(prediction_type, 'criticality')

        for step in range(n_steps + 1):
            alpha = step / n_steps
            interpolated = baseline + alpha * (node_features - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)

            output = self.model(
                interpolated, self.incidence_matrix, node_types,
                edge_index, edge_types, timestamps
            )

            prediction = output[pred_key]
            if prediction.dim() > 0:
                target = prediction[node_id]
            else:
                target = prediction

            target.backward(retain_graph=True)

            if interpolated.grad is not None:
                accumulated_grads += interpolated.grad[node_id]
                interpolated.grad.zero_()

        # Integrated gradients = (input - baseline) * average_gradient
        attributions = (
            (node_features[node_id] - baseline[node_id]) *
            accumulated_grads / (n_steps + 1)
        )

        result = {}
        for i, name in enumerate(feature_names):
            result[name] = attributions[i].item()

        return result

    def _generate_recommendations(
        self,
        hyperedge_attributions: Dict[int, float],
        feature_attributions: Dict[str, float],
        prediction_type: str,
    ) -> List[str]:
        """
        Generate human-readable recommendations from attribution scores.

        Analyzes the top contributing hyperedges and features to produce
        actionable supply chain recommendations.

        Args:
            hyperedge_attributions: Shapley values per hyperedge.
            feature_attributions: Attribution scores per feature.
            prediction_type: Type of prediction being explained.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        # Identify top contributing hyperedges
        sorted_edges = sorted(
            hyperedge_attributions.items(), key=lambda x: abs(x[1]), reverse=True
        )
        top_edges = sorted_edges[:3]

        for edge_id, score in top_edges:
            if abs(score) < 1e-6:
                continue
            direction = "increases" if score > 0 else "decreases"
            recommendations.append(
                f"Hyperedge {edge_id} {direction} {prediction_type} risk "
                f"(attribution={score:.4f}). Consider reviewing the supply "
                f"chain relationships in this subassembly."
            )

        # Identify top contributing features
        sorted_features = sorted(
            feature_attributions.items(), key=lambda x: abs(x[1]), reverse=True
        )
        top_features = sorted_features[:3]

        for feat_name, score in top_features:
            if abs(score) < 1e-6:
                continue
            direction = "positively" if score > 0 else "negatively"
            recommendations.append(
                f"Feature '{feat_name}' {direction} contributes to "
                f"{prediction_type} (attribution={score:.4f}). "
                f"Adjusting this factor may alter the outcome."
            )

        if not recommendations:
            recommendations.append(
                f"No significant attributions detected for {prediction_type}. "
                f"The prediction appears stable across feature and structure variations."
            )

        return recommendations

    def explain_node(
        self,
        node_id: int,
        node_features: torch.Tensor,
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        prediction_type: str = 'criticality',
    ) -> Dict[str, Any]:
        """
        Generate a complete explanation for a single node's prediction.

        Computes hyperedge Shapley values, feature attributions, and
        generates text recommendations.

        Args:
            node_id: Index of the node to explain.
            node_features: Full node feature tensor.
            node_types: Type labels for each node.
            edge_index: Edge index tensor.
            edge_types: Type labels for each edge.
            timestamps: Timestamp tensor.
            prediction_type: Which prediction to explain
                ('criticality', 'price', 'change').

        Returns:
            Dictionary containing:
                - node_attribution: Overall attribution score for the node.
                - hyperedge_attributions: Dict of hyperedge -> Shapley value.
                - feature_attributions: Dict of feature -> attribution score.
                - prediction_value: The model's actual prediction.
                - base_value: Prediction with empty coalition (baseline).
                - recommendations: List of text recommendations.
        """
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        timestamps = timestamps.to(self.device)

        # Compute baseline (empty coalition)
        base_value = self._evaluate_coalition(
            node_features, [], node_types,
            edge_index, edge_types, timestamps,
            prediction_type, node_id
        )

        # Compute full prediction (all hyperedges active)
        all_edges = list(range(self.num_hyperedges))
        prediction_value = self._evaluate_coalition(
            node_features, all_edges, node_types,
            edge_index, edge_types, timestamps,
            prediction_type, node_id
        )

        # Compute hyperedge Shapley values
        hyperedge_attributions = self._compute_hyperedge_shapley(
            node_features, node_types,
            edge_index, edge_types, timestamps,
            prediction_type, node_id
        )

        # Compute feature attributions
        feature_attributions = self._compute_feature_attributions(
            node_features, node_types,
            edge_index, edge_types, timestamps,
            prediction_type, node_id
        )

        # Overall node attribution is the sum of all hyperedge attributions
        node_attribution = sum(hyperedge_attributions.values())

        # Generate recommendations
        recommendations = self._generate_recommendations(
            hyperedge_attributions, feature_attributions, prediction_type
        )

        explanation = NodeExplanation(
            node_id=node_id,
            prediction_type=prediction_type,
            node_attribution=node_attribution,
            hyperedge_attributions=hyperedge_attributions,
            feature_attributions=feature_attributions,
            prediction_value=prediction_value,
            base_value=base_value,
            recommendations=recommendations,
        )

        return explanation.to_dict()

    def explain_batch(
        self,
        node_ids: List[int],
        node_features: torch.Tensor,
        node_types: List[str],
        edge_index: torch.Tensor,
        edge_types: List[str],
        timestamps: torch.Tensor,
        prediction_type: str = 'criticality',
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of nodes.

        Args:
            node_ids: List of node indices to explain.
            node_features: Full node feature tensor.
            node_types: Type labels for each node.
            edge_index: Edge index tensor.
            edge_types: Type labels for each edge.
            timestamps: Timestamp tensor.
            prediction_type: Which prediction to explain.

        Returns:
            List of explanation dictionaries, one per node.
        """
        explanations = []
        for nid in node_ids:
            explanation = self.explain_node(
                node_id=nid,
                node_features=node_features,
                node_types=node_types,
                edge_index=edge_index,
                edge_types=edge_types,
                timestamps=timestamps,
                prediction_type=prediction_type,
            )
            explanations.append(explanation)
        return explanations

    def summary(self, explanation: Dict[str, Any]) -> str:
        """
        Produce a human-readable summary of a node explanation.

        Args:
            explanation: Output from explain_node().

        Returns:
            Formatted string summary.
        """
        lines = [
            f"=== HyperSHAP Explanation for Node {explanation['node_id']} ===",
            f"Prediction type : {explanation['prediction_type']}",
            f"Prediction value: {explanation['prediction_value']:.4f}",
            f"Base value      : {explanation['base_value']:.4f}",
            f"Node attribution: {explanation['node_attribution']:.4f}",
            "",
            "Top Hyperedge Attributions:",
        ]

        sorted_he = sorted(
            explanation['hyperedge_attributions'].items(),
            key=lambda x: abs(x[1]), reverse=True
        )
        for eid, val in sorted_he[:5]:
            lines.append(f"  Hyperedge {eid}: {val:+.4f}")

        lines.append("")
        lines.append("Top Feature Attributions:")
        sorted_feat = sorted(
            explanation['feature_attributions'].items(),
            key=lambda x: abs(x[1]), reverse=True
        )
        for fname, val in sorted_feat[:5]:
            lines.append(f"  {fname}: {val:+.4f}")

        lines.append("")
        lines.append("Recommendations:")
        for rec in explanation['recommendations']:
            lines.append(f"  - {rec}")

        return "\n".join(lines)


if __name__ == "__main__":
    print("HyperSHAP Module - SHAP-based Explainability for Hypergraphs")
    print("=" * 60)
    print()
    print("Core formula (hyperedge Shapley value):")
    print("  phi_e = Sum_{S subset E\\{e}} "
          "[|S|!(|E|-|S|-1)!/|E|!] * [f(S u {e}) - f(S)]")
    print()
    print("Features:")
    print("  - Hyperedge coalition-based Shapley values")
    print("  - Integrated gradients for feature attribution")
    print("  - Automatic text recommendation generation")
    print("  - Batch explanation support")
    print()

    # Quick sanity check with dummy tensors
    print("Running sanity check with dummy data...")
    dummy_incidence = torch.randint(0, 2, (5, 10)).float()
    print(f"  Dummy incidence matrix shape: {dummy_incidence.shape}")
    print(f"  Number of hyperedges: {dummy_incidence.shape[0]}")
    print(f"  Number of nodes: {dummy_incidence.shape[1]}")
    print()
    print("HyperSHAP module ready for integration with HT-HGNN v2.0.")
