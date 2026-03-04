"""
Hyperedge-Aware Evaluation Metrics

Standard classification/regression metrics ignore the higher-order structure
encoded in hyperedges.  This module provides metrics that are aware of the
hypergraph incidence matrix:

  - Hyperedge accuracy   -- per-hyperedge prediction correctness
  - Cascade F1           -- overlap-based F1 for predicted vs actual cascades
  - Group risk consistency -- do nodes sharing a hyperedge receive consistent
                             risk scores?
  - Coverage@K           -- are the top-K ranked nodes actually critical?
  - NDCG                 -- normalised discounted cumulative gain for rankings
"""

import numpy as np
from typing import Dict, List, Optional, Set, Union

from src.hypergraph.hypergraph import Hypergraph


class HypergraphMetrics:
    """
    A collection of hyperedge-aware evaluation metrics for the HT-HGNN v2.0
    supply-chain risk prediction pipeline.

    All methods are static or class-level so the class can be used as a
    lightweight namespace, but an instance can also be kept to conveniently
    call :meth:`summary`.
    """

    # ------------------------------------------------------------------
    # Hyperedge accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hyperedge_accuracy(
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        incidence_matrix: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute accuracy at the *hyperedge* level.

        A hyperedge is predicted correctly if the majority of its member
        nodes are individually classified correctly (above ``threshold``).

        Parameters
        ----------
        predictions : np.ndarray
            Per-node predicted risk scores, shape ``(n_nodes,)``.
        ground_truth : np.ndarray
            Per-node ground-truth binary labels, shape ``(n_nodes,)``.
        incidence_matrix : np.ndarray
            Binary incidence matrix, shape ``(n_hyperedges, n_nodes)``.
        threshold : float
            Decision boundary for converting continuous predictions to
            binary labels.

        Returns
        -------
        float
            Fraction of hyperedges where the majority of member nodes
            are classified correctly.
        """
        pred_binary = (np.asarray(predictions) >= threshold).astype(int)
        gt_binary = (np.asarray(ground_truth) >= threshold).astype(int)
        H = np.asarray(incidence_matrix)

        n_hyperedges = H.shape[0]
        if n_hyperedges == 0:
            return 0.0

        correct = 0
        for he_idx in range(n_hyperedges):
            member_mask = H[he_idx] > 0
            member_pred = pred_binary[member_mask]
            member_gt = gt_binary[member_mask]
            if len(member_gt) == 0:
                continue
            accuracy = np.mean(member_pred == member_gt)
            if accuracy > 0.5:
                correct += 1

        return correct / n_hyperedges

    # ------------------------------------------------------------------
    # Cascade F1
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cascade_f1(
        predicted_cascade: Set[str],
        actual_cascade: Set[str],
    ) -> float:
        """
        Compute an F1-style overlap score between predicted and actual
        disruption cascades (sets of affected node/hyperedge IDs).

        Parameters
        ----------
        predicted_cascade : set of str
            Predicted affected entities.
        actual_cascade : set of str
            Ground truth affected entities.

        Returns
        -------
        float
            F1 score in [0, 1].  Returns 0 if both sets are empty.
        """
        predicted = set(predicted_cascade)
        actual = set(actual_cascade)

        if not predicted and not actual:
            return 0.0

        tp = len(predicted & actual)
        fp = len(predicted - actual)
        fn = len(actual - predicted)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # ------------------------------------------------------------------
    # Group risk consistency
    # ------------------------------------------------------------------

    @staticmethod
    def compute_group_risk_consistency(
        node_risks: Dict[str, float],
        hyperedge_membership: Dict[str, Set[str]],
    ) -> float:
        """
        Measure how consistent node-level risk scores are *within*
        hyperedges.

        Lower intra-hyperedge variance means higher consistency.
        The returned value is ``1 - mean(normalised_variance)`` so that
        1.0 = perfectly consistent, 0.0 = maximally inconsistent.

        Parameters
        ----------
        node_risks : dict
            Mapping ``node_id -> risk_score``.
        hyperedge_membership : dict
            Mapping ``hyperedge_id -> {node_id, ...}``.

        Returns
        -------
        float
            Consistency score in [0, 1].
        """
        variances: List[float] = []

        for _hid, members in hyperedge_membership.items():
            scores = [
                node_risks[nid]
                for nid in members
                if nid in node_risks
            ]
            if len(scores) < 2:
                continue
            var = float(np.var(scores))
            # Normalise: max possible variance for scores in [0,1] is 0.25
            normalised = min(var / 0.25, 1.0)
            variances.append(normalised)

        if not variances:
            return 1.0  # trivially consistent

        return 1.0 - float(np.mean(variances))

    # ------------------------------------------------------------------
    # Coverage@K
    # ------------------------------------------------------------------

    @staticmethod
    def compute_coverage_at_k(
        ranked_nodes: List[str],
        actual_critical: Set[str],
        k: int = 10,
    ) -> float:
        """
        Compute the proportion of truly critical nodes captured in the
        top-K of a ranked list.

        Parameters
        ----------
        ranked_nodes : list of str
            Node IDs ordered by predicted risk (highest first).
        actual_critical : set of str
            Ground-truth critical node IDs.
        k : int
            Cut-off rank.

        Returns
        -------
        float
            Coverage score in [0, 1].
        """
        if not actual_critical:
            return 0.0
        top_k = set(ranked_nodes[:k])
        return len(top_k & actual_critical) / len(actual_critical)

    # ------------------------------------------------------------------
    # NDCG
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ndcg(
        ranked_list: List[str],
        relevance_scores: Dict[str, float],
        k: Optional[int] = None,
    ) -> float:
        """
        Compute Normalised Discounted Cumulative Gain (NDCG) for a ranked
        list of items with known relevance scores.

        Parameters
        ----------
        ranked_list : list of str
            Item IDs in predicted order (best first).
        relevance_scores : dict
            Mapping ``item_id -> relevance``.  Items not present are
            assumed to have zero relevance.
        k : int or None
            Evaluate at rank *k*; ``None`` uses the full list length.

        Returns
        -------
        float
            NDCG score in [0, 1].
        """
        if not ranked_list or not relevance_scores:
            return 0.0

        cutoff = k if k is not None else len(ranked_list)
        cutoff = min(cutoff, len(ranked_list))

        # DCG
        dcg = 0.0
        for i in range(cutoff):
            rel = relevance_scores.get(ranked_list[i], 0.0)
            dcg += rel / np.log2(i + 2)  # i+2 because rank starts at 1

        # Ideal DCG (IDCG)
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:cutoff]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += rel / np.log2(i + 2)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @classmethod
    def summary(
        cls,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        hypergraph: Hypergraph,
        predicted_cascade: Optional[Set[str]] = None,
        actual_cascade: Optional[Set[str]] = None,
        node_risks: Optional[Dict[str, float]] = None,
        ranked_nodes: Optional[List[str]] = None,
        actual_critical: Optional[Set[str]] = None,
        relevance_scores: Optional[Dict[str, float]] = None,
        k: int = 10,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute all available metrics in one call and return them as a
        dictionary.

        Parameters that are ``None`` cause the corresponding metric to be
        skipped (its key will still appear with a value of ``None``).

        Parameters
        ----------
        predictions : np.ndarray
            Per-node predicted risk scores.
        ground_truth : np.ndarray
            Per-node ground-truth labels.
        hypergraph : Hypergraph
            The hypergraph instance.
        predicted_cascade : set or None
        actual_cascade : set or None
        node_risks : dict or None
        ranked_nodes : list or None
        actual_critical : set or None
        relevance_scores : dict or None
        k : int
        threshold : float

        Returns
        -------
        dict
        """
        incidence_matrix, _, _ = hypergraph.get_incidence_matrix()

        results: Dict[str, Optional[float]] = {}

        # Hyperedge accuracy (always computable from predictions + incidence)
        results["hyperedge_accuracy"] = cls.compute_hyperedge_accuracy(
            predictions, ground_truth, incidence_matrix, threshold
        )

        # Cascade F1
        if predicted_cascade is not None and actual_cascade is not None:
            results["cascade_f1"] = cls.compute_cascade_f1(
                predicted_cascade, actual_cascade
            )
        else:
            results["cascade_f1"] = None

        # Group risk consistency
        if node_risks is not None:
            membership = {
                hid: he.nodes for hid, he in hypergraph.hyperedges.items()
            }
            results["group_risk_consistency"] = cls.compute_group_risk_consistency(
                node_risks, membership
            )
        else:
            results["group_risk_consistency"] = None

        # Coverage@K
        if ranked_nodes is not None and actual_critical is not None:
            results["coverage_at_k"] = cls.compute_coverage_at_k(
                ranked_nodes, actual_critical, k
            )
        else:
            results["coverage_at_k"] = None

        # NDCG
        if ranked_nodes is not None and relevance_scores is not None:
            results["ndcg"] = cls.compute_ndcg(ranked_nodes, relevance_scores, k)
        else:
            results["ndcg"] = None

        return results


if __name__ == "__main__":
    # ---- Quick smoke test ----
    from src.hypergraph.hypergraph import Hypergraph, HypergraphNode, HypergraphEdge

    hg = Hypergraph()
    for i in range(8):
        hg.add_node(HypergraphNode(
            node_id=f"n{i}", node_type="supplier", tier=0,
            lead_time=5.0, reliability=0.9, substitutability=0.5, cost=100.0,
        ))
    hg.add_hyperedge(HypergraphEdge(hyperedge_id="he0", nodes={f"n{i}" for i in range(4)}))
    hg.add_hyperedge(HypergraphEdge(hyperedge_id="he1", nodes={f"n{i}" for i in range(4, 8)}))

    predictions = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.3, 0.15])
    ground_truth = np.array([1, 1, 1, 0, 0, 0, 0, 0])

    metrics = HypergraphMetrics()

    inc, _, _ = hg.get_incidence_matrix()
    he_acc = metrics.compute_hyperedge_accuracy(predictions, ground_truth, inc)
    print(f"Hyperedge accuracy: {he_acc:.4f}")

    cf1 = metrics.compute_cascade_f1({"n0", "n1", "n2"}, {"n0", "n1", "n3"})
    print(f"Cascade F1: {cf1:.4f}")

    ranks = [f"n{i}" for i in range(8)]
    crit = {"n0", "n1"}
    cov = metrics.compute_coverage_at_k(ranks, crit, k=3)
    print(f"Coverage@3: {cov:.4f}")

    rels = {f"n{i}": float(8 - i) for i in range(8)}
    ndcg = metrics.compute_ndcg(ranks, rels, k=5)
    print(f"NDCG@5: {ndcg:.4f}")

    node_risks = {f"n{i}": predictions[i] for i in range(8)}
    membership = {hid: he.nodes for hid, he in hg.hyperedges.items()}
    consistency = metrics.compute_group_risk_consistency(node_risks, membership)
    print(f"Group risk consistency: {consistency:.4f}")

    print("\n--- Full summary ---")
    s = metrics.summary(
        predictions, ground_truth, hg,
        predicted_cascade={"n0", "n1", "n2"},
        actual_cascade={"n0", "n1", "n3"},
        node_risks=node_risks,
        ranked_nodes=ranks,
        actual_critical=crit,
        relevance_scores=rels,
        k=5,
    )
    for key, val in s.items():
        print(f"  {key}: {val}")

    print("\nHypergraph metrics module ready for integration")
