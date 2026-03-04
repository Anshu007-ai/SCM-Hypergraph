"""
Cascade Prediction Evaluation

Evaluates the quality of disruption cascade predictions in the supply chain.
A cascade is a sequence of disruption events that propagate through the
hypergraph from an initial shock node through downstream dependencies.

Key metrics:
  - Cascade depth MAE       -- how accurately we predict how deep cascades go
  - Cascade spread accuracy -- how accurately we predict the number of nodes hit
  - Timing accuracy         -- how accurately we predict when disruptions arrive
  - Path overlap            -- structural similarity between predicted and actual
                               propagation paths
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict


class CascadeEvaluator:
    """
    Evaluate predicted disruption cascades against ground-truth cascades.

    Each cascade is represented as a dictionary with the schema::

        {
            "cascade_id": str,
            "source_node": str,
            "affected_nodes": [str, ...],
            "depth": int,
            "timestamps": {node_id: float, ...},   # time each node is hit
            "path": [(src, dst), ...],              # directed propagation edges
        }

    Fields that are absent are treated as unknown and the corresponding
    metric is skipped gracefully.

    Parameters
    ----------
    ground_truth_cascades : list of dict
        Ground-truth cascade records.
    predicted_cascades : list of dict
        Predicted cascade records. Must match ground-truth by ``cascade_id``.
    """

    def __init__(
        self,
        ground_truth_cascades: List[Dict],
        predicted_cascades: List[Dict],
    ):
        self.ground_truth = {c["cascade_id"]: c for c in ground_truth_cascades}
        self.predicted = {c["cascade_id"]: c for c in predicted_cascades}
        self._common_ids = sorted(
            set(self.ground_truth.keys()) & set(self.predicted.keys())
        )

    # ------------------------------------------------------------------
    # Cascade depth MAE
    # ------------------------------------------------------------------

    def compute_cascade_depth_mae(self) -> float:
        """
        Mean Absolute Error of cascade depth prediction.

        .. math::

            \\text{MAE}_{\\text{depth}} =
            \\frac{1}{N} \\sum_{i=1}^{N}
            \\lvert d_{\\text{pred},i} - d_{\\text{true},i} \\rvert

        Returns
        -------
        float
            MAE of cascade depth.  Returns ``float('nan')`` if no common
            cascades with depth information exist.
        """
        errors: List[float] = []
        for cid in self._common_ids:
            gt = self.ground_truth[cid]
            pr = self.predicted[cid]
            if "depth" in gt and "depth" in pr:
                errors.append(abs(float(pr["depth"]) - float(gt["depth"])))

        if not errors:
            return float("nan")
        return float(np.mean(errors))

    # ------------------------------------------------------------------
    # Cascade spread accuracy
    # ------------------------------------------------------------------

    def compute_cascade_spread_accuracy(self) -> float:
        """
        Measure how accurately the spread (number of affected nodes) of
        each cascade is predicted.

        Computes ``1 - MAPE`` (clipped to [0, 1]) where MAPE is the mean
        absolute percentage error of the cascade sizes.

        Returns
        -------
        float
            Spread accuracy in [0, 1].
        """
        apes: List[float] = []
        for cid in self._common_ids:
            gt = self.ground_truth[cid]
            pr = self.predicted[cid]
            if "affected_nodes" in gt and "affected_nodes" in pr:
                gt_size = len(gt["affected_nodes"])
                pr_size = len(pr["affected_nodes"])
                if gt_size > 0:
                    apes.append(abs(pr_size - gt_size) / gt_size)

        if not apes:
            return float("nan")
        mape = float(np.mean(apes))
        return max(0.0, 1.0 - mape)

    # ------------------------------------------------------------------
    # Timing accuracy
    # ------------------------------------------------------------------

    def compute_timing_accuracy(self) -> float:
        """
        Evaluate how well the model predicts *when* each node in a cascade
        is affected.

        For every node that appears in both predicted and ground-truth
        cascades, computes the absolute difference between predicted and
        actual hit-times.  The score is ``1 / (1 + mean_abs_error)`` so
        that a perfect prediction yields 1.0.

        Returns
        -------
        float
            Timing accuracy in (0, 1].
        """
        timing_errors: List[float] = []

        for cid in self._common_ids:
            gt = self.ground_truth[cid]
            pr = self.predicted[cid]
            gt_ts = gt.get("timestamps", {})
            pr_ts = pr.get("timestamps", {})
            common_nodes = set(gt_ts.keys()) & set(pr_ts.keys())
            for node in common_nodes:
                timing_errors.append(abs(float(pr_ts[node]) - float(gt_ts[node])))

        if not timing_errors:
            return float("nan")
        mae = float(np.mean(timing_errors))
        return 1.0 / (1.0 + mae)

    # ------------------------------------------------------------------
    # Path overlap
    # ------------------------------------------------------------------

    @staticmethod
    def compute_path_overlap(
        predicted_path: List[Tuple[str, str]],
        actual_path: List[Tuple[str, str]],
    ) -> float:
        """
        Compute the edge-level Jaccard overlap between predicted and
        actual propagation paths.

        Each path is a list of directed edges ``(src, dst)``.

        Parameters
        ----------
        predicted_path : list of (str, str)
        actual_path : list of (str, str)

        Returns
        -------
        float
            Jaccard similarity in [0, 1].
        """
        pred_set = set(tuple(e) for e in predicted_path)
        actual_set = set(tuple(e) for e in actual_path)

        if not pred_set and not actual_set:
            return 0.0

        intersection = len(pred_set & actual_set)
        union = len(pred_set | actual_set)
        return intersection / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def full_evaluation(self) -> Dict[str, Union[float, Dict]]:
        """
        Run all cascade evaluation metrics at once.

        Returns
        -------
        dict
            Keys::

                {
                    "cascade_depth_mae": float,
                    "cascade_spread_accuracy": float,
                    "timing_accuracy": float,
                    "mean_path_overlap": float,
                    "n_evaluated": int,
                    "per_cascade": {cascade_id: {...}, ...}
                }
        """
        depth_mae = self.compute_cascade_depth_mae()
        spread_acc = self.compute_cascade_spread_accuracy()
        timing_acc = self.compute_timing_accuracy()

        # Per-cascade path overlap
        path_overlaps: List[float] = []
        per_cascade: Dict[str, Dict] = {}

        for cid in self._common_ids:
            gt = self.ground_truth[cid]
            pr = self.predicted[cid]

            entry: Dict[str, Optional[float]] = {}

            # Depth error
            if "depth" in gt and "depth" in pr:
                entry["depth_error"] = abs(float(pr["depth"]) - float(gt["depth"]))

            # Spread error
            if "affected_nodes" in gt and "affected_nodes" in pr:
                gt_set = set(gt["affected_nodes"])
                pr_set = set(pr["affected_nodes"])
                entry["spread_gt"] = len(gt_set)
                entry["spread_pred"] = len(pr_set)
                # Jaccard overlap of affected node sets
                if gt_set or pr_set:
                    entry["node_overlap"] = len(gt_set & pr_set) / len(gt_set | pr_set)
                else:
                    entry["node_overlap"] = 0.0

            # Path overlap
            gt_path = gt.get("path", [])
            pr_path = pr.get("path", [])
            if gt_path or pr_path:
                po = self.compute_path_overlap(pr_path, gt_path)
                path_overlaps.append(po)
                entry["path_overlap"] = po

            per_cascade[cid] = entry

        mean_path_overlap = float(np.mean(path_overlaps)) if path_overlaps else float("nan")

        return {
            "cascade_depth_mae": depth_mae,
            "cascade_spread_accuracy": spread_acc,
            "timing_accuracy": timing_acc,
            "mean_path_overlap": mean_path_overlap,
            "n_evaluated": len(self._common_ids),
            "per_cascade": per_cascade,
        }


if __name__ == "__main__":
    # ---- Demo with synthetic cascades ----

    ground_truth_cascades = [
        {
            "cascade_id": "c1",
            "source_node": "s0",
            "affected_nodes": ["s0", "s1", "s2", "s3"],
            "depth": 3,
            "timestamps": {"s0": 0.0, "s1": 1.0, "s2": 2.0, "s3": 3.0},
            "path": [("s0", "s1"), ("s1", "s2"), ("s2", "s3")],
        },
        {
            "cascade_id": "c2",
            "source_node": "s4",
            "affected_nodes": ["s4", "s5"],
            "depth": 1,
            "timestamps": {"s4": 0.0, "s5": 1.5},
            "path": [("s4", "s5")],
        },
    ]

    predicted_cascades = [
        {
            "cascade_id": "c1",
            "source_node": "s0",
            "affected_nodes": ["s0", "s1", "s2"],
            "depth": 2,
            "timestamps": {"s0": 0.0, "s1": 1.2, "s2": 2.5},
            "path": [("s0", "s1"), ("s1", "s2")],
        },
        {
            "cascade_id": "c2",
            "source_node": "s4",
            "affected_nodes": ["s4", "s5", "s6"],
            "depth": 2,
            "timestamps": {"s4": 0.0, "s5": 1.0},
            "path": [("s4", "s5"), ("s5", "s6")],
        },
    ]

    evaluator = CascadeEvaluator(ground_truth_cascades, predicted_cascades)

    print("Cascade Depth MAE:", evaluator.compute_cascade_depth_mae())
    print("Cascade Spread Accuracy:", evaluator.compute_cascade_spread_accuracy())
    print("Timing Accuracy:", evaluator.compute_timing_accuracy())

    po = CascadeEvaluator.compute_path_overlap(
        [("s0", "s1"), ("s1", "s2")],
        [("s0", "s1"), ("s1", "s2"), ("s2", "s3")],
    )
    print(f"Path Overlap (c1): {po:.4f}")

    print("\n--- Full evaluation ---")
    results = evaluator.full_evaluation()
    for key, val in results.items():
        if key != "per_cascade":
            print(f"  {key}: {val}")
    print("  Per-cascade details:")
    for cid, detail in results["per_cascade"].items():
        print(f"    {cid}: {detail}")

    print("\nCascade evaluation module ready for integration")
