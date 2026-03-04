#!/usr/bin/env python3
"""
HT-HGNN v2.0 -- Explanation Generation CLI
=============================================
Loads a trained HT-HGNN model and a dataset, then runs HyperSHAP to
produce per-node feature-importance explanations.

Usage examples:
    python scripts/explain.py --dataset bom --node-ids S0001 S0010 S0042
    python scripts/explain.py --dataset dataco --node-ids SUPPLIER_0 SUPPLIER_5 \\
        --output outputs/explanations.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = str(PROJECT_ROOT / "outputs" / "explanations.json")

# ---------------------------------------------------------------------------
# Hypergraph loader (shared logic)
# ---------------------------------------------------------------------------

def _load_hypergraph(dataset: str):
    """Load a Hypergraph object for *dataset*."""
    from src.hypergraph.hypergraph import Hypergraph, HypergraphNode, HypergraphEdge

    processed = PROJECT_ROOT / "Data set" / "processed" / f"{dataset}_hypergraph.json"
    if processed.exists():
        print(f"[load] Found pre-built hypergraph: {processed}")
        with open(processed, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        hg = Hypergraph()
        for nid, nprops in data.get("nodes", {}).items():
            hg.add_node(HypergraphNode(
                node_id=nid,
                node_type=nprops.get("node_type", "unknown"),
                tier=nprops.get("tier", 1),
                lead_time=nprops.get("lead_time", 14.0),
                reliability=nprops.get("reliability", 0.8),
                substitutability=nprops.get("substitutability", 0.5),
                cost=nprops.get("cost", 1.0),
            ))
        for hid, hprops in data.get("hyperedges", {}).items():
            hg.add_hyperedge(HypergraphEdge(
                hyperedge_id=hid,
                nodes=set(hprops.get("nodes", [])),
                bom_weight=hprops.get("bom_weight", 0.0),
                tolerance=hprops.get("tolerance", 0.0),
                critical_path=hprops.get("critical_path", 0.0),
                tier_level=hprops.get("tier_level", 0),
            ))
        for parent, children in data.get("echelon_dependencies", {}).items():
            for child in children:
                hg.add_echelon_dependency(parent, child)
        return hg

    # Dedicated loaders
    loader_map = {
        "dataco":      ("src.data.dataco_loader",      "DataCoLoader",
                        PROJECT_ROOT / "Data set" / "DataCo"),
        "bom":         ("src.data.bom_loader",          "BOMLoader",
                        PROJECT_ROOT / "Data set" / "BOM"),
        "ports":       ("src.data.port_loader",         "PortDisruptionLoader",
                        PROJECT_ROOT / "Data set" / "Ports"),
        "maintenance": ("src.data.maintenance_loader",  "MaintenanceLoader",
                        PROJECT_ROOT / "Data set" / "Maintenance"),
        "retail":      ("src.data.retail_loader",       "RetailLoader",
                        PROJECT_ROOT / "Data set" / "Retail"),
    }

    if dataset in loader_map:
        mod_name, cls_name, data_dir = loader_map[dataset]
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            loader = cls(str(data_dir))
            raw = loader.load()
            hg = loader.build_hypergraph(raw)
            print(f"[load] Built hypergraph via {cls_name}")
            return hg
        except (ImportError, AttributeError) as exc:
            print(f"[warn] Could not use {cls_name}: {exc}")

    # Fallback
    if dataset in ("bom", "dataco", "maintenance"):
        from src.data.real_data_loader import RealDataLoader
        loader = RealDataLoader(str(PROJECT_ROOT / "Data set"))
        data = loader.load_all()
        from src.hypergraph.hypergraph import Hypergraph
        hg = Hypergraph.from_dataframes(
            nodes_df=data["nodes"],
            hyperedges_df=data["hyperedges"],
            incidence_df=data["incidence"],
        )
        return hg

    raise RuntimeError(
        f"No loader available for dataset '{dataset}'.  "
        "Build the hypergraph first with scripts/build_hypergraphs.py."
    )


# ---------------------------------------------------------------------------
# Explanation engine
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "reliability", "lead_time", "substitutability", "cost", "tier",
    "node_degree", "avg_he_size", "critical_path_exposure",
]


def _compute_node_features(hg, node_id: str) -> dict:
    """Extract a feature vector for a single node from the hypergraph."""
    node = hg.nodes.get(node_id)
    if node is None:
        return {}

    he_ids = hg.node_to_hyperedges.get(node_id, set())
    degree = len(he_ids)
    he_sizes = [len(hg.incidence.get(h, set())) for h in he_ids]
    avg_he_size = float(np.mean(he_sizes)) if he_sizes else 0.0
    crit_count = sum(
        1 for h in he_ids
        if hg.hyperedges.get(h) and hg.hyperedges[h].critical_path > 0.5
    )
    crit_exposure = crit_count / max(degree, 1)

    return {
        "reliability": round(node.reliability, 4),
        "lead_time": round(node.lead_time, 4),
        "substitutability": round(node.substitutability, 4),
        "cost": round(node.cost, 4),
        "tier": node.tier,
        "node_degree": degree,
        "avg_he_size": round(avg_he_size, 4),
        "critical_path_exposure": round(crit_exposure, 4),
    }


def _builtin_explain(hg, node_ids: list, rng: np.random.Generator) -> dict:
    """
    Lightweight SHAP-like feature-importance approximation.

    For each requested node we compute marginal contribution of each
    feature to a simple risk score via random permutation sampling.
    """
    explanations = {}

    for nid in node_ids:
        feats = _compute_node_features(hg, nid)
        if not feats:
            explanations[nid] = {"error": f"Node '{nid}' not found in hypergraph."}
            continue

        # Simple risk score: weighted combination
        weights = {
            "reliability":            -0.30,
            "lead_time":               0.15,
            "substitutability":       -0.10,
            "cost":                    0.10,
            "tier":                   -0.05,
            "node_degree":             0.15,
            "avg_he_size":             0.10,
            "critical_path_exposure":  0.15,
        }

        base_score = sum(weights.get(k, 0) * v for k, v in feats.items())

        # Permutation importance (Monte Carlo)
        n_permutations = 200
        importances = {k: 0.0 for k in feats}

        feat_keys = list(feats.keys())
        feat_vals = np.array([feats[k] for k in feat_keys])

        for _ in range(n_permutations):
            perm = rng.permutation(len(feat_keys))
            marginal_prev = 0.0
            for idx in perm:
                # Score with features up to this one
                partial = sum(
                    weights.get(feat_keys[j], 0) * feat_vals[j]
                    for j in perm[: np.where(perm == idx)[0][0] + 1]
                )
                importances[feat_keys[idx]] += (partial - marginal_prev)
                marginal_prev = partial

        # Average
        for k in importances:
            importances[k] = round(importances[k] / n_permutations, 6)

        # Rank features by absolute importance
        ranked = sorted(importances.items(), key=lambda x: -abs(x[1]))

        explanations[nid] = {
            "features": feats,
            "risk_score": round(base_score, 4),
            "shapley_values": importances,
            "feature_ranking": [
                {"feature": k, "importance": v} for k, v in ranked
            ],
            "top_positive_driver": next(
                (k for k, v in ranked if v > 0), None
            ),
            "top_negative_driver": next(
                (k for k, v in ranked if v < 0), None
            ),
        }

    return explanations


def run_explanations(hg, node_ids: list,
                     rng: np.random.Generator) -> dict:
    """
    Run HyperSHAP if available, otherwise use the built-in approximation.
    """
    try:
        from src.explainability.hypershap import HyperSHAP
        explainer = HyperSHAP(hg)
        results = explainer.explain(node_ids=node_ids)
        return results
    except (ImportError, AttributeError):
        print("[warn] HyperSHAP not available -- using built-in permutation explainer.")

    return _builtin_explain(hg, node_ids, rng)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HT-HGNN v2.0 -- Explanation generation CLI",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["dataco", "bom", "ports", "maintenance", "retail"],
        help="Which dataset / hypergraph to load.",
    )
    parser.add_argument(
        "--node-ids",
        nargs="+",
        required=True,
        help="One or more node IDs to explain (space-separated).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print("HT-HGNN v2.0 -- Explanation Generation")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Node IDs  : {args.node_ids}")
    print(f"  Seed      : {args.seed}")
    print(f"  Output    : {args.output}")

    start = time.time()

    # 1. Load
    print("\n--- Loading hypergraph ---")
    hg = _load_hypergraph(args.dataset)
    stats = hg.get_statistics()
    print(f"  Nodes: {stats['n_nodes']}, Hyperedges: {stats['n_hyperedges']}")

    # Validate node IDs
    missing = [nid for nid in args.node_ids if nid not in hg.nodes]
    if missing:
        available = sorted(list(hg.nodes.keys()))[:15]
        print(f"\n[WARN] The following node IDs were not found: {missing}")
        print(f"       Available nodes (first 15): {available}")
        valid_ids = [nid for nid in args.node_ids if nid in hg.nodes]
        if not valid_ids:
            print("[ERROR] No valid node IDs provided. Exiting.")
            sys.exit(1)
        print(f"       Continuing with valid IDs: {valid_ids}")
        args.node_ids = valid_ids

    # 2. Generate explanations
    print("\n--- Generating explanations ---")
    explanations = run_explanations(hg, args.node_ids, rng)

    # 3. Save
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_payload = {
        "dataset": args.dataset,
        "num_nodes_explained": len(args.node_ids),
        "explanations": explanations,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2, default=str)
    print(f"\n[save] Explanations written to {out_path}")

    # 4. Print summary per node
    print("\n" + "=" * 60)
    print("EXPLANATION SUMMARY")
    print("=" * 60)

    for nid in args.node_ids:
        expl = explanations.get(nid, {})
        if "error" in expl:
            print(f"\n  Node {nid}: {expl['error']}")
            continue

        print(f"\n  Node: {nid}")
        print(f"    Risk score : {expl.get('risk_score', 'N/A')}")

        ranking = expl.get("feature_ranking", [])
        if ranking:
            print("    Feature importances (Shapley values):")
            for entry in ranking:
                direction = "+" if entry["importance"] > 0 else " "
                print(f"      {entry['feature']:30s}  "
                      f"{direction}{entry['importance']:.6f}")

        top_pos = expl.get("top_positive_driver")
        top_neg = expl.get("top_negative_driver")
        if top_pos:
            print(f"    Top risk-increasing feature  : {top_pos}")
        if top_neg:
            print(f"    Top risk-decreasing feature  : {top_neg}")

    print(f"\n  Elapsed: {time.time() - start:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
