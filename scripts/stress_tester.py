#!/usr/bin/env python3
"""
HT-HGNN v2.0 -- Bulk Stress Testing CLI
==========================================
Runs multiple shock scenarios against a supply-chain hypergraph and
aggregates the results into a vulnerability report.

Shock strategies:
    random_node          -- pick a random node as shock origin
    random_hyperedge     -- pick a random hyperedge; shock all its members
    targeted_high_degree -- target the highest-degree node
    geographic_cluster   -- shock a geographic cluster (zone) of nodes

Usage examples:
    python scripts/stress_tester.py --dataset bom --num-scenarios 50
    python scripts/stress_tester.py --dataset dataco --num-scenarios 100 \\
        --shock-strategy targeted_high_degree --output outputs/stress.json
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

DEFAULT_OUTPUT = str(PROJECT_ROOT / "outputs" / "stress_test_results.json")


# ---------------------------------------------------------------------------
# Reuse the hypergraph loader from simulate_cascade
# ---------------------------------------------------------------------------

def _load_hypergraph(dataset: str):
    """Load hypergraph -- see simulate_cascade.py for the full strategy."""
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

    if dataset in ("bom", "dataco", "maintenance"):
        from src.data.real_data_loader import RealDataLoader
        loader = RealDataLoader(str(PROJECT_ROOT / "Data set"))
        data = loader.load_all()
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
# Shock scenario generators
# ---------------------------------------------------------------------------

def _select_shock_nodes(hg, strategy: str, rng: np.random.Generator) -> list:
    """Return a list of node IDs to shock for one scenario."""
    node_ids = sorted(hg.nodes.keys())
    he_ids = sorted(hg.hyperedges.keys())

    if strategy == "random_node":
        return [rng.choice(node_ids)]

    elif strategy == "random_hyperedge":
        if not he_ids:
            return [rng.choice(node_ids)]
        chosen_he = rng.choice(he_ids)
        members = list(hg.incidence.get(chosen_he, set()))
        return members if members else [rng.choice(node_ids)]

    elif strategy == "targeted_high_degree":
        degrees = {nid: len(hg.node_to_hyperedges.get(nid, set()))
                   for nid in node_ids}
        max_deg = max(degrees.values()) if degrees else 0
        top_nodes = [nid for nid, d in degrees.items() if d == max_deg]
        return [rng.choice(top_nodes)]

    elif strategy == "geographic_cluster":
        # Group by node_type as a proxy for geographic clusters
        type_groups: dict = {}
        for nid, node in hg.nodes.items():
            type_groups.setdefault(node.node_type, []).append(nid)
        if type_groups:
            chosen_type = rng.choice(list(type_groups.keys()))
            cluster = type_groups[chosen_type]
            # Shock a random subset (up to 5) of the cluster
            n_shock = min(len(cluster), max(1, int(rng.integers(1, 6))))
            return rng.choice(cluster, size=n_shock, replace=False).tolist()
        return [rng.choice(node_ids)]

    else:
        raise ValueError(f"Unknown shock strategy: {strategy}")


def _run_single_cascade(hg, shock_nodes: list, magnitude: float,
                         time_steps: int = 10) -> dict:
    """
    Run a single cascade starting from *shock_nodes*.
    Returns a summary dict (not the full timeline).
    """
    affected = {}
    frontier = set(shock_nodes)
    visited = set()
    current_mag = magnitude

    for t in range(time_steps):
        if not frontier:
            break
        next_frontier = set()
        for nid in frontier:
            if nid in visited:
                continue
            visited.add(nid)
            node = hg.nodes.get(nid)
            rel = node.reliability if node else 0.8
            impact = current_mag * (1 - rel)
            affected[nid] = {"step": t, "impact": round(impact, 4)}
            for he_id in hg.node_to_hyperedges.get(nid, set()):
                for neighbor in hg.incidence.get(he_id, set()):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
        current_mag *= 0.85
        frontier = next_frontier

    return {
        "shock_nodes": shock_nodes,
        "total_affected": len(affected),
        "affected_fraction": round(len(affected) / max(len(hg.nodes), 1), 4),
        "max_impact": max((a["impact"] for a in affected.values()), default=0),
        "steps_active": min(t + 1, time_steps),
    }


# ---------------------------------------------------------------------------
# Stress testing orchestrator
# ---------------------------------------------------------------------------

def run_stress_test(hg, num_scenarios: int, strategy: str,
                    rng: np.random.Generator,
                    magnitude: float = 1.0,
                    time_steps: int = 10) -> dict:
    """
    Try to use src.simulation.stress_tester.StressTester first;
    fall back to built-in logic.
    """
    try:
        from src.simulation.stress_tester import StressTester
        tester = StressTester(hg)
        results = tester.run(
            num_scenarios=num_scenarios,
            shock_strategy=strategy,
            magnitude=magnitude,
            time_steps=time_steps,
        )
        return results
    except (ImportError, AttributeError):
        print("[warn] StressTester not available -- using built-in stress loop.")

    scenario_results = []
    node_hit_count: dict = {}

    for i in range(num_scenarios):
        shock_nodes = _select_shock_nodes(hg, strategy, rng)
        res = _run_single_cascade(hg, shock_nodes, magnitude, time_steps)
        scenario_results.append(res)

        for nid in shock_nodes:
            node_hit_count[nid] = node_hit_count.get(nid, 0) + 1

        if (i + 1) % max(1, num_scenarios // 10) == 0:
            print(f"  scenario {i + 1}/{num_scenarios} ...")

    # Aggregate
    cascade_sizes = [r["total_affected"] for r in scenario_results]
    fractions = [r["affected_fraction"] for r in scenario_results]

    # Identify most dangerous nodes (those whose shocks cause largest cascades)
    node_max_cascade: dict = {}
    for res in scenario_results:
        for nid in res["shock_nodes"]:
            prev = node_max_cascade.get(nid, 0)
            node_max_cascade[nid] = max(prev, res["total_affected"])

    top_dangerous = sorted(node_max_cascade.items(), key=lambda x: -x[1])[:10]

    # Vulnerability score: mean affected fraction
    vulnerability_score = round(float(np.mean(fractions)), 4)

    return {
        "dataset_nodes": len(hg.nodes),
        "dataset_hyperedges": len(hg.hyperedges),
        "strategy": strategy,
        "num_scenarios": num_scenarios,
        "magnitude": magnitude,
        "time_steps": time_steps,
        "mean_cascade_size": round(float(np.mean(cascade_sizes)), 2),
        "median_cascade_size": round(float(np.median(cascade_sizes)), 2),
        "max_cascade_size": int(np.max(cascade_sizes)),
        "std_cascade_size": round(float(np.std(cascade_sizes)), 2),
        "vulnerability_score": vulnerability_score,
        "top_dangerous_nodes": [
            {"node_id": nid, "max_cascade": cnt}
            for nid, cnt in top_dangerous
        ],
        "scenarios": scenario_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HT-HGNN v2.0 -- Bulk stress testing CLI",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["dataco", "bom", "ports", "maintenance", "retail"],
        help="Which dataset / hypergraph to load.",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=50,
        help="Number of shock scenarios to run (default: 50).",
    )
    parser.add_argument(
        "--shock-strategy",
        choices=["random_node", "random_hyperedge",
                 "targeted_high_degree", "geographic_cluster"],
        default="random_node",
        help="Strategy for choosing shock targets (default: random_node).",
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        default=1.0,
        help="Shock magnitude [0, 1] (default: 1.0).",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=10,
        help="Max time steps per scenario (default: 10).",
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

    print("HT-HGNN v2.0 -- Bulk Stress Testing")
    print(f"  Dataset        : {args.dataset}")
    print(f"  Scenarios      : {args.num_scenarios}")
    print(f"  Strategy       : {args.shock_strategy}")
    print(f"  Magnitude      : {args.magnitude}")
    print(f"  Time steps     : {args.time_steps}")
    print(f"  Seed           : {args.seed}")
    print(f"  Output         : {args.output}")

    start = time.time()

    # 1. Load
    print("\n--- Loading hypergraph ---")
    hg = _load_hypergraph(args.dataset)
    stats = hg.get_statistics()
    print(f"  Nodes: {stats['n_nodes']}, Hyperedges: {stats['n_hyperedges']}")

    # 2. Stress test
    print("\n--- Running stress test ---")
    results = run_stress_test(
        hg,
        num_scenarios=args.num_scenarios,
        strategy=args.shock_strategy,
        rng=rng,
        magnitude=args.magnitude,
        time_steps=args.time_steps,
    )

    # 3. Save
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\n[save] Results written to {out_path}")

    # 4. Print summary
    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    print(f"  Strategy             : {results['strategy']}")
    print(f"  Scenarios executed   : {results['num_scenarios']}")
    print(f"  Mean cascade size    : {results['mean_cascade_size']}")
    print(f"  Median cascade size  : {results['median_cascade_size']}")
    print(f"  Max cascade size     : {results['max_cascade_size']}")
    print(f"  Std cascade size     : {results['std_cascade_size']}")
    print(f"  Vulnerability score  : {results['vulnerability_score']}")

    print("\n  Top dangerous nodes:")
    for entry in results.get("top_dangerous_nodes", []):
        print(f"    {entry['node_id']:20s} -> max cascade {entry['max_cascade']}")

    print(f"\n  Elapsed: {time.time() - start:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
