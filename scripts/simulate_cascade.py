#!/usr/bin/env python3
"""
HT-HGNN v2.0 -- Cascade Simulation CLI
=========================================
Loads a dataset, builds the hypergraph representation, injects a shock
at a user-specified node, and runs the CascadeEngine forward in time.

Usage examples:
    python scripts/simulate_cascade.py --dataset bom --shock-node S0010 --shock-magnitude 0.8
    python scripts/simulate_cascade.py --dataset dataco --shock-node SUPPLIER_5 \\
        --shock-magnitude 0.6 --time-steps 20 --output outputs/cascade_result.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = str(PROJECT_ROOT / "outputs" / "cascade_result.json")

# ---------------------------------------------------------------------------
# Dataset loading helpers (mirrors build_hypergraphs.py logic)
# ---------------------------------------------------------------------------

def _load_hypergraph(dataset: str):
    """
    Return a Hypergraph object for *dataset*.

    Strategy:
        1. Try the dedicated loader (e.g. BOMLoader).
        2. Fall back to loading a pre-built JSON from Data set/processed/.
        3. Fall back to RealDataLoader for bom/dataco/maintenance.
    """
    from src.hypergraph.hypergraph import Hypergraph

    # --- Try pre-built JSON first (fastest) --------------------------------
    processed = PROJECT_ROOT / "Data set" / "processed" / f"{dataset}_hypergraph.json"
    if processed.exists():
        print(f"[load] Found pre-built hypergraph: {processed}")
        with open(processed, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        hg = Hypergraph()
        from src.hypergraph.hypergraph import HypergraphNode, HypergraphEdge
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

    # --- Try dedicated loaders ---------------------------------------------
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
        "indigo":      ("src.data.indigo_disruption_loader", "IndiGoDisruptionLoader",
                        PROJECT_ROOT / "Data set"),
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

    # --- Fallback: RealDataLoader for BOM ----------------------------------
    if dataset in ("bom", "dataco", "maintenance"):
        print("[load] Falling back to RealDataLoader ...")
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
# Simulation
# ---------------------------------------------------------------------------

def run_cascade(hg, shock_node: str, shock_magnitude: float,
                time_steps: int) -> dict:
    """
    Run the CascadeEngine and return a results dict.

    Falls back to a simple BFS cascade if CascadeEngine is not importable.
    """
    try:
        from src.simulation.cascade_engine import CascadeEngine
        engine = CascadeEngine(hg)
        results = engine.simulate(
            shock_node=shock_node,
            shock_magnitude=shock_magnitude,
            time_steps=time_steps,
        )
        return results
    except (ImportError, AttributeError):
        print("[warn] CascadeEngine not available -- running built-in BFS cascade.")

    # ------ Built-in lightweight cascade simulation ------
    affected = {}
    current_magnitude = shock_magnitude
    frontier = {shock_node}
    visited = set()
    timeline = []

    for t in range(time_steps):
        if not frontier:
            break

        next_frontier = set()
        step_affected = []

        for node_id in frontier:
            if node_id in visited:
                continue
            visited.add(node_id)

            node = hg.nodes.get(node_id)
            reliability = node.reliability if node else 0.8
            impact = current_magnitude * (1 - reliability)
            affected[node_id] = {"step": t, "impact": round(impact, 4)}
            step_affected.append(node_id)

            # Propagate through shared hyperedges
            for he_id in hg.node_to_hyperedges.get(node_id, set()):
                for neighbor in hg.incidence.get(he_id, set()):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)

        timeline.append({
            "step": t,
            "newly_affected": step_affected,
            "cumulative_affected": len(visited),
            "current_magnitude": round(current_magnitude, 4),
        })

        current_magnitude *= 0.85  # natural decay
        frontier = next_frontier

    return {
        "shock_node": shock_node,
        "shock_magnitude": shock_magnitude,
        "time_steps_executed": len(timeline),
        "total_affected_nodes": len(affected),
        "total_nodes": len(hg.nodes),
        "affected_fraction": round(len(affected) / max(len(hg.nodes), 1), 4),
        "timeline": timeline,
        "affected_nodes": affected,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HT-HGNN v2.0 -- Cascade simulation CLI",
    )
    parser.add_argument(
        "--dataset",
        required=False,
        choices=["dataco", "bom", "ports", "maintenance", "retail", "indigo"],
        help="Which dataset / hypergraph to load.",
    )
    parser.add_argument(
        "--shock-node",
        required=False,
        help="Node ID where the shock originates (e.g. S0010).",
    )
    parser.add_argument(
        "--shock-magnitude",
        type=float,
        default=1.0,
        help="Initial shock magnitude in [0, 1] (default: 1.0).",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=10,
        help="Number of simulation time steps (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["indigo-fdtl"],
        help="Use a preset simulation configuration (e.g. indigo-fdtl).",
    )

    args = parser.parse_args()

    # --- Apply preset overrides ---
    if args.preset == "indigo-fdtl":
        args.dataset = "indigo"
        args.shock_node = "REG_01"       # DGCA_FDTL_Compliance
        args.shock_magnitude = 0.95
        args.time_steps = 12
        print("HT-HGNN v2.0 -- Cascade Simulation  [preset: indigo-fdtl]")
        print("  Simulating FDTL Phase 2 regulatory shock cascade")
        print("  Seed node: DGCA_FDTL_Compliance (REG_01)")
    else:
        if not args.dataset or not args.shock_node:
            parser.error("--dataset and --shock-node are required unless --preset is used.")

    print("HT-HGNN v2.0 -- Cascade Simulation")
    print(f"  Dataset          : {args.dataset}")
    print(f"  Shock node       : {args.shock_node}")
    print(f"  Shock magnitude  : {args.shock_magnitude}")
    print(f"  Time steps       : {args.time_steps}")
    print(f"  Output           : {args.output}")

    start = time.time()

    # 1. Load
    print("\n--- Loading hypergraph ---")
    hg = _load_hypergraph(args.dataset)
    stats = hg.get_statistics()
    print(f"  Nodes: {stats['n_nodes']}, Hyperedges: {stats['n_hyperedges']}")

    # Validate shock node
    if args.shock_node not in hg.nodes:
        available = sorted(list(hg.nodes.keys()))[:10]
        print(f"\n[ERROR] Shock node '{args.shock_node}' not found in hypergraph.")
        print(f"        Available nodes (first 10): {available}")
        sys.exit(1)

    # 2. Simulate
    print("\n--- Running cascade simulation ---")
    results = run_cascade(
        hg,
        shock_node=args.shock_node,
        shock_magnitude=args.shock_magnitude,
        time_steps=args.time_steps,
    )

    # 3. Save
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\n[save] Results written to {out_path}")

    # 4. Print timeline summary
    print("\n--- Timeline Summary ---")
    print(f"  {'Step':>4s}  {'New':>6s}  {'Cumul':>6s}  {'Magnitude':>10s}")
    print(f"  {'----':>4s}  {'---':>6s}  {'-----':>6s}  {'---------':>10s}")
    for entry in results.get("timeline", []):
        print(f"  {entry['step']:4d}  "
              f"{len(entry['newly_affected']):6d}  "
              f"{entry['cumulative_affected']:6d}  "
              f"{entry['current_magnitude']:10.4f}")

    print(f"\n  Total affected : {results['total_affected_nodes']} / "
          f"{results['total_nodes']} "
          f"({results['affected_fraction'] * 100:.1f}%)")
    print(f"  Elapsed        : {time.time() - start:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
