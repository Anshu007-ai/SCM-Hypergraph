#!/usr/bin/env python3
"""
HT-HGNN v2.0 -- Build Hypergraphs from Raw Data
==================================================
CLI utility that reads each raw dataset through its dedicated loader,
constructs the hypergraph representation, and serialises the result to
JSON in the processed-output directory.

Usage examples:
    python scripts/build_hypergraphs.py --all
    python scripts/build_hypergraphs.py --dataset dataco
    python scripts/build_hypergraphs.py --dataset bom --output "Data set/processed/"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hypergraph.hypergraph import Hypergraph, HypergraphNode, HypergraphEdge

DEFAULT_OUTPUT = str(PROJECT_ROOT / "Data set" / "processed")

# ---------------------------------------------------------------------------
# Per-dataset builders
# ---------------------------------------------------------------------------

def _save_hypergraph(hg: Hypergraph, name: str, output_dir: Path) -> Path:
    """Serialise a Hypergraph to JSON and return the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_hypergraph.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(hg.to_dict(), fh, indent=2, default=str)
    return out_path


def _print_stats(hg: Hypergraph, name: str) -> None:
    """Print a quick summary table for *hg*."""
    stats = hg.get_statistics()
    print(f"\n  --- {name} hypergraph stats ---")
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"    {key:30s}: {val:.4f}")
        else:
            print(f"    {key:30s}: {val}")


def build_dataco(output_dir: Path) -> None:
    """Build hypergraph from DataCo SMART Supply Chain data."""
    print("\n[dataco] Loading DataCo dataset ...")
    try:
        from src.data.dataco_loader import DataCoLoader
    except ImportError:
        print("  [WARN] src.data.dataco_loader.DataCoLoader not found -- "
              "falling back to RealDataLoader.")
        from src.data.real_data_loader import RealDataLoader
        loader = RealDataLoader(str(PROJECT_ROOT / "Data set"))
        loader.load_dataco_data()
        print("  [INFO] DataCo loaded but dedicated loader unavailable; "
              "skipping hypergraph construction for this dataset.")
        return

    start = time.time()
    loader = DataCoLoader(str(PROJECT_ROOT / "Data set" / "DataCo"))
    data = loader.load()
    hg = loader.build_hypergraph(data)
    out = _save_hypergraph(hg, "dataco", output_dir)
    _print_stats(hg, "dataco")
    print(f"  [OK] Saved to {out}  ({time.time() - start:.1f}s)")


def build_bom(output_dir: Path) -> None:
    """Build hypergraph from Bill of Materials data."""
    print("\n[bom] Loading BOM dataset ...")
    try:
        from src.data.bom_loader import BOMLoader
    except ImportError:
        print("  [WARN] src.data.bom_loader.BOMLoader not found -- "
              "falling back to RealDataLoader.")
        from src.data.real_data_loader import RealDataLoader
        loader = RealDataLoader(str(PROJECT_ROOT / "Data set"))
        data = loader.load_all()
        # Build from extracted DataFrames
        hg = Hypergraph.from_dataframes(
            nodes_df=data["nodes"],
            hyperedges_df=data["hyperedges"],
            incidence_df=data["incidence"],
        )
        out = _save_hypergraph(hg, "bom", output_dir)
        _print_stats(hg, "bom")
        print(f"  [OK] Saved to {out}")
        return

    start = time.time()
    loader = BOMLoader(str(PROJECT_ROOT / "Data set" / "BOM"))
    data = loader.load()
    hg = loader.build_hypergraph(data)
    out = _save_hypergraph(hg, "bom", output_dir)
    _print_stats(hg, "bom")
    print(f"  [OK] Saved to {out}  ({time.time() - start:.1f}s)")


def build_ports(output_dir: Path) -> None:
    """Build hypergraph from port disruption data."""
    print("\n[ports] Loading Port Disruption dataset ...")
    try:
        from src.data.port_loader import PortDisruptionLoader
    except ImportError:
        print("  [WARN] src.data.port_loader.PortDisruptionLoader not available.")
        print("         Ensure the module exists and the Ports dataset has been "
              "downloaded (see scripts/download_datasets.py --dataset ports).")
        return

    port_dir = PROJECT_ROOT / "Data set" / "Ports"
    if not port_dir.exists():
        print(f"  [ERROR] Port data directory not found: {port_dir}")
        print("          Run `python scripts/download_datasets.py --dataset ports` first.")
        return

    start = time.time()
    loader = PortDisruptionLoader(str(port_dir))
    data = loader.load()
    hg = loader.build_hypergraph(data)
    out = _save_hypergraph(hg, "ports", output_dir)
    _print_stats(hg, "ports")
    print(f"  [OK] Saved to {out}  ({time.time() - start:.1f}s)")


def build_maintenance(output_dir: Path) -> None:
    """Build hypergraph from the predictive-maintenance dataset."""
    print("\n[maintenance] Loading Maintenance dataset ...")
    try:
        from src.data.maintenance_loader import MaintenanceLoader
    except ImportError:
        print("  [WARN] src.data.maintenance_loader.MaintenanceLoader not available.")
        print("         Ensure the module exists and ai4i2020.csv is present in "
              "Data set/Maintenance/.")
        return

    maint_dir = PROJECT_ROOT / "Data set" / "Maintenance"
    csv_path = maint_dir / "ai4i2020.csv"
    if not csv_path.exists():
        print(f"  [ERROR] Maintenance CSV not found: {csv_path}")
        print("          Run `python scripts/download_datasets.py --dataset maintenance` first.")
        return

    start = time.time()
    loader = MaintenanceLoader(str(maint_dir))
    data = loader.load()
    hg = loader.build_hypergraph(data)
    out = _save_hypergraph(hg, "maintenance", output_dir)
    _print_stats(hg, "maintenance")
    print(f"  [OK] Saved to {out}  ({time.time() - start:.1f}s)")


def build_retail(output_dir: Path) -> None:
    """Build hypergraph from the Online Retail dataset."""
    print("\n[retail] Loading Retail dataset ...")
    try:
        from src.data.retail_loader import RetailLoader
    except ImportError:
        print("  [WARN] src.data.retail_loader.RetailLoader not available.")
        print("         Ensure the module exists and the Retail dataset has been "
              "downloaded (see scripts/download_datasets.py --dataset retail).")
        return

    retail_dir = PROJECT_ROOT / "Data set" / "Retail"
    if not retail_dir.exists():
        print(f"  [ERROR] Retail data directory not found: {retail_dir}")
        print("          Run `python scripts/download_datasets.py --dataset retail` first.")
        return

    start = time.time()
    loader = RetailLoader(str(retail_dir))
    data = loader.load()
    hg = loader.build_hypergraph(data)
    out = _save_hypergraph(hg, "retail", output_dir)
    _print_stats(hg, "retail")
    print(f"  [OK] Saved to {out}  ({time.time() - start:.1f}s)")


# ---------------------------------------------------------------------------

BUILDERS = {
    "dataco":      build_dataco,
    "bom":         build_bom,
    "ports":       build_ports,
    "maintenance": build_maintenance,
    "retail":      build_retail,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HT-HGNN v2.0 -- Build hypergraphs from raw datasets",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build hypergraphs for every available dataset.",
    )
    parser.add_argument(
        "--dataset",
        choices=list(BUILDERS.keys()),
        help="Build a hypergraph for a single dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for processed JSON files (default: {DEFAULT_OUTPUT}).",
    )

    args = parser.parse_args()

    if not args.all and args.dataset is None:
        parser.print_help()
        sys.exit(1)

    output_dir = Path(args.output).resolve()
    print("HT-HGNN v2.0 -- Build Hypergraphs")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output dir   : {output_dir}")

    overall_start = time.time()

    if args.all:
        for name, builder in BUILDERS.items():
            try:
                builder(output_dir)
            except FileNotFoundError as exc:
                print(f"  [ERROR] {name}: missing file -- {exc}")
            except Exception as exc:
                print(f"  [ERROR] {name}: {exc}")
    else:
        try:
            BUILDERS[args.dataset](output_dir)
        except FileNotFoundError as exc:
            print(f"  [ERROR] {args.dataset}: missing file -- {exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"  [ERROR] {args.dataset}: {exc}")
            sys.exit(1)

    print(f"\nTotal elapsed: {time.time() - overall_start:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
