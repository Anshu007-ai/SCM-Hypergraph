#!/usr/bin/env python3
"""
HT-HGNN v2.0 -- Augment Maintenance Dataset
==============================================
Reads the base UCI AI4I 2020 predictive-maintenance CSV (10 000 rows),
generates four auxiliary synthetic tables that model a realistic
manufacturing-maintenance supply chain, and augments the original
dataset from 10K to 100K records via realistic perturbation.

Generated files (all written to  Data set/Maintenance/):
    - machine_topology.csv          -- factory floor layout
    - maintenance_crews.csv         -- crew / shift information
    - spare_parts_inventory.csv     -- part stock & lead times
    - failure_cascade_log.csv       -- inter-machine failure cascades
    - ai4i2020_augmented.csv        -- 100K augmented sensor records

Usage:
    python scripts/augment_maintenance.py
    python scripts/augment_maintenance.py --records 50000
    python scripts/augment_maintenance.py --seed 123
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "Data set" / "Maintenance"
BASE_CSV = DATA_DIR / "ai4i2020.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_base(path: Path) -> pd.DataFrame:
    """Load the base AI4I 2020 CSV, raising early if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Base maintenance CSV not found: {path}\n"
            "Run  python scripts/download_datasets.py --dataset maintenance  first."
        )
    df = pd.read_csv(path)
    print(f"[load] Base dataset: {len(df)} rows x {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------------------
# 1. Machine topology
# ---------------------------------------------------------------------------

def generate_machine_topology(rng: np.random.Generator,
                              n_machines: int = 200) -> pd.DataFrame:
    """
    Create a factory-floor machine topology table.

    Columns:
        machine_id, machine_type, zone, line_id, upstream_machine_id,
        install_year, criticality
    """
    print("[gen] Generating machine_topology.csv ...")
    types = ["CNC", "Lathe", "Press", "Welder", "Robot", "Conveyor",
             "Furnace", "Grinder", "Assembler", "Inspector"]
    zones = ["Zone-A", "Zone-B", "Zone-C", "Zone-D"]
    lines = [f"LINE-{i}" for i in range(1, 11)]

    rows = []
    for i in range(n_machines):
        mid = f"MACH-{i:04d}"
        mtype = rng.choice(types)
        zone = rng.choice(zones)
        line = rng.choice(lines)
        # upstream dependency -- roughly 60 % of machines depend on another
        upstream = (
            f"MACH-{rng.integers(0, max(i, 1)):04d}"
            if i > 0 and rng.random() < 0.6
            else ""
        )
        install_year = int(rng.integers(2005, 2024))
        criticality = round(float(rng.beta(2, 5)), 3)
        rows.append({
            "machine_id": mid,
            "machine_type": mtype,
            "zone": zone,
            "line_id": line,
            "upstream_machine_id": upstream,
            "install_year": install_year,
            "criticality": criticality,
        })
    df = pd.DataFrame(rows)
    print(f"       {len(df)} machines across {len(zones)} zones, "
          f"{len(lines)} lines")
    return df


# ---------------------------------------------------------------------------
# 2. Maintenance crews
# ---------------------------------------------------------------------------

def generate_maintenance_crews(rng: np.random.Generator,
                                n_crews: int = 40) -> pd.DataFrame:
    """
    Create a maintenance crew / shift table.

    Columns:
        crew_id, crew_name, shift, specialization, avg_repair_hours,
        certifications, home_zone
    """
    print("[gen] Generating maintenance_crews.csv ...")
    shifts = ["Day", "Night", "Swing"]
    specializations = ["Mechanical", "Electrical", "Hydraulic",
                       "Pneumatic", "PLC", "General"]
    zones = ["Zone-A", "Zone-B", "Zone-C", "Zone-D"]

    rows = []
    for i in range(n_crews):
        cid = f"CREW-{i:03d}"
        name = f"Team-{chr(65 + i % 26)}{i // 26 + 1}"
        shift = rng.choice(shifts)
        spec = rng.choice(specializations)
        avg_hours = round(float(rng.lognormal(mean=1.0, sigma=0.5)), 2)
        certs = int(rng.integers(1, 6))
        zone = rng.choice(zones)
        rows.append({
            "crew_id": cid,
            "crew_name": name,
            "shift": shift,
            "specialization": spec,
            "avg_repair_hours": avg_hours,
            "certifications": certs,
            "home_zone": zone,
        })
    df = pd.DataFrame(rows)
    print(f"       {len(df)} crews, shifts: {shifts}")
    return df


# ---------------------------------------------------------------------------
# 3. Spare parts inventory
# ---------------------------------------------------------------------------

def generate_spare_parts_inventory(rng: np.random.Generator,
                                    n_parts: int = 300) -> pd.DataFrame:
    """
    Create a spare-parts inventory table.

    Columns:
        part_id, part_name, compatible_machine_types, stock_qty,
        reorder_point, lead_time_days, unit_cost, supplier_id
    """
    print("[gen] Generating spare_parts_inventory.csv ...")
    machine_types = ["CNC", "Lathe", "Press", "Welder", "Robot",
                     "Conveyor", "Furnace", "Grinder", "Assembler",
                     "Inspector"]
    part_prefixes = ["Bearing", "Seal", "Belt", "Gear", "Filter",
                     "Valve", "Sensor", "Motor", "Pump", "Relay",
                     "Fuse", "Coupling", "Shaft", "Cylinder", "Nozzle"]

    rows = []
    for i in range(n_parts):
        pid = f"PART-{i:04d}"
        prefix = rng.choice(part_prefixes)
        name = f"{prefix}-{rng.integers(100, 999)}"
        # compatible with 1-3 machine types
        n_compat = int(rng.integers(1, 4))
        compat = "|".join(rng.choice(machine_types, size=n_compat,
                                      replace=False).tolist())
        stock = int(rng.poisson(lam=20))
        reorder = int(rng.integers(3, 15))
        lead_time = int(rng.integers(1, 45))
        unit_cost = round(float(rng.lognormal(mean=3.0, sigma=1.2)), 2)
        supplier = f"SUPPLIER-{rng.integers(0, 50):03d}"
        rows.append({
            "part_id": pid,
            "part_name": name,
            "compatible_machine_types": compat,
            "stock_qty": stock,
            "reorder_point": reorder,
            "lead_time_days": lead_time,
            "unit_cost": unit_cost,
            "supplier_id": supplier,
        })
    df = pd.DataFrame(rows)
    print(f"       {len(df)} spare parts, "
          f"avg stock {df['stock_qty'].mean():.1f}")
    return df


# ---------------------------------------------------------------------------
# 4. Failure cascade log
# ---------------------------------------------------------------------------

def generate_failure_cascade_log(
    rng: np.random.Generator,
    topology: pd.DataFrame,
    n_events: int = 2000,
) -> pd.DataFrame:
    """
    Simulate failure-cascade events across the machine topology.

    Columns:
        event_id, origin_machine, affected_machine, cascade_depth,
        timestamp, root_cause, severity, downtime_hours
    """
    print("[gen] Generating failure_cascade_log.csv ...")
    causes = ["TWF", "HDF", "PWF", "OSF", "RNF",
              "Electrical", "Operator_Error", "Supply_Delay"]

    # Build upstream lookup
    machine_ids = topology["machine_id"].tolist()
    upstream_map: dict = {}
    for _, row in topology.iterrows():
        if row["upstream_machine_id"]:
            upstream_map.setdefault(row["upstream_machine_id"], []).append(
                row["machine_id"]
            )

    rows = []
    base_ts = pd.Timestamp("2023-01-01")

    for eid in range(n_events):
        origin = rng.choice(machine_ids)
        cause = rng.choice(causes)
        severity = round(float(rng.beta(2, 5)), 3)
        ts = base_ts + pd.Timedelta(hours=int(rng.integers(0, 17520)))

        # Walk downstream cascade
        cascade = [origin]
        visited = {origin}
        depth = 0
        current_frontier = [origin]
        max_depth = int(rng.integers(1, 5))

        while current_frontier and depth < max_depth:
            next_frontier = []
            for m in current_frontier:
                for downstream in upstream_map.get(m, []):
                    if downstream not in visited and rng.random() < 0.4:
                        visited.add(downstream)
                        next_frontier.append(downstream)
                        cascade.append(downstream)
            current_frontier = next_frontier
            depth += 1

        for d, affected in enumerate(cascade):
            downtime = round(float(rng.exponential(scale=4.0 / (d + 1))), 2)
            rows.append({
                "event_id": f"EVT-{eid:05d}",
                "origin_machine": origin,
                "affected_machine": affected,
                "cascade_depth": d,
                "timestamp": ts.isoformat(),
                "root_cause": cause,
                "severity": severity,
                "downtime_hours": downtime,
            })

    df = pd.DataFrame(rows)
    print(f"       {n_events} root events -> {len(df)} total cascade rows")
    return df


# ---------------------------------------------------------------------------
# 5. Augment base dataset 10K -> 100K
# ---------------------------------------------------------------------------

def augment_base_dataset(
    base_df: pd.DataFrame,
    rng: np.random.Generator,
    target_rows: int = 100_000,
) -> pd.DataFrame:
    """
    Expand the base 10K dataset to *target_rows* via realistic perturbation.

    Strategy:
        - Repeat the base set enough times to exceed *target_rows*.
        - For every copy, apply Gaussian jitter to numerical columns and
          small random shifts to categorical columns, keeping the
          statistical fingerprint realistic.
        - Trim to exactly *target_rows*.
    """
    print(f"[aug] Augmenting {len(base_df)} -> {target_rows} records ...")

    n_copies = (target_rows // len(base_df)) + 1
    numeric_cols = base_df.select_dtypes(include=[np.number]).columns.tolist()

    # Columns that should stay untouched (identifiers / binary flags)
    id_cols = {"UDI"}
    binary_cols = {"Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"}
    perturb_cols = [c for c in numeric_cols
                    if c not in id_cols and c not in binary_cols]

    frames = [base_df.copy()]
    for copy_idx in range(1, n_copies):
        chunk = base_df.copy()

        # New unique UDI
        chunk["UDI"] = chunk["UDI"] + copy_idx * len(base_df)

        # New Product ID suffix
        if "Product ID" in chunk.columns:
            chunk["Product ID"] = chunk["Product ID"].apply(
                lambda x: x.split(",")[0] if "," in str(x) else str(x)
            )
            chunk["Product ID"] = (
                chunk["Product ID"].astype(str)
                + f"_aug{copy_idx}"
            )

        # Gaussian jitter on continuous sensor columns
        for col in perturb_cols:
            col_std = base_df[col].std()
            if col_std > 0:
                noise = rng.normal(loc=0, scale=col_std * 0.05,
                                   size=len(chunk))
                chunk[col] = chunk[col] + noise

        # Small probability of flipping failure labels to add variety
        for bcol in binary_cols:
            if bcol in chunk.columns:
                flip_mask = rng.random(size=len(chunk)) < 0.02
                chunk.loc[flip_mask, bcol] = 1 - chunk.loc[flip_mask, bcol]

        frames.append(chunk)

    augmented = pd.concat(frames, ignore_index=True).head(target_rows)
    augmented["UDI"] = range(1, len(augmented) + 1)

    print(f"       Final shape: {augmented.shape}")
    return augmented


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HT-HGNN v2.0 -- Augment the UCI Maintenance dataset",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=100_000,
        help="Target number of augmented records (default: 100000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print("HT-HGNN v2.0 -- Augment Maintenance Dataset")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Target rows  : {args.records}")
    print(f"Seed         : {args.seed}")

    start = time.time()

    # Load base
    base_df = _load_base(BASE_CSV)

    # Generate auxiliary tables
    topology = generate_machine_topology(rng)
    crews = generate_maintenance_crews(rng)
    parts = generate_spare_parts_inventory(rng)
    cascades = generate_failure_cascade_log(rng, topology)

    # Augment base
    augmented = augment_base_dataset(base_df, rng, target_rows=args.records)

    # Save all
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "machine_topology.csv": topology,
        "maintenance_crews.csv": crews,
        "spare_parts_inventory.csv": parts,
        "failure_cascade_log.csv": cascades,
        "ai4i2020_augmented.csv": augmented,
    }

    print("\n[save] Writing files ...")
    for fname, df in outputs.items():
        path = DATA_DIR / fname
        df.to_csv(path, index=False)
        print(f"  {fname:40s}  {len(df):>8,} rows  ->  {path}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
