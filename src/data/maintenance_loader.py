"""
Maintenance (AI4I 2020) Dataset Loader (HT-HGNN v2.0)

Loads the UCI AI4I 2020 Predictive Maintenance dataset (~10K records)
and constructs a hypergraph that maps failure modes to production-line
and infrastructure relationships.

Data path: Data set/Maintenance/ai4i2020.csv

Failure mode columns:
    TWF -- Tool Wear Failure
    HDF -- Heat Dissipation Failure
    PWF -- Power Failure
    OSF -- Overstrain Failure
    RNF -- Random Failure

Hyperedge types (mapped from failure modes):
    1. **Production line**       -- machines on the same production line
                                    (grouped by Type, e.g., L/M/H quality).
    2. **Shared maintenance crew** -- machines with similar maintenance
                                    profiles (failure co-occurrence).
    3. **Thermal zone**          -- machines operating in correlated thermal
                                    bands (Air/Process temperature clusters).
    4. **Power circuit**         -- machines sharing power-related failure
                                    characteristics (PWF-correlated).
    5. **Shared tooling**        -- machines that share similar tool wear
                                    patterns (TWF-correlated).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import sys

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.hypergraph.hypergraph import Hypergraph, HypergraphNode, HypergraphEdge


class MaintenanceLoader:
    """
    Loader for the UCI AI4I 2020 Predictive Maintenance dataset.

    Reads ``ai4i2020.csv`` (10,000 records), builds per-machine feature
    vectors, and constructs a hypergraph whose hyperedges encode shared
    production context (line, tooling, thermal zone, power circuit, and
    maintenance crew relationships).

    Node features (7 dimensions):
        0. air_temperature      -- ambient air temperature [K]
        1. process_temperature  -- manufacturing process temperature [K]
        2. rotational_speed     -- spindle rotational speed [rpm]
        3. torque               -- torque [Nm]
        4. tool_wear            -- cumulative tool wear [min]
        5. failure_flag         -- binary machine-failure indicator
        6. quality_type_encoded -- quality type (L=0, M=1, H=2)

    Example:
        loader = MaintenanceLoader(data_dir='Data set')
        data = loader.build_hypergraph()
    """

    FEATURE_COLUMNS = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'Machine failure',
    ]

    FAILURE_MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    def __init__(self, data_dir: str = "Data set"):
        """
        Initialize the Maintenance loader.

        Args:
            data_dir: Root path containing ``Maintenance/ai4i2020.csv``.
        """
        self.data_dir = Path(data_dir).resolve()
        self.csv_path = self.data_dir / "Maintenance" / "ai4i2020.csv"
        self.df: Optional[pd.DataFrame] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Load the AI4I 2020 Predictive Maintenance CSV.

        Returns:
            The loaded DataFrame.

        Raises:
            FileNotFoundError: If the CSV file is not at the expected path.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Maintenance dataset not found at {self.csv_path}. "
                f"Please place ai4i2020.csv inside "
                f"'{self.data_dir / 'Maintenance'}'."
            )

        print(f"Loading Maintenance dataset from {self.csv_path} ...")
        self.df = pd.read_csv(self.csv_path)
        self._loaded = True
        print(f"  Loaded {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
        return self.df

    # ------------------------------------------------------------------
    # Hypergraph construction
    # ------------------------------------------------------------------

    def build_hypergraph(
        self,
        temp_bins: int = 5,
        wear_bins: int = 4,
        min_hyperedge_size: int = 3,
    ) -> Dict[str, Any]:
        """
        Construct a maintenance hypergraph.

        Hyperedge types:
            * ``production_line`` -- machines of the same quality Type (L/M/H).
            * ``shared_maintenance_crew`` -- machines with overlapping failure
              modes (any pair of modes co-occurring).
            * ``thermal_zone`` -- machines binned into the same air/process
              temperature cluster.
            * ``power_circuit`` -- machines flagged with Power Failure (PWF) or
              similar torque/speed profiles.
            * ``shared_tooling`` -- machines in the same tool-wear bin.

        Args:
            temp_bins: Number of temperature bins for thermal zone hyperedges.
            wear_bins: Number of tool-wear bins for shared-tooling hyperedges.
            min_hyperedge_size: Minimum number of machines to form a hyperedge.

        Returns:
            Dictionary compatible with :class:`DataAdapter`.
        """
        if not self._loaded:
            self.load()

        df = self.df.copy()

        # --- Validate columns ---
        self._validate_columns(df)

        # --- Build node features ---
        features, node_ids, node_types, df = self._build_node_features(df)
        n_nodes = len(node_ids)
        node_idx_map = {nid: i for i, nid in enumerate(node_ids)}

        hypergraph = Hypergraph()
        for idx, nid in enumerate(node_ids):
            hypergraph.add_node(HypergraphNode(
                node_id=nid,
                node_type=node_types[idx],
                tier=1,
                lead_time=float(features[idx, 4]) / 60.0,  # tool wear as proxy
                reliability=1.0 - float(features[idx, 5]),
                substitutability=0.5,
                cost=float(features[idx, 3]),  # torque as proxy for load
            ))

        incidence_cols: List[np.ndarray] = []
        edge_types: List[str] = []
        hyperedge_weights: List[float] = []
        he_counter = 0

        # --- Type 1: Production line hyperedges (by Type: L/M/H) ---
        if 'Type' in df.columns:
            for machine_type, grp in df.groupby('Type'):
                member_indices = grp.index.tolist()
                if len(member_indices) < min_hyperedge_size:
                    continue

                col = np.zeros(n_nodes, dtype=np.float32)
                member_node_ids = set()
                for mi in member_indices:
                    col[mi] = 1.0
                    member_node_ids.add(node_ids[mi])

                failure_rate = float(grp['Machine failure'].mean()) if 'Machine failure' in grp.columns else 0.0

                he_id = f"HE_MNT_LINE_{he_counter:04d}"
                he_counter += 1
                incidence_cols.append(col)
                edge_types.append('production_line')
                hyperedge_weights.append(1.0 + failure_rate)

                hypergraph.add_hyperedge(HypergraphEdge(
                    hyperedge_id=he_id,
                    nodes=member_node_ids,
                    bom_weight=1.0 + failure_rate,
                    tolerance=0.5,
                    critical_path=1.0 if failure_rate > 0.05 else 0.0,
                    tier_level=1,
                ))

        # --- Type 2: Shared maintenance crew (failure mode co-occurrence) ---
        available_modes = [m for m in self.FAILURE_MODES if m in df.columns]
        mode_pairs = []
        for i in range(len(available_modes)):
            for j in range(i + 1, len(available_modes)):
                mode_pairs.append((available_modes[i], available_modes[j]))

        for mode_a, mode_b in mode_pairs:
            mask = (df[mode_a] == 1) & (df[mode_b] == 1)
            member_indices = df[mask].index.tolist()
            if len(member_indices) < min_hyperedge_size:
                continue

            col = np.zeros(n_nodes, dtype=np.float32)
            member_node_ids = set()
            for mi in member_indices:
                col[mi] = 1.0
                member_node_ids.add(node_ids[mi])

            he_id = f"HE_MNT_CREW_{he_counter:04d}"
            he_counter += 1
            incidence_cols.append(col)
            edge_types.append('shared_maintenance_crew')
            hyperedge_weights.append(float(len(member_indices)) / n_nodes * 10.0)

            hypergraph.add_hyperedge(HypergraphEdge(
                hyperedge_id=he_id,
                nodes=member_node_ids,
                bom_weight=float(len(member_indices)),
                tolerance=0.3,
                critical_path=1.0,
                tier_level=1,
            ))

        # --- Type 3: Thermal zone hyperedges ---
        if 'Air temperature [K]' in df.columns and 'Process temperature [K]' in df.columns:
            df['_air_temp_bin'] = pd.cut(
                df['Air temperature [K]'], bins=temp_bins, labels=False
            ).fillna(0).astype(int)
            df['_proc_temp_bin'] = pd.cut(
                df['Process temperature [K]'], bins=temp_bins, labels=False
            ).fillna(0).astype(int)
            df['_thermal_zone'] = df['_air_temp_bin'].astype(str) + '_' + df['_proc_temp_bin'].astype(str)

            for zone, grp in df.groupby('_thermal_zone'):
                member_indices = grp.index.tolist()
                if len(member_indices) < min_hyperedge_size:
                    continue

                col = np.zeros(n_nodes, dtype=np.float32)
                member_node_ids = set()
                for mi in member_indices:
                    col[mi] = 1.0
                    member_node_ids.add(node_ids[mi])

                he_id = f"HE_MNT_THERM_{he_counter:04d}"
                he_counter += 1

                # Higher weight if the zone has HDF failures
                hdf_rate = 0.0
                if 'HDF' in grp.columns:
                    hdf_rate = float(grp['HDF'].mean())
                incidence_cols.append(col)
                edge_types.append('thermal_zone')
                hyperedge_weights.append(1.0 + hdf_rate * 5.0)

                hypergraph.add_hyperedge(HypergraphEdge(
                    hyperedge_id=he_id,
                    nodes=member_node_ids,
                    bom_weight=1.0 + hdf_rate * 5.0,
                    tolerance=0.4,
                    critical_path=1.0 if hdf_rate > 0.05 else 0.0,
                    tier_level=1,
                ))

        # --- Type 4: Power circuit hyperedges ---
        if 'PWF' in df.columns:
            pwf_machines = df[df['PWF'] == 1].index.tolist()
            if len(pwf_machines) >= min_hyperedge_size:
                col = np.zeros(n_nodes, dtype=np.float32)
                member_node_ids = set()
                for mi in pwf_machines:
                    col[mi] = 1.0
                    member_node_ids.add(node_ids[mi])

                he_id = f"HE_MNT_PWR_{he_counter:04d}"
                he_counter += 1
                incidence_cols.append(col)
                edge_types.append('power_circuit')
                hyperedge_weights.append(2.0)

                hypergraph.add_hyperedge(HypergraphEdge(
                    hyperedge_id=he_id,
                    nodes=member_node_ids,
                    bom_weight=2.0,
                    tolerance=0.2,
                    critical_path=1.0,
                    tier_level=1,
                ))

        # --- Type 5: Shared tooling hyperedges (by tool wear bin) ---
        if 'Tool wear [min]' in df.columns:
            df['_wear_bin'] = pd.cut(
                df['Tool wear [min]'], bins=wear_bins, labels=False
            ).fillna(0).astype(int)

            for wear_bin, grp in df.groupby('_wear_bin'):
                member_indices = grp.index.tolist()
                if len(member_indices) < min_hyperedge_size:
                    continue

                col = np.zeros(n_nodes, dtype=np.float32)
                member_node_ids = set()
                for mi in member_indices:
                    col[mi] = 1.0
                    member_node_ids.add(node_ids[mi])

                twf_rate = 0.0
                if 'TWF' in grp.columns:
                    twf_rate = float(grp['TWF'].mean())

                he_id = f"HE_MNT_TOOL_{he_counter:04d}"
                he_counter += 1
                incidence_cols.append(col)
                edge_types.append('shared_tooling')
                hyperedge_weights.append(1.0 + twf_rate * 5.0)

                hypergraph.add_hyperedge(HypergraphEdge(
                    hyperedge_id=he_id,
                    nodes=member_node_ids,
                    bom_weight=1.0 + twf_rate * 5.0,
                    tolerance=0.3,
                    critical_path=1.0 if twf_rate > 0.02 else 0.0,
                    tier_level=1,
                ))

        # --- Assemble incidence matrix ---
        if incidence_cols:
            incidence_matrix = np.column_stack(incidence_cols)
        else:
            incidence_matrix = np.zeros((n_nodes, 1), dtype=np.float32)
            edge_types = ['empty']
            hyperedge_weights = [0.0]

        hw = np.array(hyperedge_weights, dtype=np.float32)
        hw_max = hw.max() if hw.max() > 0 else 1.0
        hw = hw / hw_max

        print(f"  Built maintenance hypergraph: {n_nodes:,} nodes, "
              f"{incidence_matrix.shape[1]} hyperedges "
              f"(production_line={edge_types.count('production_line')}, "
              f"maintenance_crew={edge_types.count('shared_maintenance_crew')}, "
              f"thermal_zone={edge_types.count('thermal_zone')}, "
              f"power_circuit={edge_types.count('power_circuit')}, "
              f"shared_tooling={edge_types.count('shared_tooling')})")

        return {
            'node_features': features,
            'incidence_matrix': incidence_matrix,
            'timestamps': None,
            'node_types': node_types,
            'edge_types': edge_types,
            'hyperedge_weights': hw,
            '_source': 'maintenance',
            'hypergraph': hypergraph,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_node_features(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str], List[str], pd.DataFrame]:
        """
        Build a 7-dimensional feature vector per machine.

        Returns:
            (features (N, 7), node_ids, node_types, updated df).
        """
        n = len(df)
        node_ids = [f"MNT_{i:05d}" for i in range(n)]

        # Quality type encoding
        type_map = {'L': 0, 'M': 1, 'H': 2}
        if 'Type' in df.columns:
            df['_type_encoded'] = df['Type'].map(type_map).fillna(0).astype(int)
            node_types = df['Type'].astype(str).tolist()
        else:
            df['_type_encoded'] = 0
            node_types = ['unknown'] * n

        feature_cols_present = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        for col in feature_cols_present:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        features = np.zeros((n, 7), dtype=np.float64)
        for feat_idx, col in enumerate(self.FEATURE_COLUMNS):
            if col in df.columns:
                features[:, feat_idx] = df[col].values.astype(np.float64)
        features[:, 6] = df['_type_encoded'].values.astype(np.float64)

        return features, node_ids, node_types, df

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        """Warn about missing expected columns."""
        expected = {
            'Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
        }
        missing = expected - set(df.columns)
        if missing:
            print(f"  WARNING: Missing columns: {missing}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return summary statistics about the maintenance dataset.

        Returns:
            Dict with record count, failure rates, and type breakdown.
        """
        if not self._loaded:
            return {'status': 'not_loaded'}

        df = self.df
        stats: Dict[str, Any] = {
            'n_records': len(df),
            'n_columns': len(df.columns),
            'columns': list(df.columns),
        }
        if 'Machine failure' in df.columns:
            stats['failure_rate'] = float(df['Machine failure'].mean())
        if 'Type' in df.columns:
            stats['type_distribution'] = df['Type'].value_counts().to_dict()
        for mode in self.FAILURE_MODES:
            if mode in df.columns:
                stats[f'{mode}_count'] = int(df[mode].sum())
        return stats


if __name__ == "__main__":
    print("=" * 60)
    print("MaintenanceLoader - Predictive Maintenance Dataset Loader")
    print("HT-HGNN v2.0 Data Pipeline")
    print("=" * 60)
    print()
    print("Dataset: UCI AI4I 2020 Predictive Maintenance (~10K records)")
    print("Path:    Data set/Maintenance/ai4i2020.csv")
    print()
    print("Failure modes: TWF, HDF, PWF, OSF, RNF")
    print()
    print("Hyperedge types:")
    print("  - production_line        (by machine quality Type L/M/H)")
    print("  - shared_maintenance_crew (failure-mode co-occurrence)")
    print("  - thermal_zone           (air/process temperature clusters)")
    print("  - power_circuit          (PWF-correlated machines)")
    print("  - shared_tooling         (tool-wear bin grouping)")
    print()
    print("Usage:")
    print("  loader = MaintenanceLoader(data_dir='Data set')")
    print("  data   = loader.build_hypergraph()")
    print()
    print("Module loaded successfully.")
