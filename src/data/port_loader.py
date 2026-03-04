"""
Global Port Disruption Loader (HT-HGNN v2.0)

Loads port disruption data from ``Data set/Ports/`` and constructs a
temporal hypergraph modelling global shipping infrastructure.

Expected files (graceful fallback to synthetic sample when absent):
    - port_nodes.csv          -- port metadata (location, capacity, etc.)
    - disruption_events.csv   -- historical disruption records
    - shipping_lanes.csv      -- origin-destination lane definitions

Hyperedge types:
    1. **Shipping corridor** -- ports connected by a common trade lane.
    2. **Congestion cluster** -- ports that experience correlated congestion.
    3. **Geopolitical zone**  -- ports within the same geopolitical risk region.

Supports temporal slicing via ``build_temporal_hypergraph()``.
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


class PortDisruptionLoader:
    """
    Loader for global port disruption data.

    If real data files exist under ``Data set/Ports/`` they are loaded
    directly. Otherwise the loader generates a realistic synthetic sample
    so that downstream code can still be developed and tested.

    Node features (6 dimensions):
        0. throughput_capacity  -- annual TEU capacity (normalised)
        1. avg_dwell_time       -- average vessel dwell time in days
        2. congestion_index     -- current congestion level [0, 1]
        3. disruption_history   -- historical disruption frequency [0, 1]
        4. connectivity_degree  -- number of connected shipping lanes
        5. geopolitical_risk    -- composite geopolitical risk score [0, 1]

    Example:
        loader = PortDisruptionLoader(data_dir='Data set')
        data = loader.build_temporal_hypergraph(2020, 2024, temporal_window_months=6)
    """

    FEATURE_NAMES = [
        'throughput_capacity',
        'avg_dwell_time',
        'congestion_index',
        'disruption_history',
        'connectivity_degree',
        'geopolitical_risk',
    ]

    # Representative ports for synthetic fallback
    _SAMPLE_PORTS = [
        ('Shanghai', 'East Asia', 47.0),
        ('Singapore', 'Southeast Asia', 37.0),
        ('Ningbo-Zhoushan', 'East Asia', 31.0),
        ('Shenzhen', 'East Asia', 26.0),
        ('Guangzhou', 'East Asia', 23.0),
        ('Busan', 'East Asia', 22.0),
        ('Qingdao', 'East Asia', 21.0),
        ('Rotterdam', 'Europe', 14.0),
        ('Dubai (Jebel Ali)', 'Middle East', 13.5),
        ('Tianjin', 'East Asia', 13.0),
        ('Port Klang', 'Southeast Asia', 12.0),
        ('Antwerp', 'Europe', 11.0),
        ('Xiamen', 'East Asia', 10.0),
        ('Kaohsiung', 'East Asia', 9.5),
        ('Los Angeles', 'North America', 9.0),
        ('Hamburg', 'Europe', 8.5),
        ('Tanjung Pelepas', 'Southeast Asia', 8.0),
        ('Long Beach', 'North America', 7.5),
        ('Laem Chabang', 'Southeast Asia', 7.0),
        ('Ho Chi Minh City', 'Southeast Asia', 6.5),
        ('Colombo', 'South Asia', 6.0),
        ('Piraeus', 'Europe', 5.5),
        ('Savannah', 'North America', 5.0),
        ('Durban', 'Africa', 3.0),
        ('Santos', 'South America', 3.5),
    ]

    # Sample trade lanes (origin_idx, dest_idx)
    _SAMPLE_LANES = [
        (0, 7), (0, 14), (0, 1), (1, 7), (1, 14), (1, 8),
        (2, 7), (2, 14), (3, 14), (3, 17), (4, 1), (5, 14),
        (6, 7), (7, 11), (7, 14), (8, 1), (8, 20), (9, 7),
        (10, 1), (10, 8), (11, 7), (12, 14), (13, 14),
        (14, 17), (15, 11), (16, 1), (18, 1), (19, 1),
        (20, 8), (21, 7), (22, 14), (23, 8), (24, 7),
    ]

    def __init__(self, data_dir: str = "Data set"):
        """
        Initialize the port disruption loader.

        Args:
            data_dir: Root path containing ``Ports/`` subfolder.
        """
        self.data_dir = Path(data_dir).resolve()
        self.ports_dir = self.data_dir / "Ports"

        self.ports_df: Optional[pd.DataFrame] = None
        self.disruptions_df: Optional[pd.DataFrame] = None
        self.lanes_df: Optional[pd.DataFrame] = None
        self._is_synthetic = False
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Load port data files or fall back to synthetic generation.

        Returns:
            Dict with keys ``'ports'``, ``'disruptions'``, ``'lanes'``.
        """
        ports_path = self.ports_dir / "port_nodes.csv"
        disruptions_path = self.ports_dir / "disruption_events.csv"
        lanes_path = self.ports_dir / "shipping_lanes.csv"

        if ports_path.exists() and lanes_path.exists():
            print(f"Loading port data from {self.ports_dir} ...")
            self.ports_df = pd.read_csv(ports_path)
            self.lanes_df = pd.read_csv(lanes_path)
            if disruptions_path.exists():
                self.disruptions_df = pd.read_csv(disruptions_path)
            else:
                print("  WARNING: disruption_events.csv not found; "
                      "generating synthetic disruptions.")
                self.disruptions_df = self._generate_synthetic_disruptions(
                    n_ports=len(self.ports_df)
                )
            self._is_synthetic = False
        else:
            print(f"  Port data files not found at {self.ports_dir}.")
            print("  Generating synthetic port network for development ...")
            self.ports_df, self.disruptions_df, self.lanes_df = (
                self._generate_synthetic_data()
            )
            self._is_synthetic = True

        self._loaded = True
        print(f"  Ports: {len(self.ports_df)} | "
              f"Disruption events: {len(self.disruptions_df)} | "
              f"Shipping lanes: {len(self.lanes_df)}")

        return {
            'ports': self.ports_df,
            'disruptions': self.disruptions_df,
            'lanes': self.lanes_df,
        }

    # ------------------------------------------------------------------
    # Synthetic data generation (fallback)
    # ------------------------------------------------------------------

    def _generate_synthetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate a realistic small-scale synthetic port dataset."""
        rng = np.random.RandomState(42)

        # --- Ports ---
        records = []
        for i, (name, region, teu) in enumerate(self._SAMPLE_PORTS):
            records.append({
                'port_id': f"PORT_{i:03d}",
                'port_name': name,
                'region': region,
                'throughput_mTEU': teu,
                'avg_dwell_time': round(rng.uniform(1.5, 7.0), 2),
                'congestion_index': round(rng.uniform(0.1, 0.9), 3),
                'geopolitical_risk': round(rng.uniform(0.05, 0.8), 3),
            })
        ports_df = pd.DataFrame(records)

        # --- Lanes ---
        lane_records = []
        for idx, (o, d) in enumerate(self._SAMPLE_LANES):
            lane_records.append({
                'lane_id': f"LANE_{idx:03d}",
                'origin_port_id': f"PORT_{o:03d}",
                'dest_port_id': f"PORT_{d:03d}",
                'distance_nm': int(rng.uniform(500, 12000)),
                'transit_days': int(rng.uniform(3, 35)),
            })
        lanes_df = pd.DataFrame(lane_records)

        # --- Disruptions ---
        disruptions_df = self._generate_synthetic_disruptions(
            n_ports=len(ports_df), rng=rng
        )

        return ports_df, disruptions_df, lanes_df

    @staticmethod
    def _generate_synthetic_disruptions(
        n_ports: int,
        n_events: int = 200,
        rng: Optional[np.random.RandomState] = None,
    ) -> pd.DataFrame:
        """Generate synthetic disruption events across ports."""
        if rng is None:
            rng = np.random.RandomState(123)

        event_types = ['typhoon', 'labour_strike', 'equipment_failure',
                       'pandemic_lockdown', 'canal_blockage', 'cyber_attack']
        records = []
        for i in range(n_events):
            year = rng.choice(range(2018, 2026))
            month = rng.randint(1, 13)
            records.append({
                'event_id': f"EVT_{i:04d}",
                'port_id': f"PORT_{rng.randint(0, n_ports):03d}",
                'year': year,
                'month': month,
                'event_type': rng.choice(event_types),
                'severity': round(rng.uniform(0.1, 1.0), 3),
                'duration_days': int(rng.uniform(1, 60)),
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Hypergraph construction
    # ------------------------------------------------------------------

    def build_temporal_hypergraph(
        self,
        start_year: int = 2020,
        end_year: int = 2024,
        temporal_window_months: int = 6,
    ) -> Dict[str, Any]:
        """
        Build a temporal hypergraph over the specified time range.

        Hyperedge types:
            * ``shipping_corridor`` -- all ports along a trade lane.
            * ``congestion_cluster`` -- ports that simultaneously exceed a
              congestion threshold within the same temporal window.
            * ``geopolitical_zone`` -- ports grouped by geopolitical region.

        Args:
            start_year: First year of the temporal range.
            end_year: Last year of the temporal range (inclusive).
            temporal_window_months: Width of each temporal snapshot in months.

        Returns:
            Dictionary compatible with :class:`DataAdapter`.
        """
        if not self._loaded:
            self.load()

        ports = self.ports_df.copy()
        lanes = self.lanes_df.copy()
        disruptions = self.disruptions_df.copy()

        # --- Node features ---
        n_ports = len(ports)
        node_ids = ports['port_id'].tolist() if 'port_id' in ports.columns else [
            f"PORT_{i:03d}" for i in range(n_ports)
        ]
        port_to_idx = {pid: i for i, pid in enumerate(node_ids)}

        features = np.zeros((n_ports, 6), dtype=np.float64)

        # Fill features from available columns
        col_map = {
            0: 'throughput_mTEU',
            1: 'avg_dwell_time',
            2: 'congestion_index',
            3: None,   # computed from disruptions
            4: None,   # computed from lanes
            5: 'geopolitical_risk',
        }
        for feat_idx, col_name in col_map.items():
            if col_name and col_name in ports.columns:
                features[:, feat_idx] = ports[col_name].values.astype(np.float64)

        # disruption_history: count per port normalised
        if 'port_id' in disruptions.columns:
            dis_counts = disruptions['port_id'].value_counts()
            for pid, cnt in dis_counts.items():
                if pid in port_to_idx:
                    features[port_to_idx[pid], 3] = cnt
            max_dis = features[:, 3].max()
            if max_dis > 0:
                features[:, 3] /= max_dis

        # connectivity_degree: count lanes per port
        if 'origin_port_id' in lanes.columns and 'dest_port_id' in lanes.columns:
            for _, row in lanes.iterrows():
                o = row['origin_port_id']
                d = row['dest_port_id']
                if o in port_to_idx:
                    features[port_to_idx[o], 4] += 1
                if d in port_to_idx:
                    features[port_to_idx[d], 4] += 1
            max_conn = features[:, 4].max()
            if max_conn > 0:
                features[:, 4] /= max_conn

        # --- Build hypergraph ---
        hypergraph = Hypergraph()
        node_types: List[str] = []

        for idx, nid in enumerate(node_ids):
            region = 'unknown'
            if 'region' in ports.columns:
                region = str(ports.iloc[idx]['region'])
            node_types.append(region)

            hypergraph.add_node(HypergraphNode(
                node_id=nid,
                node_type=region,
                tier=1,
                lead_time=float(features[idx, 1]),
                reliability=1.0 - float(features[idx, 3]),
                substitutability=float(features[idx, 4]),
                cost=float(features[idx, 0]),
            ))

        incidence_cols: List[np.ndarray] = []
        edge_types: List[str] = []
        timestamps: List[float] = []
        hyperedge_weights: List[float] = []
        he_counter = 0

        # --- Type 1: Shipping corridor hyperedges ---
        if 'origin_port_id' in lanes.columns and 'dest_port_id' in lanes.columns:
            # Group lanes that share an origin into corridor hyperedges
            for origin, grp in lanes.groupby('origin_port_id'):
                dest_ports = grp['dest_port_id'].unique().tolist()
                member_ports = [origin] + dest_ports
                member_indices = [port_to_idx[p] for p in member_ports if p in port_to_idx]

                if len(member_indices) < 2:
                    continue

                col = np.zeros(n_ports, dtype=np.float32)
                member_node_ids = set()
                for mi in member_indices:
                    col[mi] = 1.0
                    member_node_ids.add(node_ids[mi])

                he_id = f"HE_PORT_COR_{he_counter:04d}"
                he_counter += 1
                incidence_cols.append(col)
                edge_types.append('shipping_corridor')
                timestamps.append(float(start_year))
                hyperedge_weights.append(float(len(member_indices)))

                hypergraph.add_hyperedge(HypergraphEdge(
                    hyperedge_id=he_id,
                    nodes=member_node_ids,
                    bom_weight=float(len(member_indices)),
                    tolerance=0.5,
                    critical_path=1.0,
                    tier_level=1,
                ))

        # --- Type 2: Congestion cluster hyperedges (per temporal window) ---
        congestion_threshold = 0.5
        for year in range(start_year, end_year + 1):
            for window_start_month in range(1, 13, temporal_window_months):
                window_end_month = min(window_start_month + temporal_window_months - 1, 12)
                ts = year + (window_start_month - 1) / 12.0

                # Find ports with disruptions in this window
                mask = (
                    (disruptions['year'] == year) &
                    (disruptions['month'] >= window_start_month) &
                    (disruptions['month'] <= window_end_month)
                )
                window_disruptions = disruptions[mask]

                if len(window_disruptions) == 0:
                    continue

                affected_ports = window_disruptions['port_id'].unique().tolist()
                # Also add ports with high base congestion
                high_cong = [
                    node_ids[i] for i in range(n_ports)
                    if features[i, 2] >= congestion_threshold
                ]
                cluster_ports = list(set(affected_ports + high_cong))
                member_indices = [port_to_idx[p] for p in cluster_ports if p in port_to_idx]

                if len(member_indices) < 2:
                    continue

                col = np.zeros(n_ports, dtype=np.float32)
                member_node_ids = set()
                for mi in member_indices:
                    col[mi] = 1.0
                    member_node_ids.add(node_ids[mi])

                avg_severity = float(window_disruptions['severity'].mean()) if 'severity' in window_disruptions.columns else 0.5

                he_id = f"HE_PORT_CON_{he_counter:04d}"
                he_counter += 1
                incidence_cols.append(col)
                edge_types.append('congestion_cluster')
                timestamps.append(ts)
                hyperedge_weights.append(avg_severity)

                hypergraph.add_hyperedge(HypergraphEdge(
                    hyperedge_id=he_id,
                    nodes=member_node_ids,
                    bom_weight=avg_severity,
                    tolerance=0.3,
                    critical_path=1.0 if avg_severity > 0.7 else 0.0,
                    tier_level=1,
                ))

        # --- Type 3: Geopolitical zone hyperedges ---
        if 'region' in ports.columns:
            for region, grp in ports.groupby('region'):
                member_port_ids = grp['port_id'].tolist() if 'port_id' in grp.columns else []
                member_indices = [port_to_idx[p] for p in member_port_ids if p in port_to_idx]

                if len(member_indices) < 2:
                    continue

                col = np.zeros(n_ports, dtype=np.float32)
                member_node_ids = set()
                for mi in member_indices:
                    col[mi] = 1.0
                    member_node_ids.add(node_ids[mi])

                avg_geo_risk = float(grp['geopolitical_risk'].mean()) if 'geopolitical_risk' in grp.columns else 0.5

                he_id = f"HE_PORT_GEO_{he_counter:04d}"
                he_counter += 1
                incidence_cols.append(col)
                edge_types.append('geopolitical_zone')
                timestamps.append(float(start_year))
                hyperedge_weights.append(avg_geo_risk)

                hypergraph.add_hyperedge(HypergraphEdge(
                    hyperedge_id=he_id,
                    nodes=member_node_ids,
                    bom_weight=avg_geo_risk,
                    tolerance=0.4,
                    critical_path=1.0 if avg_geo_risk > 0.6 else 0.0,
                    tier_level=1,
                ))

        # --- Assemble incidence matrix ---
        if incidence_cols:
            incidence_matrix = np.column_stack(incidence_cols)
        else:
            incidence_matrix = np.zeros((n_ports, 1), dtype=np.float32)
            edge_types = ['empty']
            timestamps = [float(start_year)]
            hyperedge_weights = [0.0]

        hw = np.array(hyperedge_weights, dtype=np.float32)
        hw_max = hw.max() if hw.max() > 0 else 1.0
        hw = hw / hw_max

        synth_tag = " (synthetic)" if self._is_synthetic else ""
        print(f"  Built port hypergraph{synth_tag}: {n_ports} nodes, "
              f"{incidence_matrix.shape[1]} hyperedges "
              f"(corridors={edge_types.count('shipping_corridor')}, "
              f"congestion={edge_types.count('congestion_cluster')}, "
              f"geopolitical={edge_types.count('geopolitical_zone')})")

        return {
            'node_features': features,
            'incidence_matrix': incidence_matrix,
            'timestamps': timestamps,
            'node_types': node_types,
            'edge_types': edge_types,
            'hyperedge_weights': hw,
            '_source': 'ports',
            'hypergraph': hypergraph,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return summary statistics about the loaded port data.

        Returns:
            Dict with port, lane, and disruption counts.
        """
        if not self._loaded:
            return {'status': 'not_loaded'}

        stats: Dict[str, Any] = {
            'is_synthetic': self._is_synthetic,
            'n_ports': len(self.ports_df) if self.ports_df is not None else 0,
            'n_lanes': len(self.lanes_df) if self.lanes_df is not None else 0,
            'n_disruptions': len(self.disruptions_df) if self.disruptions_df is not None else 0,
        }
        if self.ports_df is not None and 'region' in self.ports_df.columns:
            stats['regions'] = self.ports_df['region'].value_counts().to_dict()
        return stats


if __name__ == "__main__":
    print("=" * 60)
    print("PortDisruptionLoader - Global Port Disruption Loader")
    print("HT-HGNN v2.0 Data Pipeline")
    print("=" * 60)
    print()
    print("Dataset: Port disruption (port_nodes, disruption_events, shipping_lanes)")
    print("Path:    Data set/Ports/")
    print("Note:    Falls back to synthetic data if files are absent.")
    print()
    print("Node features (6-dim):")
    for i, name in enumerate(PortDisruptionLoader.FEATURE_NAMES):
        print(f"  [{i}] {name}")
    print()
    print("Hyperedge types: shipping_corridor, congestion_cluster, geopolitical_zone")
    print()
    print("Usage:")
    print("  loader = PortDisruptionLoader(data_dir='Data set')")
    print("  data   = loader.build_temporal_hypergraph(2020, 2024, 6)")
    print()
    print("Module loaded successfully.")
