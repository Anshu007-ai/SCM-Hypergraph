"""
DataCo Supply Chain Dataset Loader (HT-HGNN v2.0)

Loads the DataCo Supply Chain Dataset (~180K records, 52 features) and
constructs a supply-chain hypergraph where orders are nodes and hyperedges
represent groups of orders that share a common shipping route corridor,
delivery window, and product category.

Key columns used:
    Late_delivery_risk, Order_Item_Profit_Ratio, Shipping_Mode,
    Customer_Segment, Order_Country, Order_Region, Category_Name,
    order_date_(DateOrders), Days_for_shipping_(real),
    Days_for_shipment_(scheduled), Benefit_per_order, Sales_per_customer

Data path: Data set/DataCo/DataCoSupplyChainDataset.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import torch

import sys
import os

# Ensure project root is importable
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.hypergraph.hypergraph import Hypergraph, HypergraphNode, HypergraphEdge


class DataCoLoader:
    """
    Loader for the DataCo Supply Chain Dataset.

    Reads ``DataCoSupplyChainDataset.csv`` (180,519 records x 52 columns),
    extracts order-level features, and builds a hypergraph in which:
        - **Nodes** represent individual orders (or order-items).
        - **Hyperedges** group orders that share the same shipping route
          corridor, delivery time window, and product category.

    The loader returns a dictionary that is directly compatible with
    :class:`DataAdapter` for ingestion into the HT-HGNN pipeline.

    Example:
        loader = DataCoLoader(data_dir='Data set')
        hg_data = loader.build_hypergraph(window_days=30, min_hyperedge_size=3)
    """

    # Columns we try to read from the CSV
    KEY_COLUMNS = [
        'Late_delivery_risk', 'Order_Item_Profit_Ratio', 'Shipping Mode',
        'Customer Segment', 'Order Country', 'Order Region',
        'Category Name', 'order date (DateOrders)',
        'Days for shipping (real)', 'Days for shipment (scheduled)',
        'Benefit per order', 'Sales per customer',
        'Order Item Discount', 'Order Item Quantity',
        'Order Item Total', 'Order Profit Per Order',
    ]

    # 8 numeric features used for node feature vectors
    FEATURE_COLUMNS = [
        'Late_delivery_risk',
        'Order_Item_Profit_Ratio',
        'Days for shipping (real)',
        'Days for shipment (scheduled)',
        'Benefit per order',
        'Sales per customer',
        'Order Item Discount',
        'Order Item Quantity',
    ]

    def __init__(self, data_dir: str = "Data set", use_synthetic: bool = False):
        """
        Initialize the DataCo loader.

        Args:
            data_dir: Root path to the ``Data set`` folder that contains
                      ``DataCo/DataCoSupplyChainDataset.csv``.
            use_synthetic: If True, force generation of synthetic data
                           even if the CSV exists.
        """
        self.data_dir = Path(data_dir).resolve()
        self.file_path = self.data_dir / "DataCo" / "DataCoSupplyChainDataset.csv"
        self.csv_path = self.file_path # For compatibility
        self.df: Optional[pd.DataFrame] = None
        self._loaded = False
        self.is_synthetic = False

        if self.file_path.exists() and not use_synthetic:
            print(f"INFO: Found DataCo CSV at: {self.file_path}")
            self.df = self.load()
        else:
            print(f"WARNING: DataCo CSV not found at {self.file_path} or synthetic forced.")
            print("Generating synthetic data as a fallback.")
            self.df = self.generate_synthetic_data()
            self.is_synthetic = True

    def generate_synthetic_data(self, num_rows=1000) -> pd.DataFrame:
        """Generates a synthetic DataFrame with the same schema."""
        print("Generating synthetic DataCo data...")
        data = {
            'Late_delivery_risk': np.random.randint(0, 2, size=num_rows),
            'Order_Item_Profit_Ratio': np.random.uniform(-0.5, 0.5, size=num_rows),
            'Shipping Mode': np.random.choice(['Standard Class', 'First Class', 'Second Class', 'Same Day'], size=num_rows),
            'Customer Segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], size=num_rows),
            'Order Country': np.random.choice(['USA', 'Mexico', 'Brazil', 'Germany', 'Australia'], size=num_rows),
            'Order Region': np.random.choice(['South', 'North', 'West', 'East', 'Central'], size=num_rows),
            'Category Name': np.random.choice(['Fishing', 'Camping', 'Fitness', 'Golf', 'Apparel'], size=num_rows),
            'order date (DateOrders)': [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(num_rows)],
            'Days for shipping (real)': np.random.randint(1, 10, size=num_rows),
            'Days for shipment (scheduled)': np.random.randint(1, 10, size=num_rows),
            'Benefit per order': np.random.uniform(10, 100, size=num_rows),
            'Sales per customer': np.random.uniform(100, 1000, size=num_rows),
            'Order Item Discount': np.random.uniform(0, 50, size=num_rows),
            'Order Item Quantity': np.random.randint(1, 5, size=num_rows),
            'Order Item Total': np.random.uniform(100, 2000, size=num_rows),
            'Order Profit Per Order': np.random.uniform(-50, 150, size=num_rows),
        }
        df = pd.DataFrame(data)
        # Add a unique ID for each row to act as a node identifier
        df['Order Id'] = range(num_rows)
        return df


    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load the DataCo CSV file into a pandas DataFrame.

        Args:
            max_rows: If set, only read the first ``max_rows`` rows (useful
                      for quick prototyping on a 180K row file).

        Returns:
            The loaded DataFrame.

        Raises:
            FileNotFoundError: If the CSV file is not at the expected path.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"DataCo dataset not found at {self.csv_path}. "
                f"Please place DataCoSupplyChainDataset.csv inside "
                f"'{self.data_dir / 'DataCo'}'."
            )

        print(f"Loading DataCo dataset from {self.csv_path} ...")
        self.df = pd.read_csv(
            self.csv_path,
            encoding='latin-1',
            nrows=max_rows,
        )
        self._loaded = True

        print(f"  Loaded {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
        return self.df

    # ------------------------------------------------------------------
    # Hypergraph construction
    # ------------------------------------------------------------------

    def build_hypergraph(
        self,
        window_days: int = 30,
        min_hyperedge_size: int = 3,
        risk_threshold: float = 0.6,
        max_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Construct a hypergraph from the DataCo dataset.

        Hyperedge construction logic:
            Orders are grouped into the same hyperedge when they share the
            same *shipping route corridor* (Order Country + Shipping Mode),
            fall within the same *delivery window* (``window_days``-wide
            buckets), and belong to the same *product category*.

        Args:
            window_days: Width of the temporal delivery window used to bucket
                         orders (default 30 days).
            min_hyperedge_size: Minimum number of orders required to form a
                                valid hyperedge (default 3).
            risk_threshold: Late-delivery risk threshold used to tag
                            high-risk hyperedges with elevated weight
                            (default 0.6).
            max_rows: Optionally limit rows for faster iteration.

        Returns:
            Dictionary compatible with :class:`DataAdapter`::

                {
                    'node_features':     np.ndarray (N, 8),
                    'incidence_matrix':  torch.sparse_coo_tensor (N, M),
                    'timestamps':        list[float] length M,
                    'node_types':        list[str]   length N,
                    'edge_types':        list[str]   length M,
                    'hyperedge_weights': np.ndarray  length M,
                    '_source':           'dataco',
                    'hypergraph':        Hypergraph object,
                }
        """
        if not self._loaded or (max_rows and max_rows != len(self.df)):
            self.load(max_rows=max_rows)

        df = self.df.copy()

        # --- Ensure essential columns exist ---
        self._validate_columns(df)

        # --- Parse dates and create temporal buckets ---
        df = self._parse_dates(df, window_days)

        # --- Build grouping key for hyperedges ---
        df['_shipping_corridor'] = (
            df['Order Country'].astype(str) + '|' +
            df['Shipping Mode'].astype(str)
        )
        df['_group_key'] = (
            df['_shipping_corridor'] + '|' +
            df['_time_bucket'].astype(str) + '|' +
            df['Category Name'].astype(str)
        )

        # --- Build node features ---
        feature_cols = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        node_features = df[feature_cols].values.astype(np.float64)
        node_ids = [f"DCO_{i:06d}" for i in range(len(df))]

        # --- Determine node types from Customer Segment ---
        if 'Customer Segment' in df.columns:
            node_types = df['Customer Segment'].astype(str).tolist()
        else:
            node_types = ['order'] * len(df)

        # --- Construct hyperedges via grouping ---
        groups = df.groupby('_group_key')

        hypergraph = Hypergraph()
        row_indices: List[int] = []
        col_indices: List[int] = []
        edge_types: List[str] = []
        timestamps: List[float] = []
        hyperedge_weights: List[float] = []

        node_idx_map = {nid: i for i, nid in enumerate(node_ids)}
        n_nodes = len(node_ids)

        # Add nodes to hypergraph
        for idx, nid in enumerate(node_ids):
            node = HypergraphNode(
                node_id=nid,
                node_type=node_types[idx],
                tier=1,
                lead_time=float(node_features[idx, 2]) if node_features.shape[1] > 2 else 0.0,
                reliability=1.0 - float(node_features[idx, 0]) if node_features.shape[1] > 0 else 0.8,
                substitutability=0.5,
                cost=float(node_features[idx, 4]) if node_features.shape[1] > 4 else 0.0,
                raw_data_idx=idx  # This was the missing attribute
            )
            hypergraph.add_node(node)

        he_counter = 0
        for group_key, group_df in groups:
            if len(group_df) < min_hyperedge_size:
                continue

            member_indices = group_df.index.tolist()
            member_node_ids = {node_ids[i] for i in member_indices}

            # Edge type from shipping mode
            parts = str(group_key).split('|')
            shipping_mode = parts[1] if len(parts) > 1 else 'unknown'
            etype = f"corridor_{shipping_mode}"

            # Timestamp: median of the time bucket
            ts_val = group_df['_time_bucket'].iloc[0] if '_time_bucket' in group_df.columns else 0.0

            # Weight: elevated if average late-delivery risk exceeds threshold
            avg_risk = 0.0
            if 'Late_delivery_risk' in group_df.columns:
                avg_risk = group_df['Late_delivery_risk'].mean()
            weight = 1.0 + avg_risk if avg_risk >= risk_threshold else 1.0

            he_id = f"HE_DCO_{he_counter:05d}"

            # Populate sparse indices
            for member_idx in member_indices:
                row_indices.append(member_idx)
                col_indices.append(he_counter)

            he_counter += 1

            edge_types.append(etype)
            timestamps.append(float(ts_val))
            hyperedge_weights.append(weight)

            # Add to Hypergraph object
            he = HypergraphEdge(
                hyperedge_id=he_id,
                nodes=member_node_ids,
                bom_weight=weight,
                tolerance=0.5,
                critical_path=1.0 if avg_risk >= risk_threshold else 0.0,
                tier_level=1,
            )
            hypergraph.add_hyperedge(he)

        # --- Assemble sparse incidence matrix (N x M) ---
        if he_counter > 0:
            indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
            values = torch.ones(len(row_indices), dtype=torch.float32)
            incidence_matrix = torch.sparse_coo_tensor(
                indices, values, size=(n_nodes, he_counter)
            ).coalesce()
        else:
            incidence_matrix = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                (n_nodes, 0)
            )
            edge_types = ['empty']
            timestamps = [0.0]
            hyperedge_weights = [0.0]

        print(f"  Built hypergraph: {n_nodes:,} nodes, "
              f"{incidence_matrix.shape[1]:,} hyperedges")

        return {
            'node_features': node_features,
            'incidence_matrix': incidence_matrix,
            'timestamps': timestamps,
            'node_types': node_types,
            'edge_types': edge_types,
            'hyperedge_weights': np.array(hyperedge_weights, dtype=np.float32),
            '_source': 'dataco',
            'hypergraph': hypergraph,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        """Warn about missing columns but do not fail."""
        expected = {
            'Late_delivery_risk', 'Order_Item_Profit_Ratio',
            'Shipping Mode', 'Customer Segment', 'Order Country',
            'Order Region', 'Category Name',
        }
        missing = expected - set(df.columns)
        if missing:
            print(f"  WARNING: Missing expected columns: {missing}")
            print(f"  Available columns ({len(df.columns)}): "
                  f"{list(df.columns)[:15]}...")

    @staticmethod
    def _parse_dates(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
        """Parse order dates and create temporal bucket indices."""
        date_col = None
        for candidate in ['order date (DateOrders)', 'order_date_(DateOrders)',
                          'order_date', 'Order_Date']:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col is not None:
            # Ensure we handle potential timezone issues by making it timezone-naive
            parsed_dates = pd.to_datetime(df[date_col], errors='coerce').dt.tz_localize(None)
            df = df.assign(_parsed_date=parsed_dates).dropna(subset=['_parsed_date'])
            
            min_date = df['_parsed_date'].min()
            
            # Perform calculation on the Series directly to avoid alignment issues
            time_delta_days = (df['_parsed_date'] - min_date).dt.days
            df['_time_bucket'] = (time_delta_days // window_days).fillna(0).astype(int)
        else:
            # Fallback: assign all to bucket 0
            df['_time_bucket'] = 0
            print("  WARNING: No date column found; all orders placed in bucket 0.")

        return df

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return summary statistics about the loaded dataset.

        Returns:
            Dict with record counts, column info, and date range.
        """
        if not self._loaded:
            return {'status': 'not_loaded'}

        stats: Dict[str, Any] = {
            'n_records': len(self.df),
            'n_columns': len(self.df.columns),
            'columns': list(self.df.columns),
        }

        if 'Late_delivery_risk' in self.df.columns:
            stats['late_delivery_pct'] = float(self.df['Late_delivery_risk'].mean())
        if 'Shipping Mode' in self.df.columns:
            stats['shipping_modes'] = self.df['Shipping Mode'].value_counts().to_dict()
        if 'Customer Segment' in self.df.columns:
            stats['customer_segments'] = self.df['Customer Segment'].value_counts().to_dict()

        return stats


if __name__ == "__main__":
    print("=" * 60)
    print("DataCoLoader - DataCo Supply Chain Dataset Loader")
    print("HT-HGNN v2.0 Data Pipeline")
    print("=" * 60)
    print()
    print("Dataset: DataCo Supply Chain (~180K records, 52 features)")
    print("Path:    Data set/DataCo/DataCoSupplyChainDataset.csv")
    print()
    print("Hyperedge construction:")
    print("  - Groups orders by shipping corridor + delivery window + category")
    print("  - Configurable window_days, min_hyperedge_size, risk_threshold")
    print()
    print("Usage:")
    print("  loader = DataCoLoader(data_dir='Data set')")
    print("  data   = loader.build_hypergraph(window_days=30)")
    print()
    print("Module loaded successfully.")
