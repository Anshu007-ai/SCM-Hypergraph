"""
Automotive Bill of Materials (BOM) Loader (HT-HGNN v2.0)

Loads automotive BOM datasets from ``Data set/BOM/train_set.csv`` and
``test_set.csv``, constructs component nodes with 8 supply-chain risk
features, and builds a hypergraph with three hyperedge types:

    1. **Assembly hyperedges** -- components belonging to the same product
       category / GL grouping.
    2. **Supplier concentration hyperedges** -- components sourced from a
       small number of overlapping vendors.
    3. **Critical path hyperedges** -- high-cost components on the longest
       lead-time path.

Data path:
    Data set/BOM/train_set.csv
    Data set/BOM/test_set.csv
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


class BOMLoader:
    """
    Loader for the automotive Bill of Materials dataset.

    Reads ``train_set.csv`` and ``test_set.csv``, extracts per-component
    features, and constructs a hypergraph whose hyperedges encode assembly
    relationships, supplier concentration, and critical-path dependencies.

    Node feature vector (8 dimensions):
        0. sole_source_ratio       -- fraction of spend with a single vendor
        1. lead_time_weeks         -- estimated lead time in weeks
        2. safety_stock_days       -- safety-stock buffer in days
        3. substitutability        -- how easily the component can be replaced
        4. geographic_concentration -- concentration in a single geography
        5. demand_volatility       -- coefficient of variation of demand
        6. quality_reject_rate     -- estimated reject rate [0, 1]
        7. disruption_frequency    -- estimated disruption frequency [0, 1]

    Example:
        loader = BOMLoader(data_dir='Data set')
        result = loader.build_hypergraph()
    """

    FEATURE_NAMES = [
        'sole_source_ratio',
        'lead_time_weeks',
        'safety_stock_days',
        'substitutability',
        'geographic_concentration',
        'demand_volatility',
        'quality_reject_rate',
        'disruption_frequency',
    ]

    def __init__(self, data_dir: str = "Data set"):
        """
        Initialize the BOM loader.

        Args:
            data_dir: Root path containing ``BOM/train_set.csv`` and
                      ``BOM/test_set.csv``.
        """
        self.data_dir = Path(data_dir).resolve()
        self.bom_dir = self.data_dir / "BOM"
        self.train_path = self.bom_dir / "train_set.csv"
        self.test_path = self.bom_dir / "test_set.csv"

        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, split: str = 'train') -> pd.DataFrame:
        """
        Load BOM data for the given split.

        Args:
            split: ``'train'``, ``'test'``, or ``'both'``.

        Returns:
            DataFrame (for 'train' or 'test') or the train DataFrame when
            split='both' (test is also stored internally).

        Raises:
            FileNotFoundError: If the expected CSV file does not exist.
        """
        if split in ('train', 'both'):
            if not self.train_path.exists():
                raise FileNotFoundError(
                    f"BOM train set not found at {self.train_path}. "
                    f"Please place train_set.csv inside '{self.bom_dir}'."
                )
            self.train_df = pd.read_csv(self.train_path)
            print(f"  BOM train: {self.train_df.shape[0]:,} rows x "
                  f"{self.train_df.shape[1]} cols")

        if split in ('test', 'both'):
            if not self.test_path.exists():
                raise FileNotFoundError(
                    f"BOM test set not found at {self.test_path}. "
                    f"Please place test_set.csv inside '{self.bom_dir}'."
                )
            self.test_df = pd.read_csv(self.test_path)
            print(f"  BOM test:  {self.test_df.shape[0]:,} rows x "
                  f"{self.test_df.shape[1]} cols")

        self._loaded = True
        return self.train_df if split != 'test' else self.test_df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_component_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """
        Derive the 8-dimensional feature vector for every unique component
        (identified by Vendor_Code).

        Returns:
            Tuple of (features array (N, 8), node_id list, summary DataFrame).
        """
        # Group by vendor to get component-level stats
        vendor_stats = df.groupby('Vendor_Code').agg(
            invoice_count=('Inv_Amt', 'count'),
            total_spend=('Inv_Amt', 'sum'),
            avg_spend=('Inv_Amt', 'mean'),
            std_spend=('Inv_Amt', 'std'),
            min_spend=('Inv_Amt', 'min'),
            max_spend=('Inv_Amt', 'max'),
            n_gl_codes=('GL_Code', 'nunique'),
            n_categories=('Product_Category', 'nunique'),
        ).reset_index()

        vendor_stats['std_spend'] = vendor_stats['std_spend'].fillna(0.0)
        n = len(vendor_stats)

        # Derive the 8 features
        max_spend = vendor_stats['total_spend'].max() if n > 0 else 1.0
        max_invoices = vendor_stats['invoice_count'].max() if n > 0 else 1.0
        max_gl = vendor_stats['n_gl_codes'].max() if n > 0 else 1.0

        # 0. sole_source_ratio: vendors supplying fewer categories are more sole-source
        vendor_stats['sole_source_ratio'] = 1.0 - (
            vendor_stats['n_categories'] / vendor_stats['n_categories'].max()
        ).clip(0, 1)

        # 1. lead_time_weeks: proxy -- inverse of invoice frequency
        vendor_stats['lead_time_weeks'] = (
            (52.0 * (1.0 - vendor_stats['invoice_count'] / max_invoices)) + 1.0
        ).clip(1, 52)

        # 2. safety_stock_days: proportional to spend volatility
        vendor_stats['safety_stock_days'] = (
            30.0 * vendor_stats['std_spend'] / (vendor_stats['avg_spend'] + 1e-8)
        ).clip(0, 90)

        # 3. substitutability: more GL codes -> more substitutable
        vendor_stats['substitutability'] = (
            vendor_stats['n_gl_codes'] / max_gl
        ).clip(0, 1)

        # 4. geographic_concentration: proxy -- inverse of category spread
        vendor_stats['geographic_concentration'] = (
            1.0 - vendor_stats['n_categories'] / vendor_stats['n_categories'].max()
        ).clip(0, 1)

        # 5. demand_volatility: coefficient of variation
        vendor_stats['demand_volatility'] = (
            vendor_stats['std_spend'] / (vendor_stats['avg_spend'] + 1e-8)
        ).clip(0, 5) / 5.0

        # 6. quality_reject_rate: synthetic -- based on spend skew
        vendor_stats['quality_reject_rate'] = np.random.RandomState(42).beta(
            2, 20, size=n
        )

        # 7. disruption_frequency: synthetic -- based on lead-time proxy
        vendor_stats['disruption_frequency'] = (
            vendor_stats['lead_time_weeks'] / 52.0
        ).clip(0, 1) * np.random.RandomState(99).uniform(0.5, 1.0, size=n)

        node_ids = [f"BOM_{i:05d}" for i in range(n)]

        features = vendor_stats[self.FEATURE_NAMES].values.astype(np.float64)

        return features, node_ids, vendor_stats

    # ------------------------------------------------------------------
    # Hypergraph construction
    # ------------------------------------------------------------------

    def build_hypergraph(
        self,
        split: str = 'train',
        supplier_concentration_k: int = 5,
        critical_cost_quantile: float = 0.75,
    ) -> Dict[str, Any]:
        """
        Build a BOM hypergraph with three hyperedge types.

        Hyperedge types:
            * ``assembly`` -- components sharing the same Product_Category.
            * ``supplier_concentration`` -- top-k vendors by spend that
              frequently co-occur across GL codes.
            * ``critical_path`` -- components whose total spend exceeds the
              ``critical_cost_quantile`` threshold.

        Args:
            split: Which split to use ('train', 'test', 'both').
            supplier_concentration_k: Number of top vendors per GL code used
                                      to form supplier-concentration edges.
            critical_cost_quantile: Spend quantile above which a component is
                                    classified as critical-path.

        Returns:
            Dictionary compatible with :class:`DataAdapter`::

                {
                    'node_features':     np.ndarray (N, 8),
                    'incidence_matrix':  np.ndarray (N, M),
                    'timestamps':        None,
                    'node_types':        list[str]   length N,
                    'edge_types':        list[str]   length M,
                    'hyperedge_weights': np.ndarray  length M,
                    '_source':           'bom',
                    'hypergraph':        Hypergraph,
                }
        """
        if not self._loaded:
            self.load(split=split)

        df = self.train_df.copy() if split != 'test' else self.test_df.copy()
        if split == 'both' and self.test_df is not None:
            df = pd.concat([self.train_df, self.test_df], ignore_index=True)

        features, node_ids, vendor_stats = self._build_component_features(df)
        n_nodes = len(node_ids)

        # Map vendor code -> node index
        vendor_to_idx = dict(zip(vendor_stats['Vendor_Code'], range(n_nodes)))

        hypergraph = Hypergraph()
        node_types: List[str] = []

        # Add nodes to hypergraph
        for idx, nid in enumerate(node_ids):
            ntype = 'component'
            node_types.append(ntype)
            node = HypergraphNode(
                node_id=nid,
                node_type=ntype,
                tier=1,
                lead_time=float(features[idx, 1]),
                reliability=1.0 - float(features[idx, 6]),
                substitutability=float(features[idx, 3]),
                cost=float(vendor_stats.iloc[idx]['total_spend']),
            )
            hypergraph.add_node(node)

        incidence_cols: List[np.ndarray] = []
        edge_types: List[str] = []
        hyperedge_weights: List[float] = []
        he_counter = 0

        # --- Type 1: Assembly hyperedges (by Product_Category) ---
        for cat, grp in df.groupby('Product_Category'):
            member_vendors = grp['Vendor_Code'].unique()
            member_indices = [vendor_to_idx[v] for v in member_vendors if v in vendor_to_idx]
            if len(member_indices) < 2:
                continue

            col = np.zeros(n_nodes, dtype=np.float32)
            member_node_ids = set()
            for mi in member_indices:
                col[mi] = 1.0
                member_node_ids.add(node_ids[mi])

            he_id = f"HE_BOM_ASM_{he_counter:04d}"
            he_counter += 1

            weight = float(grp['Inv_Amt'].sum())
            incidence_cols.append(col)
            edge_types.append('assembly')
            hyperedge_weights.append(weight)

            hypergraph.add_hyperedge(HypergraphEdge(
                hyperedge_id=he_id,
                nodes=member_node_ids,
                bom_weight=weight,
                tolerance=0.5,
                critical_path=0.0,
                tier_level=1,
            ))

        # --- Type 2: Supplier concentration hyperedges (per GL_Code) ---
        for gl, grp in df.groupby('GL_Code'):
            top_vendors = (
                grp.groupby('Vendor_Code')['Inv_Amt']
                .sum()
                .nlargest(supplier_concentration_k)
                .index.tolist()
            )
            member_indices = [vendor_to_idx[v] for v in top_vendors if v in vendor_to_idx]
            if len(member_indices) < 2:
                continue

            col = np.zeros(n_nodes, dtype=np.float32)
            member_node_ids = set()
            for mi in member_indices:
                col[mi] = 1.0
                member_node_ids.add(node_ids[mi])

            he_id = f"HE_BOM_SUP_{he_counter:04d}"
            he_counter += 1

            # Weight: higher when fewer vendors dominate
            concentration = len(top_vendors) / max(grp['Vendor_Code'].nunique(), 1)
            incidence_cols.append(col)
            edge_types.append('supplier_concentration')
            hyperedge_weights.append(float(concentration))

            hypergraph.add_hyperedge(HypergraphEdge(
                hyperedge_id=he_id,
                nodes=member_node_ids,
                bom_weight=concentration,
                tolerance=0.3,
                critical_path=0.0,
                tier_level=1,
            ))

        # --- Type 3: Critical path hyperedges ---
        spend_threshold = vendor_stats['total_spend'].quantile(critical_cost_quantile)
        critical_vendors = vendor_stats[
            vendor_stats['total_spend'] >= spend_threshold
        ]['Vendor_Code'].tolist()
        critical_indices = [vendor_to_idx[v] for v in critical_vendors if v in vendor_to_idx]

        if len(critical_indices) >= 2:
            col = np.zeros(n_nodes, dtype=np.float32)
            member_node_ids = set()
            for mi in critical_indices:
                col[mi] = 1.0
                member_node_ids.add(node_ids[mi])

            he_id = f"HE_BOM_CRT_{he_counter:04d}"
            he_counter += 1
            incidence_cols.append(col)
            edge_types.append('critical_path')
            hyperedge_weights.append(2.0)

            hypergraph.add_hyperedge(HypergraphEdge(
                hyperedge_id=he_id,
                nodes=member_node_ids,
                bom_weight=2.0,
                tolerance=0.1,
                critical_path=1.0,
                tier_level=1,
            ))

        # --- Assemble incidence matrix ---
        if incidence_cols:
            incidence_matrix = np.column_stack(incidence_cols)
        else:
            incidence_matrix = np.zeros((n_nodes, 1), dtype=np.float32)
            edge_types = ['empty']
            hyperedge_weights = [0.0]

        # Normalize weights to [0, 1]
        hw = np.array(hyperedge_weights, dtype=np.float32)
        hw_max = hw.max() if hw.max() > 0 else 1.0
        hw = hw / hw_max

        print(f"  Built BOM hypergraph: {n_nodes} nodes, "
              f"{incidence_matrix.shape[1]} hyperedges "
              f"(assembly={edge_types.count('assembly')}, "
              f"supplier_conc={edge_types.count('supplier_concentration')}, "
              f"critical_path={edge_types.count('critical_path')})")

        return {
            'node_features': features,
            'incidence_matrix': incidence_matrix,
            'timestamps': None,
            'node_types': node_types,
            'edge_types': edge_types,
            'hyperedge_weights': hw,
            '_source': 'bom',
            'hypergraph': hypergraph,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return summary statistics about the loaded BOM data.

        Returns:
            Dict with counts, column names, and vendor/category breakdowns.
        """
        if not self._loaded:
            return {'status': 'not_loaded'}

        df = self.train_df
        stats: Dict[str, Any] = {
            'n_train_rows': len(self.train_df) if self.train_df is not None else 0,
            'n_test_rows': len(self.test_df) if self.test_df is not None else 0,
            'columns': list(df.columns) if df is not None else [],
        }
        if df is not None:
            stats['n_vendors'] = df['Vendor_Code'].nunique() if 'Vendor_Code' in df.columns else 0
            stats['n_categories'] = df['Product_Category'].nunique() if 'Product_Category' in df.columns else 0
            stats['n_gl_codes'] = df['GL_Code'].nunique() if 'GL_Code' in df.columns else 0
        return stats


if __name__ == "__main__":
    print("=" * 60)
    print("BOMLoader - Automotive Bill of Materials Loader")
    print("HT-HGNN v2.0 Data Pipeline")
    print("=" * 60)
    print()
    print("Dataset: BOM train_set.csv / test_set.csv")
    print("Path:    Data set/BOM/")
    print()
    print("Node features (8-dim):")
    for i, name in enumerate(BOMLoader.FEATURE_NAMES):
        print(f"  [{i}] {name}")
    print()
    print("Hyperedge types: assembly, supplier_concentration, critical_path")
    print()
    print("Usage:")
    print("  loader = BOMLoader(data_dir='Data set')")
    print("  data   = loader.build_hypergraph(split='train')")
    print()
    print("Module loaded successfully.")
