"""
Retail M5 (Walmart Sales) Loader (HT-HGNN v2.0)

Loads retail sales data from ``Data set/Retail/`` (M5 Walmart Forecasting
Competition format) and constructs a hypergraph modelling product-store
relationships through demand patterns.

Expected files (graceful fallback to synthetic sample when absent):
    - sales_train_evaluation.csv   -- daily unit sales per product-store
    - sell_prices.csv              -- weekly sell prices
    - calendar.csv                 -- date metadata and SNAP indicators

Hyperedge types:
    1. **Category co-purchase** -- products frequently purchased together
       within a department or category on the same days.
    2. **Promotion wave** -- products affected by the same SNAP/promotion
       event window.
    3. **Stockout cascade** -- products whose zero-sale streaks overlap in
       time, indicating correlated stockout patterns.
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


class RetailLoader:
    """
    Loader for the M5 Walmart Retail Sales dataset.

    If real M5 data files exist under ``Data set/Retail/`` they are loaded.
    Otherwise the loader generates a small synthetic sample so that
    downstream code can be developed and tested without the full dataset.

    Node features (6 dimensions):
        0. avg_daily_sales       -- mean daily unit sales
        1. sales_volatility      -- coefficient of variation of daily sales
        2. avg_price             -- mean weekly sell price
        3. price_elasticity      -- estimated price sensitivity
        4. promo_sensitivity     -- responsiveness to SNAP events
        5. stockout_frequency    -- fraction of days with zero sales

    Example:
        loader = RetailLoader(data_dir='Data set')
        data = loader.build_hypergraph()
    """

    FEATURE_NAMES = [
        'avg_daily_sales',
        'sales_volatility',
        'avg_price',
        'price_elasticity',
        'promo_sensitivity',
        'stockout_frequency',
    ]

    # M5 dataset structure constants
    _M5_CATEGORIES = ['HOBBIES', 'HOUSEHOLD', 'FOODS']
    _M5_DEPARTMENTS = [
        'HOBBIES_1', 'HOBBIES_2',
        'HOUSEHOLD_1', 'HOUSEHOLD_2',
        'FOODS_1', 'FOODS_2', 'FOODS_3',
    ]
    _M5_STORES = [
        'CA_1', 'CA_2', 'CA_3', 'CA_4',
        'TX_1', 'TX_2', 'TX_3',
        'WI_1', 'WI_2', 'WI_3',
    ]

    def __init__(self, data_dir: str = "Data set"):
        """
        Initialize the Retail M5 loader.

        Args:
            data_dir: Root path containing ``Retail/`` subfolder.
        """
        self.data_dir = Path(data_dir).resolve()
        self.retail_dir = self.data_dir / "Retail"

        self.sales_df: Optional[pd.DataFrame] = None
        self.prices_df: Optional[pd.DataFrame] = None
        self.calendar_df: Optional[pd.DataFrame] = None
        self._is_synthetic = False
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, max_items: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Load M5 data files or generate synthetic fallback.

        Args:
            max_items: Maximum number of product-store items to include
                       (limits memory for the full dataset).

        Returns:
            Dict with keys ``'sales'``, ``'prices'``, ``'calendar'``.
        """
        sales_path = self.retail_dir / "sales_train_evaluation.csv"
        prices_path = self.retail_dir / "sell_prices.csv"
        calendar_path = self.retail_dir / "calendar.csv"

        if sales_path.exists():
            print(f"Loading Retail M5 data from {self.retail_dir} ...")
            self.sales_df = pd.read_csv(sales_path, nrows=max_items)
            if prices_path.exists():
                self.prices_df = pd.read_csv(prices_path)
            else:
                print("  WARNING: sell_prices.csv not found; "
                      "generating synthetic prices.")
                self.prices_df = self._generate_synthetic_prices()
            if calendar_path.exists():
                self.calendar_df = pd.read_csv(calendar_path)
            else:
                print("  WARNING: calendar.csv not found; "
                      "generating synthetic calendar.")
                self.calendar_df = self._generate_synthetic_calendar()
            self._is_synthetic = False
        else:
            print(f"  Retail M5 data not found at {self.retail_dir}.")
            print("  Generating synthetic retail dataset for development ...")
            self.sales_df, self.prices_df, self.calendar_df = (
                self._generate_synthetic_data(n_items=max_items or 200)
            )
            self._is_synthetic = True

        self._loaded = True
        print(f"  Sales rows: {len(self.sales_df):,} | "
              f"Prices rows: {len(self.prices_df):,} | "
              f"Calendar rows: {len(self.calendar_df):,}")

        return {
            'sales': self.sales_df,
            'prices': self.prices_df,
            'calendar': self.calendar_df,
        }

    # ------------------------------------------------------------------
    # Synthetic data generation (fallback)
    # ------------------------------------------------------------------

    def _generate_synthetic_data(
        self, n_items: int = 200, n_days: int = 365
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate a small synthetic M5-like dataset."""
        rng = np.random.RandomState(42)

        # --- Sales ---
        day_cols = [f"d_{i+1}" for i in range(n_days)]
        records = []
        for i in range(n_items):
            dept = rng.choice(self._M5_DEPARTMENTS)
            cat = dept.rsplit('_', 1)[0]
            store = rng.choice(self._M5_STORES)
            state = store.split('_')[0]
            item_id = f"ITEM_{i:04d}"
            row = {
                'id': f"{item_id}_{store}_evaluation",
                'item_id': item_id,
                'dept_id': dept,
                'cat_id': cat,
                'store_id': store,
                'state_id': state,
            }
            # Simulate daily sales with Poisson + zero-inflation
            base_rate = rng.uniform(0.5, 10.0)
            sales = rng.poisson(base_rate, n_days)
            # Zero-inflation
            zero_mask = rng.random(n_days) < 0.15
            sales[zero_mask] = 0
            for d, val in zip(day_cols, sales):
                row[d] = int(val)
            records.append(row)

        sales_df = pd.DataFrame(records)

        # --- Prices ---
        prices_df = self._generate_synthetic_prices(
            items=sales_df['item_id'].unique().tolist(),
            stores=sales_df['store_id'].unique().tolist(),
            rng=rng,
        )

        # --- Calendar ---
        calendar_df = self._generate_synthetic_calendar(n_days=n_days, rng=rng)

        return sales_df, prices_df, calendar_df

    @staticmethod
    def _generate_synthetic_prices(
        items: Optional[List[str]] = None,
        stores: Optional[List[str]] = None,
        n_weeks: int = 52,
        rng: Optional[np.random.RandomState] = None,
    ) -> pd.DataFrame:
        """Generate synthetic weekly prices."""
        if rng is None:
            rng = np.random.RandomState(99)
        if items is None:
            items = [f"ITEM_{i:04d}" for i in range(50)]
        if stores is None:
            stores = RetailLoader._M5_STORES

        records = []
        for item in items:
            base_price = round(rng.uniform(1.0, 25.0), 2)
            for store in stores:
                for wm_yr_wk in range(11101, 11101 + n_weeks):
                    price = round(base_price * rng.uniform(0.85, 1.15), 2)
                    records.append({
                        'store_id': store,
                        'item_id': item,
                        'wm_yr_wk': wm_yr_wk,
                        'sell_price': price,
                    })
        return pd.DataFrame(records)

    @staticmethod
    def _generate_synthetic_calendar(
        n_days: int = 365,
        rng: Optional[np.random.RandomState] = None,
    ) -> pd.DataFrame:
        """Generate synthetic calendar with SNAP indicators."""
        if rng is None:
            rng = np.random.RandomState(77)

        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        records = []
        for i, dt in enumerate(dates):
            records.append({
                'd': f"d_{i+1}",
                'date': dt.strftime('%Y-%m-%d'),
                'wm_yr_wk': 11101 + i // 7,
                'weekday': dt.strftime('%A'),
                'month': dt.month,
                'year': dt.year,
                'snap_CA': int(rng.random() < 0.15),
                'snap_TX': int(rng.random() < 0.12),
                'snap_WI': int(rng.random() < 0.10),
                'event_name': '',
                'event_type': '',
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_item_features(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Compute the 6-dimensional feature vector per product-store item.

        Returns:
            (features (N, 6), node_ids, node_types).
        """
        df = self.sales_df.copy()

        # Identify day columns
        day_cols = [c for c in df.columns if c.startswith('d_')]
        if not day_cols:
            raise ValueError("Sales DataFrame has no day columns (d_1, d_2, ...).")

        sales_matrix = df[day_cols].values.astype(np.float64)
        n_items = len(df)

        # 0. avg_daily_sales
        avg_sales = sales_matrix.mean(axis=1)

        # 1. sales_volatility (CV)
        std_sales = sales_matrix.std(axis=1)
        volatility = std_sales / (avg_sales + 1e-8)

        # 2. avg_price
        avg_price = np.zeros(n_items)
        if self.prices_df is not None and 'item_id' in df.columns:
            price_means = self.prices_df.groupby('item_id')['sell_price'].mean()
            for i, item_id in enumerate(df['item_id']):
                if item_id in price_means.index:
                    avg_price[i] = price_means[item_id]

        # 3. price_elasticity (proxy: correlation between price change and sales)
        # Simplified: random proxy scaled by volatility
        rng = np.random.RandomState(55)
        price_elasticity = volatility * rng.uniform(0.2, 1.0, size=n_items)

        # 4. promo_sensitivity (proxy: fraction of sales on SNAP days)
        promo_sensitivity = np.zeros(n_items)
        if self.calendar_df is not None:
            snap_cols = [c for c in self.calendar_df.columns if c.startswith('snap_')]
            if snap_cols and len(self.calendar_df) >= len(day_cols):
                snap_vector = self.calendar_df[snap_cols].max(axis=1).values[:len(day_cols)]
                snap_days = snap_vector.astype(bool)
                for i in range(n_items):
                    total = sales_matrix[i].sum()
                    if total > 0 and len(snap_days) == sales_matrix.shape[1]:
                        snap_sales = sales_matrix[i][snap_days].sum()
                        promo_sensitivity[i] = snap_sales / total

        # 5. stockout_frequency
        stockout_freq = (sales_matrix == 0).mean(axis=1)

        features = np.column_stack([
            avg_sales, volatility, avg_price,
            price_elasticity, promo_sensitivity, stockout_freq,
        ])

        # Node IDs
        if 'id' in df.columns:
            node_ids = df['id'].tolist()
        else:
            node_ids = [f"RTL_{i:05d}" for i in range(n_items)]

        # Node types (department)
        if 'dept_id' in df.columns:
            node_types = df['dept_id'].astype(str).tolist()
        elif 'cat_id' in df.columns:
            node_types = df['cat_id'].astype(str).tolist()
        else:
            node_types = ['retail_item'] * n_items

        return features, node_ids, node_types

    # ------------------------------------------------------------------
    # Hypergraph construction
    # ------------------------------------------------------------------

    def build_hypergraph(
        self,
        co_purchase_min: int = 3,
        stockout_streak_days: int = 3,
        min_hyperedge_size: int = 3,
        max_items: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build a retail hypergraph with three hyperedge types.

        Hyperedge types:
            * ``category_co_purchase`` -- items in the same department/category
              within the same store.
            * ``promotion_wave`` -- items simultaneously affected by a
              SNAP/promotion event.
            * ``stockout_cascade`` -- items whose zero-sale streaks overlap,
              indicating correlated stockout patterns.

        Args:
            co_purchase_min: Minimum items to form a co-purchase hyperedge.
            stockout_streak_days: Consecutive zero-sale days defining a streak.
            min_hyperedge_size: Minimum items per hyperedge (general).
            max_items: Limit items loaded for memory management.

        Returns:
            Dictionary compatible with :class:`DataAdapter`.
        """
        if not self._loaded:
            self.load(max_items=max_items)

        features, node_ids, node_types = self._build_item_features()
        n_nodes = len(node_ids)
        node_idx_map = {nid: i for i, nid in enumerate(node_ids)}

        df = self.sales_df.copy()
        day_cols = [c for c in df.columns if c.startswith('d_')]
        sales_matrix = df[day_cols].values.astype(np.float64)

        hypergraph = Hypergraph()
        for idx, nid in enumerate(node_ids):
            hypergraph.add_node(HypergraphNode(
                node_id=nid,
                node_type=node_types[idx],
                tier=1,
                lead_time=float(features[idx, 5]) * 30,  # stockout as lead-time proxy
                reliability=1.0 - float(features[idx, 5]),
                substitutability=float(features[idx, 1]),  # volatile = less substitutable
                cost=float(features[idx, 2]),
            ))

        incidence_cols: List[np.ndarray] = []
        edge_types: List[str] = []
        hyperedge_weights: List[float] = []
        he_counter = 0

        # --- Type 1: Category co-purchase hyperedges ---
        if 'dept_id' in df.columns and 'store_id' in df.columns:
            for (dept, store), grp in df.groupby(['dept_id', 'store_id']):
                member_indices = grp.index.tolist()
                if len(member_indices) < max(co_purchase_min, min_hyperedge_size):
                    continue

                col = np.zeros(n_nodes, dtype=np.float32)
                member_node_ids = set()
                for mi in member_indices:
                    col[mi] = 1.0
                    member_node_ids.add(node_ids[mi])

                avg_sales = float(features[member_indices, 0].mean())

                he_id = f"HE_RTL_COP_{he_counter:04d}"
                he_counter += 1
                incidence_cols.append(col)
                edge_types.append('category_co_purchase')
                hyperedge_weights.append(avg_sales)

                hypergraph.add_hyperedge(HypergraphEdge(
                    hyperedge_id=he_id,
                    nodes=member_node_ids,
                    bom_weight=avg_sales,
                    tolerance=0.5,
                    critical_path=0.0,
                    tier_level=1,
                ))

        # --- Type 2: Promotion wave hyperedges ---
        if self.calendar_df is not None:
            snap_cols = [c for c in self.calendar_df.columns if c.startswith('snap_')]
            if snap_cols and len(self.calendar_df) >= len(day_cols):
                # Identify SNAP event windows
                snap_any = self.calendar_df[snap_cols].max(axis=1).values[:len(day_cols)]
                snap_day_indices = np.where(snap_any == 1)[0]

                if len(snap_day_indices) > 0:
                    # Group consecutive SNAP days into waves
                    waves = []
                    current_wave = [snap_day_indices[0]]
                    for si in snap_day_indices[1:]:
                        if si - current_wave[-1] <= 2:  # within 2-day gap
                            current_wave.append(si)
                        else:
                            waves.append(current_wave)
                            current_wave = [si]
                    waves.append(current_wave)

                    for wave_days in waves:
                        # Items that had above-average sales during this wave
                        wave_sales = sales_matrix[:, wave_days].sum(axis=1)
                        avg_wave = wave_sales.mean()
                        boosted = np.where(wave_sales > avg_wave * 1.2)[0]

                        if len(boosted) < min_hyperedge_size:
                            continue

                        col = np.zeros(n_nodes, dtype=np.float32)
                        member_node_ids = set()
                        for mi in boosted:
                            col[mi] = 1.0
                            member_node_ids.add(node_ids[mi])

                        he_id = f"HE_RTL_PRM_{he_counter:04d}"
                        he_counter += 1
                        incidence_cols.append(col)
                        edge_types.append('promotion_wave')
                        hyperedge_weights.append(float(len(boosted)) / n_nodes)

                        hypergraph.add_hyperedge(HypergraphEdge(
                            hyperedge_id=he_id,
                            nodes=member_node_ids,
                            bom_weight=float(len(boosted)),
                            tolerance=0.6,
                            critical_path=0.0,
                            tier_level=1,
                        ))

        # --- Type 3: Stockout cascade hyperedges ---
        # Find items with overlapping zero-sale streaks
        zero_matrix = (sales_matrix == 0).astype(int)
        n_days = zero_matrix.shape[1]

        # Scan for stockout streaks using a sliding window
        streak_windows = list(range(0, max(n_days - stockout_streak_days, 1),
                                    stockout_streak_days * 2))

        for window_start in streak_windows:
            window_end = min(window_start + stockout_streak_days, n_days)
            window_zeros = zero_matrix[:, window_start:window_end].sum(axis=1)
            # Items with all-zero in this window
            stockout_items = np.where(window_zeros == (window_end - window_start))[0]

            if len(stockout_items) < min_hyperedge_size:
                continue

            col = np.zeros(n_nodes, dtype=np.float32)
            member_node_ids = set()
            for mi in stockout_items:
                col[mi] = 1.0
                member_node_ids.add(node_ids[mi])

            he_id = f"HE_RTL_STO_{he_counter:04d}"
            he_counter += 1
            incidence_cols.append(col)
            edge_types.append('stockout_cascade')
            cascade_severity = float(len(stockout_items)) / n_nodes
            hyperedge_weights.append(cascade_severity)

            hypergraph.add_hyperedge(HypergraphEdge(
                hyperedge_id=he_id,
                nodes=member_node_ids,
                bom_weight=cascade_severity,
                tolerance=0.2,
                critical_path=1.0 if cascade_severity > 0.1 else 0.0,
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

        synth_tag = " (synthetic)" if self._is_synthetic else ""
        print(f"  Built retail hypergraph{synth_tag}: {n_nodes} nodes, "
              f"{incidence_matrix.shape[1]} hyperedges "
              f"(co_purchase={edge_types.count('category_co_purchase')}, "
              f"promotion={edge_types.count('promotion_wave')}, "
              f"stockout={edge_types.count('stockout_cascade')})")

        return {
            'node_features': features,
            'incidence_matrix': incidence_matrix,
            'timestamps': None,
            'node_types': node_types,
            'edge_types': edge_types,
            'hyperedge_weights': hw,
            '_source': 'retail',
            'hypergraph': hypergraph,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return summary statistics about the loaded retail data.

        Returns:
            Dict with item counts, category breakdown, and date range.
        """
        if not self._loaded:
            return {'status': 'not_loaded'}

        stats: Dict[str, Any] = {
            'is_synthetic': self._is_synthetic,
            'n_items': len(self.sales_df) if self.sales_df is not None else 0,
            'n_price_records': len(self.prices_df) if self.prices_df is not None else 0,
            'n_calendar_days': len(self.calendar_df) if self.calendar_df is not None else 0,
        }
        if self.sales_df is not None:
            day_cols = [c for c in self.sales_df.columns if c.startswith('d_')]
            stats['n_days'] = len(day_cols)
            if 'cat_id' in self.sales_df.columns:
                stats['categories'] = self.sales_df['cat_id'].value_counts().to_dict()
            if 'store_id' in self.sales_df.columns:
                stats['stores'] = self.sales_df['store_id'].value_counts().to_dict()
        return stats


if __name__ == "__main__":
    print("=" * 60)
    print("RetailLoader - Retail M5 (Walmart Sales) Loader")
    print("HT-HGNN v2.0 Data Pipeline")
    print("=" * 60)
    print()
    print("Dataset: M5 Walmart Sales Forecasting Competition")
    print("Path:    Data set/Retail/")
    print("Note:    Falls back to synthetic data if files are absent.")
    print()
    print("Node features (6-dim):")
    for i, name in enumerate(RetailLoader.FEATURE_NAMES):
        print(f"  [{i}] {name}")
    print()
    print("Hyperedge types: category_co_purchase, promotion_wave, stockout_cascade")
    print()
    print("Usage:")
    print("  loader = RetailLoader(data_dir='Data set')")
    print("  data   = loader.build_hypergraph()")
    print()
    print("Module loaded successfully.")
