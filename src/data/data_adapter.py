"""
Unified Dataset Normalization Layer (HT-HGNN v2.0)

DataAdapter normalizes all dataset sources into a common HT-HGNN input format.
Accepts output from any of the 6 dataset loaders (dataco, bom, ports, maintenance,
retail, indigo) and returns a standardized dictionary suitable for model training.

Standardized output format:
    - node_features:     (N x F) tensor of normalized node features
    - incidence_matrix:  (N x M) binary incidence matrix
    - timestamps:        list of temporal markers per hyperedge
    - node_types:        list of node type labels (length N)
    - edge_types:        list of hyperedge type labels (length M)
    - hyperedge_weights: (M,) weight vector for hyperedges
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union


class DataAdapter:
    """
    Unified normalization layer that converts heterogeneous dataset loader
    outputs into a standardized HT-HGNN input format.

    Supports:
        - DataCoLoader (supply chain logistics)
        - BOMLoader (automotive bill of materials)
        - PortDisruptionLoader (global port disruption)
        - MaintenanceLoader (predictive maintenance)
        - RetailLoader (retail M5 sales)
        - IndiGoDisruptionLoader (aviation disruption 2025)
        - Static and dynamic hyperedge construction modes

    Example usage:
        adapter = DataAdapter()
        loader_output = dataco_loader.build_hypergraph()
        standardized = adapter.transform(loader_output, source='dataco')
    """

    # Column name conventions expected from each loader
    REQUIRED_KEYS = [
        'node_features', 'incidence_matrix', 'node_types', 'edge_types'
    ]
    OPTIONAL_KEYS = ['timestamps', 'hyperedge_weights']

    # Default feature dimension per source (used for validation)
    SOURCE_FEATURE_DIMS = {
        'dataco': 8,
        'bom': 8,
        'ports': 6,
        'maintenance': 7,
        'retail': 6,
        'indigo': 10,
    }

    def __init__(self, normalize: bool = True, feature_range: tuple = (0.0, 1.0)):
        """
        Initialize the DataAdapter.

        Args:
            normalize: Whether to min-max normalize node features to feature_range.
            feature_range: Target range for normalization (default [0, 1]).
        """
        self.normalize = normalize
        self.feature_range = feature_range
        self._fitted_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, loader_output: Dict[str, Any],
                  source: str = 'auto',
                  dynamic: bool = False,
                  temporal_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Transform a loader output dict into the standardized HT-HGNN format.

        Args:
            loader_output: Raw dictionary produced by one of the 5 loaders.
                Must contain at minimum:
                    'node_features' (np.ndarray or pd.DataFrame),
                    'incidence_matrix' (np.ndarray or sparse),
                    'node_types' (list[str]),
                    'edge_types' (list[str]).
                Optional:
                    'timestamps' (list), 'hyperedge_weights' (np.ndarray).
            source: One of 'dataco', 'bom', 'ports', 'maintenance', 'retail',
                    or 'auto' to infer from loader_output metadata.
            dynamic: If True, enables dynamic hyperedge construction where
                     hyperedges are re-built per temporal window.
            temporal_window: Number of time steps per window when dynamic=True.

        Returns:
            Standardized dict with keys:
                node_features     - np.ndarray shape (N, F)
                incidence_matrix  - np.ndarray shape (N, M)
                timestamps        - list of length M (or None)
                node_types        - list of length N
                edge_types        - list of length M
                hyperedge_weights - np.ndarray of length M
        """
        source = self._resolve_source(loader_output, source)
        self._validate_input(loader_output)

        # --- Extract and convert node features ---
        node_features = self._to_numpy(loader_output['node_features'])

        # --- Normalize features ---
        if self.normalize:
            node_features = self._normalize_features(node_features, source)

        # --- Build incidence matrix ---
        incidence_matrix = self._to_numpy(loader_output['incidence_matrix'])

        # Ensure incidence is (N, M): rows = nodes, cols = hyperedges
        n_nodes = node_features.shape[0]
        if incidence_matrix.shape[0] != n_nodes and incidence_matrix.shape[1] == n_nodes:
            incidence_matrix = incidence_matrix.T

        # --- Node and edge types ---
        node_types = list(loader_output['node_types'])
        edge_types = list(loader_output['edge_types'])

        # --- Timestamps ---
        timestamps = loader_output.get('timestamps', None)
        if timestamps is not None:
            timestamps = list(timestamps)

        # --- Hyperedge weights ---
        hyperedge_weights = loader_output.get('hyperedge_weights', None)
        if hyperedge_weights is None:
            n_edges = incidence_matrix.shape[1]
            hyperedge_weights = np.ones(n_edges, dtype=np.float32)
        else:
            hyperedge_weights = np.asarray(hyperedge_weights, dtype=np.float32)

        # --- Dynamic hyperedge construction ---
        if dynamic and timestamps is not None and temporal_window is not None:
            incidence_matrix, edge_types, timestamps, hyperedge_weights = (
                self._build_dynamic_hyperedges(
                    incidence_matrix, edge_types, timestamps,
                    hyperedge_weights, temporal_window
                )
            )

        result = {
            'node_features': node_features.astype(np.float32),
            'incidence_matrix': incidence_matrix.astype(np.float32),
            'timestamps': timestamps,
            'node_types': node_types,
            'edge_types': edge_types,
            'hyperedge_weights': hyperedge_weights,
        }

        self._validate_output(result)
        return result

    def fit_transform(self, loader_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Fit normalization parameters from training data and transform.

        Args:
            loader_output: Raw loader output (used to fit normalization).
            **kwargs: Forwarded to transform().

        Returns:
            Standardized dict (same as transform).
        """
        # Reset fitted params so they are re-computed from this data
        self._fitted_params = {}
        return self.transform(loader_output, **kwargs)

    def get_metadata(self, standardized: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return summary metadata for a standardized dataset.

        Args:
            standardized: Output of transform().

        Returns:
            Dict with n_nodes, n_features, n_hyperedges, etc.
        """
        nf = standardized['node_features']
        im = standardized['incidence_matrix']
        return {
            'n_nodes': nf.shape[0],
            'n_features': nf.shape[1],
            'n_hyperedges': im.shape[1],
            'node_type_counts': pd.Series(standardized['node_types']).value_counts().to_dict(),
            'edge_type_counts': pd.Series(standardized['edge_types']).value_counts().to_dict(),
            'has_timestamps': standardized['timestamps'] is not None,
            'feature_range': (float(nf.min()), float(nf.max())),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(data) -> np.ndarray:
        """Convert pandas DataFrame / Series or list to numpy array."""
        if isinstance(data, pd.DataFrame):
            return data.values.astype(np.float64)
        if isinstance(data, pd.Series):
            return data.values.astype(np.float64)
        if isinstance(data, np.ndarray):
            return data.astype(np.float64)
        return np.asarray(data, dtype=np.float64)

    def _normalize_features(self, features: np.ndarray, source: str) -> np.ndarray:
        """
        Min-max normalize features to self.feature_range.

        Stores fitted min/max per source so the same transform can be applied
        to test data later.
        """
        lo, hi = self.feature_range

        if source not in self._fitted_params:
            col_min = features.min(axis=0)
            col_max = features.max(axis=0)
            # Avoid division by zero for constant columns
            col_range = col_max - col_min
            col_range[col_range == 0] = 1.0
            self._fitted_params[source] = {'min': col_min, 'range': col_range}

        params = self._fitted_params[source]
        normalized = (features - params['min']) / params['range']
        normalized = normalized * (hi - lo) + lo
        return np.clip(normalized, lo, hi)

    @staticmethod
    def _resolve_source(loader_output: Dict, source: str) -> str:
        """Resolve the dataset source label."""
        if source != 'auto':
            return source
        # Try to infer from metadata key injected by loaders
        if '_source' in loader_output:
            return loader_output['_source']
        return 'unknown'

    def _validate_input(self, loader_output: Dict) -> None:
        """Validate that required keys are present in loader output."""
        missing = [k for k in self.REQUIRED_KEYS if k not in loader_output]
        if missing:
            raise ValueError(
                f"Loader output is missing required keys: {missing}. "
                f"Expected keys: {self.REQUIRED_KEYS}"
            )

    @staticmethod
    def _validate_output(result: Dict) -> None:
        """Run basic consistency checks on the standardized output."""
        nf = result['node_features']
        im = result['incidence_matrix']
        n_nodes = nf.shape[0]
        n_edges = im.shape[1]

        if im.shape[0] != n_nodes:
            raise ValueError(
                f"Incidence matrix rows ({im.shape[0]}) must equal number of "
                f"nodes ({n_nodes})."
            )
        if len(result['node_types']) != n_nodes:
            raise ValueError(
                f"node_types length ({len(result['node_types'])}) must match "
                f"number of nodes ({n_nodes})."
            )
        if len(result['edge_types']) != n_edges:
            raise ValueError(
                f"edge_types length ({len(result['edge_types'])}) must match "
                f"number of hyperedges ({n_edges})."
            )
        if result['hyperedge_weights'].shape[0] != n_edges:
            raise ValueError(
                f"hyperedge_weights length ({result['hyperedge_weights'].shape[0]}) "
                f"must match number of hyperedges ({n_edges})."
            )

    @staticmethod
    def _build_dynamic_hyperedges(
        incidence: np.ndarray,
        edge_types: List[str],
        timestamps: List,
        weights: np.ndarray,
        window: int
    ):
        """
        Re-partition hyperedges into temporal windows for dynamic construction.

        Static hyperedges that span multiple time windows are duplicated into
        each window they overlap with, producing a larger set of
        window-specific hyperedges.

        Args:
            incidence:  (N, M) incidence matrix.
            edge_types: length-M list of edge type labels.
            timestamps: length-M list of numeric timestamps.
            weights:    length-M weight array.
            window:     size of each temporal window.

        Returns:
            Tuple of (new_incidence, new_edge_types, new_timestamps, new_weights).
        """
        ts_array = np.asarray(timestamps, dtype=np.float64)
        if len(ts_array) == 0:
            return incidence, edge_types, timestamps, weights

        t_min = ts_array.min()
        t_max = ts_array.max()

        new_inc_cols = []
        new_edge_types = []
        new_timestamps = []
        new_weights = []

        current_start = t_min
        while current_start <= t_max:
            window_end = current_start + window
            mask = (ts_array >= current_start) & (ts_array < window_end)
            indices = np.where(mask)[0]

            for idx in indices:
                new_inc_cols.append(incidence[:, idx])
                new_edge_types.append(edge_types[idx])
                new_timestamps.append(timestamps[idx])
                new_weights.append(weights[idx])

            current_start = window_end

        if len(new_inc_cols) == 0:
            return incidence, edge_types, timestamps, weights

        new_incidence = np.column_stack(new_inc_cols)
        new_weights = np.array(new_weights, dtype=np.float32)

        return new_incidence, new_edge_types, new_timestamps, new_weights


if __name__ == "__main__":
    print("=" * 60)
    print("DataAdapter - Unified Dataset Normalization Layer")
    print("HT-HGNN v2.0 Data Pipeline")
    print("=" * 60)
    print()
    print("Supported sources: dataco, bom, ports, maintenance, retail")
    print("Output keys:       node_features, incidence_matrix, timestamps,")
    print("                   node_types, edge_types, hyperedge_weights")
    print()
    print("Usage:")
    print("  adapter = DataAdapter(normalize=True)")
    print("  result  = adapter.transform(loader.build_hypergraph(), source='dataco')")
    print()
    print("Module loaded successfully.")
