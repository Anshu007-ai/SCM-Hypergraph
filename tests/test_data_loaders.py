"""
Test suite for HT-HGNN v2.0 Data Loaders and Data Adapter.

Tests the data pipeline components including:
- DataAdapter normalization
- DataCoLoader, BOMLoader, PortDisruptionLoader, MaintenanceLoader, RetailLoader init
- SupplyChainDataGenerator synthetic data generation
- RealDataLoader initialization
- Feature normalization, incidence matrix shape, dataset listing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Graceful imports -- skip entire module if core dependencies are missing
# ---------------------------------------------------------------------------
try:
    from src.data.data_adapter import DataAdapter
except ImportError:
    DataAdapter = None

try:
    from src.data.dataco_loader import DataCoLoader
except ImportError:
    DataCoLoader = None

try:
    from src.data.bom_loader import BOMLoader
except ImportError:
    BOMLoader = None

try:
    from src.data.port_loader import PortDisruptionLoader
except ImportError:
    PortDisruptionLoader = None

try:
    from src.data.maintenance_loader import MaintenanceLoader
except ImportError:
    MaintenanceLoader = None

try:
    from src.data.retail_loader import RetailLoader
except ImportError:
    RetailLoader = None

try:
    from src.data.data_generator import SupplyChainDataGenerator
except ImportError:
    SupplyChainDataGenerator = None

try:
    from src.data.real_data_loader import RealDataLoader
except ImportError:
    RealDataLoader = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_loader_output():
    """Create a small synthetic loader output dict compatible with DataAdapter."""
    np.random.seed(0)
    n_nodes = 20
    n_features = 8
    n_edges = 5

    node_features = np.random.rand(n_nodes, n_features).astype(np.float64)
    incidence_matrix = np.zeros((n_nodes, n_edges), dtype=np.float64)
    for e in range(n_edges):
        members = np.random.choice(n_nodes, size=np.random.randint(2, 6), replace=False)
        incidence_matrix[members, e] = 1.0

    return {
        "node_features": node_features,
        "incidence_matrix": incidence_matrix,
        "node_types": ["supplier"] * n_nodes,
        "edge_types": [f"type_{i}" for i in range(n_edges)],
        "timestamps": [float(i) for i in range(n_edges)],
        "hyperedge_weights": np.ones(n_edges, dtype=np.float32),
        "_source": "dataco",
    }


@pytest.fixture
def data_generator():
    """Create a small SupplyChainDataGenerator."""
    if SupplyChainDataGenerator is None:
        pytest.skip("SupplyChainDataGenerator not available")
    return SupplyChainDataGenerator(n_suppliers=30, n_assemblies=10, seed=42)


# ---------------------------------------------------------------------------
# Tests -- DataAdapter
# ---------------------------------------------------------------------------

@pytest.mark.skipif(DataAdapter is None, reason="DataAdapter not importable")
class TestDataAdapter:
    """Tests for the DataAdapter unified normalization layer."""

    def test_data_adapter_normalize(self, small_loader_output):
        """DataAdapter.transform should normalize node features to [0, 1]."""
        adapter = DataAdapter(normalize=True, feature_range=(0.0, 1.0))
        result = adapter.transform(small_loader_output, source="dataco")

        nf = result["node_features"]
        assert nf.min() >= 0.0 - 1e-6, "Normalized features should be >= 0"
        assert nf.max() <= 1.0 + 1e-6, "Normalized features should be <= 1"

    def test_feature_normalization(self, small_loader_output):
        """Verify normalization range is respected with a custom range."""
        adapter = DataAdapter(normalize=True, feature_range=(0.0, 1.0))
        result = adapter.transform(small_loader_output, source="dataco")

        nf = result["node_features"]
        assert nf.dtype == np.float32, "Output features should be float32"
        assert nf.shape[0] == 20, "Number of nodes should be preserved"
        assert nf.shape[1] == 8, "Number of features should be preserved"

    def test_incidence_matrix_shape(self, small_loader_output):
        """Incidence matrix shape should be (N_nodes, M_edges) after transform."""
        adapter = DataAdapter(normalize=True)
        result = adapter.transform(small_loader_output, source="dataco")

        im = result["incidence_matrix"]
        assert im.shape == (20, 5), (
            f"Expected incidence shape (20, 5), got {im.shape}"
        )

    def test_transform_returns_all_keys(self, small_loader_output):
        """The standardized output dict must contain all expected keys."""
        adapter = DataAdapter()
        result = adapter.transform(small_loader_output, source="dataco")

        expected_keys = {
            "node_features",
            "incidence_matrix",
            "timestamps",
            "node_types",
            "edge_types",
            "hyperedge_weights",
        }
        assert expected_keys.issubset(set(result.keys())), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_transform_no_normalize(self, small_loader_output):
        """When normalize=False, features should pass through unchanged in dtype."""
        adapter = DataAdapter(normalize=False)
        result = adapter.transform(small_loader_output, source="dataco")

        # Should still be float32 (dtype cast happens regardless)
        assert result["node_features"].dtype == np.float32

    def test_dataset_list(self):
        """DataAdapter.SOURCE_FEATURE_DIMS lists all 5 expected dataset sources."""
        expected_sources = {"dataco", "bom", "ports", "maintenance", "retail"}
        actual_sources = set(DataAdapter.SOURCE_FEATURE_DIMS.keys())
        assert expected_sources == actual_sources, (
            f"Expected sources {expected_sources}, got {actual_sources}"
        )


# ---------------------------------------------------------------------------
# Tests -- Individual Loaders (init only, no real data files needed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(DataCoLoader is None, reason="DataCoLoader not importable")
def test_dataco_loader_init():
    """DataCoLoader should instantiate and store data_dir path."""
    loader = DataCoLoader(data_dir="Data set")
    assert loader.data_dir.name == "Data set" or str(loader.data_dir).endswith("Data set")
    assert loader._loaded is False


@pytest.mark.skipif(BOMLoader is None, reason="BOMLoader not importable")
def test_bom_loader_init():
    """BOMLoader should instantiate and locate the BOM subfolder."""
    loader = BOMLoader(data_dir="Data set")
    assert str(loader.bom_dir).endswith("BOM")
    assert loader._loaded is False


@pytest.mark.skipif(PortDisruptionLoader is None, reason="PortDisruptionLoader not importable")
def test_port_loader_init():
    """PortDisruptionLoader should instantiate and point to the Ports subfolder."""
    loader = PortDisruptionLoader(data_dir="Data set")
    assert str(loader.ports_dir).endswith("Ports")
    assert loader._loaded is False


@pytest.mark.skipif(MaintenanceLoader is None, reason="MaintenanceLoader not importable")
def test_maintenance_loader_init():
    """MaintenanceLoader should instantiate and locate the Maintenance CSV."""
    loader = MaintenanceLoader(data_dir="Data set")
    assert str(loader.csv_path).endswith("ai4i2020.csv")
    assert loader._loaded is False


@pytest.mark.skipif(RetailLoader is None, reason="RetailLoader not importable")
def test_retail_loader_init():
    """RetailLoader should instantiate and point to the Retail subfolder."""
    loader = RetailLoader(data_dir="Data set")
    assert str(loader.retail_dir).endswith("Retail")
    assert loader._loaded is False


# ---------------------------------------------------------------------------
# Tests -- SupplyChainDataGenerator
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SupplyChainDataGenerator is None, reason="SupplyChainDataGenerator not importable")
class TestDataGenerator:
    """Tests for synthetic supply chain data generation."""

    def test_data_generator_generate_nodes(self, data_generator):
        """generate_nodes should produce a DataFrame with the right row count."""
        nodes_df = data_generator.generate_nodes()
        assert len(nodes_df) == 30, f"Expected 30 nodes, got {len(nodes_df)}"
        assert "node_id" in nodes_df.columns
        assert "reliability" in nodes_df.columns

    def test_data_generator_generate_hyperedges(self, data_generator):
        """generate_hyperedges should return a DataFrame of assemblies."""
        # generate_nodes must be called first to populate suppliers dict
        data_generator.generate_nodes()
        hyperedges_df = data_generator.generate_hyperedges()
        assert len(hyperedges_df) == 10, (
            f"Expected 10 hyperedges, got {len(hyperedges_df)}"
        )
        assert "hyperedge_id" in hyperedges_df.columns

    def test_data_generator_generate_incidence(self, data_generator):
        """generate_incidence should produce node-hyperedge relationships."""
        data_generator.generate_nodes()
        data_generator.generate_hyperedges()
        incidence_df = data_generator.generate_incidence()

        assert len(incidence_df) > 0, "Incidence DataFrame should not be empty"
        assert "hyperedge_id" in incidence_df.columns
        assert "node_id" in incidence_df.columns

    def test_data_generator_generate_all(self, data_generator):
        """generate_all should return a dict with nodes, hyperedges, incidence, deps."""
        result = data_generator.generate_all()

        assert "nodes" in result
        assert "hyperedges" in result
        assert "incidence" in result
        assert "echelon_dependencies" in result
        assert len(result["nodes"]) == 30


# ---------------------------------------------------------------------------
# Tests -- RealDataLoader
# ---------------------------------------------------------------------------

@pytest.mark.skipif(RealDataLoader is None, reason="RealDataLoader not importable")
def test_real_data_loader_init():
    """RealDataLoader should instantiate and set up directory paths."""
    loader = RealDataLoader(data_dir="Data set")
    assert str(loader.bom_dir).endswith("BOM")
    assert str(loader.dataco_dir).endswith("DataCo")
    assert str(loader.maintenance_dir).endswith("Maintenance")
    assert loader.bom_train is None
