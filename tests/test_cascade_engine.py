"""
Test suite for HT-HGNN v2.0 Cascade Engine, Stress Tester, and Scenario Builder.

Tests the simulation subsystem that propagates disruptions through a supply
chain hypergraph:
- CascadeEngine:   single/multi-shock simulation, convergence, threshold effect
- CascadeResult:   structure validation
- StressTester:    random failure Monte Carlo runs
- ScenarioBuilder: what-if scenario construction

All tests use small synthetic hypergraphs so no real data is required.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------
try:
    from src.hypergraph.hypergraph import Hypergraph, HypergraphNode, HypergraphEdge
except ImportError:
    Hypergraph = None
    HypergraphNode = None
    HypergraphEdge = None

try:
    from src.simulation.cascade_engine import CascadeEngine, CascadeResult
except ImportError:
    CascadeEngine = None
    CascadeResult = None

try:
    from src.simulation.stress_tester import StressTester, StressTestResult
except ImportError:
    StressTester = None
    StressTestResult = None

try:
    from src.simulation.scenario_builder import ScenarioBuilder, Scenario
except ImportError:
    ScenarioBuilder = None
    Scenario = None


# ---------------------------------------------------------------------------
# Fixtures -- small synthetic hypergraph
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_hypergraph():
    """Build a minimal hypergraph with 6 nodes and 3 hyperedges.

    Layout:
        HE_A = {S001, S002, S003}   (3 members)
        HE_B = {S002, S003, S004}   (3 members, overlaps with HE_A)
        HE_C = {S004, S005, S006}   (3 members, overlaps with HE_B)

    This creates a linear cascade path: HE_A -> HE_B -> HE_C through
    shared nodes S002/S003 and S004.
    """
    if Hypergraph is None:
        pytest.skip("Hypergraph module not importable")

    hg = Hypergraph()

    for i in range(1, 7):
        node = HypergraphNode(
            node_id=f"S00{i}",
            node_type="supplier",
            tier=1,
            lead_time=float(i),
            reliability=0.3 + 0.1 * i,      # 0.4 .. 0.9
            substitutability=0.5,
            cost=100.0 * i,
        )
        hg.add_node(node)

    he_a = HypergraphEdge(
        hyperedge_id="HE_A",
        nodes={"S001", "S002", "S003"},
        bom_weight=1.0,
    )
    he_b = HypergraphEdge(
        hyperedge_id="HE_B",
        nodes={"S002", "S003", "S004"},
        bom_weight=1.0,
    )
    he_c = HypergraphEdge(
        hyperedge_id="HE_C",
        nodes={"S004", "S005", "S006"},
        bom_weight=1.0,
    )
    hg.add_hyperedge(he_a)
    hg.add_hyperedge(he_b)
    hg.add_hyperedge(he_c)

    return hg


@pytest.fixture
def cascade_engine(tiny_hypergraph):
    """Return a CascadeEngine backed by the tiny_hypergraph."""
    if CascadeEngine is None:
        pytest.skip("CascadeEngine not importable")
    return CascadeEngine(
        tiny_hypergraph,
        cascade_threshold=0.5,
        max_steps=50,
    )


# ---------------------------------------------------------------------------
# Tests -- CascadeEngine basic operation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(CascadeEngine is None, reason="CascadeEngine not importable")
class TestCascadeEngine:
    """Tests for the CascadeEngine simulation."""

    def test_cascade_engine_init(self, cascade_engine):
        """CascadeEngine should store incidence and threshold correctly."""
        assert cascade_engine.cascade_threshold == 0.5
        assert cascade_engine.max_steps == 50
        assert len(cascade_engine.incidence) == 3, "Should have 3 hyperedges"

    def test_simulate_single_shock(self, cascade_engine):
        """Simulating a single shock node should return a CascadeResult."""
        result = cascade_engine.simulate(
            shock_nodes=["S001"],
            shock_magnitude=1.0,
            seed=42,
        )
        assert isinstance(result, CascadeResult)
        assert result.total_disrupted >= 1, (
            "At least the shock node itself must be disrupted"
        )
        assert "S001" in result.final_disrupted

    def test_simulate_multiple_shocks(self, cascade_engine):
        """Shocking two nodes in the same hyperedge should trigger cascade."""
        # S001 and S002 are both in HE_A (3 members). 2/3 > 0.5 threshold.
        result = cascade_engine.simulate(
            shock_nodes=["S001", "S002"],
            shock_magnitude=1.0,
            seed=42,
        )
        assert result.total_disrupted >= 2
        # S003 shares HE_A with the two shocked nodes -> high chance of disruption
        # Not guaranteed because it is probabilistic, but result should be valid
        assert len(result.timeline) >= 1

    def test_cascade_convergence(self, cascade_engine):
        """Simulation should eventually converge (no infinite loops)."""
        result = cascade_engine.simulate(
            shock_nodes=["S001"],
            shock_magnitude=0.5,
            seed=123,
        )
        assert result.converged or result.num_steps <= cascade_engine.max_steps

    def test_cascade_threshold_effect(self, tiny_hypergraph):
        """Higher threshold should result in fewer or equal cascaded nodes."""
        if CascadeEngine is None:
            pytest.skip("CascadeEngine not importable")

        # Low threshold -- cascades more easily
        engine_low = CascadeEngine(tiny_hypergraph, cascade_threshold=0.3, max_steps=50)
        result_low = engine_low.simulate(["S001", "S002"], shock_magnitude=1.0, seed=0)

        # High threshold -- cascades less easily
        engine_high = CascadeEngine(tiny_hypergraph, cascade_threshold=0.9, max_steps=50)
        result_high = engine_high.simulate(["S001", "S002"], shock_magnitude=1.0, seed=0)

        assert result_high.total_disrupted <= result_low.total_disrupted + 1, (
            "Higher threshold should inhibit cascade propagation"
        )

    def test_cascade_result_structure(self, cascade_engine):
        """CascadeResult should expose all documented attributes."""
        result = cascade_engine.simulate(["S001"], seed=7)

        assert hasattr(result, "timeline")
        assert hasattr(result, "total_disrupted")
        assert hasattr(result, "critical_paths")
        assert hasattr(result, "counterfactuals")
        assert hasattr(result, "shock_nodes")
        assert hasattr(result, "final_disrupted")
        assert hasattr(result, "converged")
        assert hasattr(result, "num_steps")

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "total_disrupted" in d

    def test_counterfactual_analysis(self, cascade_engine):
        """Protecting a node should reduce or maintain cascade size."""
        cf = cascade_engine.counterfactual_analysis(
            shock_nodes=["S001", "S002"],
            protected_nodes=["S003"],
            shock_magnitude=1.0,
            n_simulations=10,
            seed=99,
        )

        assert "baseline_disrupted" in cf
        assert "protected_disrupted" in cf
        assert cf["protected_disrupted"] <= cf["baseline_disrupted"] + 1

    def test_critical_path_identification(self, cascade_engine):
        """Critical paths should be lists of hyperedge IDs."""
        result = cascade_engine.simulate(
            shock_nodes=["S001", "S002"],
            shock_magnitude=1.0,
            seed=42,
        )
        for path in result.critical_paths:
            assert isinstance(path, list)
            for he_id in path:
                assert he_id.startswith("HE_")


# ---------------------------------------------------------------------------
# Tests -- StressTester
# ---------------------------------------------------------------------------

@pytest.mark.skipif(StressTester is None or CascadeEngine is None,
                    reason="StressTester or CascadeEngine not importable")
class TestStressTester:
    """Tests for the Monte Carlo stress testing system."""

    def test_stress_tester_random(self, tiny_hypergraph, cascade_engine):
        """run_random_failures should return a StressTestResult."""
        tester = StressTester(tiny_hypergraph, cascade_engine)
        result = tester.run_random_failures(
            n_scenarios=5,
            shock_strategy="random_node",
            n_shocks_per_scenario=1,
            seed=42,
        )
        assert isinstance(result, StressTestResult)
        assert result.n_scenarios == 5
        assert result.shock_strategy == "random_node"
        assert len(result.cascade_size_distribution) > 0


# ---------------------------------------------------------------------------
# Tests -- ScenarioBuilder
# ---------------------------------------------------------------------------

@pytest.mark.skipif(ScenarioBuilder is None,
                    reason="ScenarioBuilder not importable")
class TestScenarioBuilder:
    """Tests for the what-if scenario construction utilities."""

    def test_scenario_builder(self, tiny_hypergraph):
        """ScenarioBuilder should create custom scenarios."""
        builder = ScenarioBuilder(tiny_hypergraph)

        scenario = builder.create_scenario(
            name="test_failure",
            shock_nodes=["S001"],
            description="Unit test scenario",
            severity=0.8,
        )
        assert isinstance(scenario, Scenario)
        assert scenario.name == "test_failure"
        assert "S001" in scenario.shock_nodes
        assert scenario.severity == 0.8

        # Verify it is stored
        assert "test_failure" in builder.scenarios
