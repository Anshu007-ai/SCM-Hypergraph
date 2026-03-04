"""
Stress Tester: Automated Stress Testing for Supply Chain Hypergraphs

Provides automated scenario generation and stress testing by running
thousands of cascade simulations with various shock strategies to identify
system vulnerabilities, critical nodes, and resilient hyperedges.

Shock strategies:
- random_node: Randomly select nodes to fail
- random_hyperedge: Randomly disable entire hyperedges
- targeted_high_degree: Target the highest-degree nodes first
- geographic_cluster: Simulate geographic-region failures

Author: HT-HGNN v2.0 Project
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class StressTestResult:
    """
    Container for stress test results.

    Attributes:
        n_scenarios: Number of scenarios that were run.
        shock_strategy: Name of the shock strategy used.
        mean_cascade_size: Mean number of nodes disrupted across simulations.
        percentile_95_cascade_size: 95th percentile of cascade sizes.
        max_cascade_size: Largest observed cascade.
        min_cascade_size: Smallest observed cascade.
        top_dangerous_nodes: Top nodes whose failure causes the largest cascades.
        most_resilient_hyperedges: Hyperedges whose removal most reduces cascade risk.
        vulnerability_score: Overall system vulnerability metric (0 to 1).
        cascade_size_distribution: List of all cascade sizes for further analysis.
        convergence_rate: Fraction of simulations that converged.
    """
    n_scenarios: int = 0
    shock_strategy: str = ''
    mean_cascade_size: float = 0.0
    percentile_95_cascade_size: float = 0.0
    max_cascade_size: int = 0
    min_cascade_size: int = 0
    top_dangerous_nodes: List[Tuple[str, float]] = field(default_factory=list)
    most_resilient_hyperedges: List[Tuple[str, float]] = field(default_factory=list)
    vulnerability_score: float = 0.0
    cascade_size_distribution: List[int] = field(default_factory=list)
    convergence_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            'n_scenarios': self.n_scenarios,
            'shock_strategy': self.shock_strategy,
            'mean_cascade_size': self.mean_cascade_size,
            'percentile_95_cascade_size': self.percentile_95_cascade_size,
            'max_cascade_size': self.max_cascade_size,
            'min_cascade_size': self.min_cascade_size,
            'top_dangerous_nodes': self.top_dangerous_nodes,
            'most_resilient_hyperedges': self.most_resilient_hyperedges,
            'vulnerability_score': self.vulnerability_score,
            'cascade_size_distribution': self.cascade_size_distribution,
            'convergence_rate': self.convergence_rate,
        }


class StressTester:
    """
    Automated stress testing for supply chain hypergraph systems.

    Runs Monte Carlo simulations with configurable shock strategies
    to assess system resilience, identify critical failure points,
    and rank nodes/hyperedges by their vulnerability impact.

    Attributes:
        hypergraph: The supply chain hypergraph structure.
        cascade_engine: A CascadeEngine instance for running simulations.
        node_ids: List of all node IDs in the hypergraph.
        hyperedge_ids: List of all hyperedge IDs.
    """

    def __init__(self, hypergraph: Any, cascade_engine: Any):
        """
        Initialize the stress tester.

        Args:
            hypergraph: Hypergraph object with nodes, hyperedges, incidence,
                and node_to_hyperedges attributes.
            cascade_engine: A CascadeEngine instance configured for
                this hypergraph.
        """
        self.hypergraph = hypergraph
        self.cascade_engine = cascade_engine
        self.node_ids = list(hypergraph.nodes.keys())
        self.hyperedge_ids = list(hypergraph.hyperedges.keys())

    def _select_shock_nodes(
        self,
        strategy: str,
        n_shocks: int = 1,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Select nodes to shock based on the given strategy.

        Args:
            strategy: Shock strategy name.
            n_shocks: Number of nodes to shock.
            seed: Random seed for reproducibility.

        Returns:
            List of node IDs to disrupt.

        Raises:
            ValueError: If an unsupported strategy is specified.
        """
        if seed is not None:
            np.random.seed(seed)

        n_shocks = min(n_shocks, len(self.node_ids))

        if strategy == 'random_node':
            return list(np.random.choice(self.node_ids, size=n_shocks, replace=False))

        elif strategy == 'random_hyperedge':
            # Select a random hyperedge and shock all its members
            he_id = np.random.choice(self.hyperedge_ids)
            members = list(self.hypergraph.incidence.get(he_id, set()))
            return members[:n_shocks] if members else []

        elif strategy == 'targeted_high_degree':
            # Target nodes with the most hyperedge connections
            node_degrees = {
                nid: len(self.hypergraph.node_to_hyperedges.get(nid, set()))
                for nid in self.node_ids
            }
            sorted_nodes = sorted(
                node_degrees.items(), key=lambda x: x[1], reverse=True
            )
            return [nid for nid, _ in sorted_nodes[:n_shocks]]

        elif strategy == 'geographic_cluster':
            # Simulate geographic clustering by selecting a seed node
            # and its immediate hypergraph neighbors
            seed_node = np.random.choice(self.node_ids)
            cluster = {seed_node}

            # Add neighbors through shared hyperedges
            for he_id in self.hypergraph.node_to_hyperedges.get(seed_node, set()):
                members = self.hypergraph.incidence.get(he_id, set())
                cluster.update(members)
                if len(cluster) >= n_shocks:
                    break

            return list(cluster)[:n_shocks]

        else:
            raise ValueError(
                f"Unknown shock strategy '{strategy}'. "
                f"Supported: 'random_node', 'random_hyperedge', "
                f"'targeted_high_degree', 'geographic_cluster'"
            )

    def run_random_failures(
        self,
        n_scenarios: int = 1000,
        shock_strategy: str = 'random_node',
        n_shocks_per_scenario: int = 1,
        shock_magnitude: float = 1.0,
        seed: Optional[int] = None,
    ) -> StressTestResult:
        """
        Run multiple cascade simulations with the specified shock strategy.

        Args:
            n_scenarios: Number of independent simulations to run.
            shock_strategy: Strategy for selecting shock nodes.
            n_shocks_per_scenario: Number of nodes to shock per scenario.
            shock_magnitude: Intensity of each shock.
            seed: Optional master random seed.

        Returns:
            StressTestResult with aggregate statistics.
        """
        if seed is not None:
            np.random.seed(seed)

        cascade_sizes = []
        convergence_count = 0
        node_disruption_counts: Dict[str, int] = defaultdict(int)

        for i in range(n_scenarios):
            scenario_seed = seed + i if seed is not None else None
            shock_nodes = self._select_shock_nodes(
                strategy=shock_strategy,
                n_shocks=n_shocks_per_scenario,
                seed=scenario_seed,
            )

            if not shock_nodes:
                continue

            result = self.cascade_engine.simulate(
                shock_nodes=shock_nodes,
                shock_magnitude=shock_magnitude,
                seed=scenario_seed,
            )

            cascade_sizes.append(result.total_disrupted)
            if result.converged:
                convergence_count += 1

            # Track which nodes get disrupted most often
            for node_id in result.final_disrupted:
                node_disruption_counts[node_id] += 1

        if not cascade_sizes:
            return StressTestResult(
                n_scenarios=n_scenarios,
                shock_strategy=shock_strategy,
            )

        # Compute statistics
        cascade_array = np.array(cascade_sizes)
        total_nodes = len(self.node_ids)

        # Top dangerous nodes (most frequently disrupted)
        sorted_nodes = sorted(
            node_disruption_counts.items(), key=lambda x: x[1], reverse=True
        )
        top_dangerous = [
            (nid, count / n_scenarios) for nid, count in sorted_nodes[:10]
        ]

        # Vulnerability score: mean cascade / total nodes
        vulnerability = float(cascade_array.mean() / max(total_nodes, 1))

        stress_result = StressTestResult(
            n_scenarios=n_scenarios,
            shock_strategy=shock_strategy,
            mean_cascade_size=float(cascade_array.mean()),
            percentile_95_cascade_size=float(np.percentile(cascade_array, 95)),
            max_cascade_size=int(cascade_array.max()),
            min_cascade_size=int(cascade_array.min()),
            top_dangerous_nodes=top_dangerous,
            most_resilient_hyperedges=[],  # Computed separately
            vulnerability_score=min(vulnerability, 1.0),
            cascade_size_distribution=cascade_sizes,
            convergence_rate=convergence_count / n_scenarios,
        )

        return stress_result

    def find_critical_nodes(
        self,
        top_k: int = 10,
        shock_magnitude: float = 1.0,
        n_simulations_per_node: int = 10,
        seed: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find the top-k nodes whose individual failure causes the largest cascades.

        Runs simulations with each node as the sole shock source and ranks
        by average cascade size.

        Args:
            top_k: Number of critical nodes to return.
            shock_magnitude: Shock intensity for each simulation.
            n_simulations_per_node: Monte Carlo runs per node for averaging.
            seed: Optional random seed.

        Returns:
            List of (node_id, average_cascade_size) tuples, sorted by
            cascade size in descending order.
        """
        if seed is not None:
            np.random.seed(seed)

        node_cascade_scores: Dict[str, float] = {}

        for node_id in self.node_ids:
            cascade_sizes = []
            for sim in range(n_simulations_per_node):
                sim_seed = None
                if seed is not None:
                    sim_seed = seed + hash(node_id) + sim

                result = self.cascade_engine.simulate(
                    shock_nodes=[node_id],
                    shock_magnitude=shock_magnitude,
                    seed=sim_seed,
                )
                cascade_sizes.append(result.total_disrupted)

            node_cascade_scores[node_id] = float(np.mean(cascade_sizes))

        # Sort and return top-k
        sorted_nodes = sorted(
            node_cascade_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_nodes[:top_k]

    def find_resilient_hyperedges(
        self,
        top_k: int = 10,
        shock_nodes: Optional[List[str]] = None,
        shock_magnitude: float = 1.0,
        n_simulations: int = 10,
        seed: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find hyperedges whose removal most reduces cascade risk.

        For each hyperedge, temporarily removes it from the incidence
        structure and measures the resulting cascade size reduction.

        Args:
            top_k: Number of resilient hyperedges to return.
            shock_nodes: Specific shock nodes to test. If None, uses
                the top-3 highest-degree nodes.
            shock_magnitude: Shock intensity.
            n_simulations: Monte Carlo runs for averaging.
            seed: Optional random seed.

        Returns:
            List of (hyperedge_id, cascade_reduction) tuples, sorted by
            reduction in descending order (most impactful first).
        """
        if seed is not None:
            np.random.seed(seed)

        # Default shock nodes: top-3 highest degree
        if shock_nodes is None:
            node_degrees = {
                nid: len(self.hypergraph.node_to_hyperedges.get(nid, set()))
                for nid in self.node_ids
            }
            sorted_by_degree = sorted(
                node_degrees.items(), key=lambda x: x[1], reverse=True
            )
            shock_nodes = [nid for nid, _ in sorted_by_degree[:3]]

        # Baseline cascade size
        baseline_sizes = []
        for _ in range(n_simulations):
            result = self.cascade_engine.simulate(
                shock_nodes=shock_nodes,
                shock_magnitude=shock_magnitude,
            )
            baseline_sizes.append(result.total_disrupted)
        baseline_avg = np.mean(baseline_sizes)

        # Test each hyperedge removal
        hyperedge_reductions: Dict[str, float] = {}
        original_incidence = dict(self.cascade_engine.incidence)
        original_n2he = {
            k: set(v) for k, v in self.cascade_engine.node_to_hyperedges.items()
        }

        for he_id in self.hyperedge_ids:
            # Temporarily remove hyperedge
            members = self.cascade_engine.incidence.pop(he_id, set())
            for node_id in members:
                node_hes = self.cascade_engine.node_to_hyperedges.get(node_id, set())
                node_hes.discard(he_id)

            # Run simulations without this hyperedge
            reduced_sizes = []
            for _ in range(n_simulations):
                result = self.cascade_engine.simulate(
                    shock_nodes=shock_nodes,
                    shock_magnitude=shock_magnitude,
                )
                reduced_sizes.append(result.total_disrupted)

            reduced_avg = np.mean(reduced_sizes)
            reduction = baseline_avg - reduced_avg
            hyperedge_reductions[he_id] = float(reduction)

            # Restore hyperedge
            self.cascade_engine.incidence[he_id] = members
            for node_id in members:
                if node_id not in self.cascade_engine.node_to_hyperedges:
                    self.cascade_engine.node_to_hyperedges[node_id] = set()
                self.cascade_engine.node_to_hyperedges[node_id].add(he_id)

        # Restore full state (safety measure)
        self.cascade_engine.incidence = original_incidence
        self.cascade_engine.node_to_hyperedges = original_n2he

        # Sort by reduction (largest first = most resilient when removed)
        sorted_edges = sorted(
            hyperedge_reductions.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_edges[:top_k]

    def comprehensive_stress_test(
        self,
        n_scenarios_per_strategy: int = 250,
        shock_magnitude: float = 1.0,
        seed: Optional[int] = None,
    ) -> Dict[str, StressTestResult]:
        """
        Run stress tests across all four shock strategies.

        Args:
            n_scenarios_per_strategy: Simulations per strategy.
            shock_magnitude: Shock intensity.
            seed: Optional random seed.

        Returns:
            Dictionary mapping strategy name to its StressTestResult.
        """
        strategies = [
            'random_node',
            'random_hyperedge',
            'targeted_high_degree',
            'geographic_cluster',
        ]

        results = {}
        for i, strategy in enumerate(strategies):
            strategy_seed = seed + i * 10000 if seed is not None else None
            results[strategy] = self.run_random_failures(
                n_scenarios=n_scenarios_per_strategy,
                shock_strategy=strategy,
                shock_magnitude=shock_magnitude,
                seed=strategy_seed,
            )

        return results

    def summary(self, result: StressTestResult) -> str:
        """
        Generate a human-readable summary of stress test results.

        Args:
            result: StressTestResult from run_random_failures().

        Returns:
            Formatted multi-line string.
        """
        lines = [
            "=== Stress Test Summary ===",
            f"Strategy:              {result.shock_strategy}",
            f"Scenarios run:         {result.n_scenarios}",
            f"Convergence rate:      {result.convergence_rate:.1%}",
            "",
            "Cascade Size Statistics:",
            f"  Mean:                {result.mean_cascade_size:.1f}",
            f"  95th percentile:     {result.percentile_95_cascade_size:.1f}",
            f"  Max:                 {result.max_cascade_size}",
            f"  Min:                 {result.min_cascade_size}",
            "",
            f"Vulnerability Score:   {result.vulnerability_score:.4f}",
            "",
        ]

        if result.top_dangerous_nodes:
            lines.append(f"Top {len(result.top_dangerous_nodes)} Dangerous Nodes:")
            for i, (nid, freq) in enumerate(result.top_dangerous_nodes):
                bar = "#" * int(freq * 30)
                lines.append(f"  {i+1:3d}. {nid:<15s} freq={freq:.3f} |{bar}")

        if result.most_resilient_hyperedges:
            lines.extend(["", f"Top Resilient Hyperedges:"])
            for i, (hid, red) in enumerate(result.most_resilient_hyperedges):
                lines.append(f"  {i+1:3d}. {hid:<20s} reduction={red:.2f}")

        return "\n".join(lines)

    def comparative_summary(
        self,
        results: Dict[str, StressTestResult],
    ) -> str:
        """
        Generate a comparative summary across multiple strategies.

        Args:
            results: Dictionary of strategy -> StressTestResult.

        Returns:
            Formatted comparison string.
        """
        lines = [
            "=== Comparative Stress Test Summary ===",
            "",
            f"{'Strategy':<25s} {'Mean':>8s} {'P95':>8s} {'Max':>6s} {'Vuln':>8s}",
            "-" * 60,
        ]

        for strategy, result in results.items():
            lines.append(
                f"{strategy:<25s} "
                f"{result.mean_cascade_size:>8.1f} "
                f"{result.percentile_95_cascade_size:>8.1f} "
                f"{result.max_cascade_size:>6d} "
                f"{result.vulnerability_score:>8.4f}"
            )

        # Find most vulnerable strategy
        most_vulnerable = max(results.items(), key=lambda x: x[1].vulnerability_score)
        lines.extend([
            "",
            f"Most vulnerable strategy: {most_vulnerable[0]} "
            f"(score={most_vulnerable[1].vulnerability_score:.4f})",
        ])

        return "\n".join(lines)


if __name__ == "__main__":
    print("Stress Tester Module - Automated Stress Testing")
    print("=" * 50)
    print()
    print("Shock strategies:")
    print("  1. random_node          - Random individual node failures")
    print("  2. random_hyperedge     - Random complete hyperedge failures")
    print("  3. targeted_high_degree - Target the most connected nodes")
    print("  4. geographic_cluster   - Simulate regional cluster failures")
    print()
    print("Analysis capabilities:")
    print("  - Monte Carlo cascade size estimation")
    print("  - Critical node identification")
    print("  - Resilient hyperedge discovery")
    print("  - Comprehensive multi-strategy comparison")
    print()
    print("Usage example:")
    print("  tester = StressTester(hypergraph, cascade_engine)")
    print("  result = tester.run_random_failures(n_scenarios=1000)")
    print("  critical = tester.find_critical_nodes(top_k=10)")
    print("  resilient = tester.find_resilient_hyperedges(top_k=10)")
    print()
    print("Module ready for integration with HT-HGNN v2.0.")
