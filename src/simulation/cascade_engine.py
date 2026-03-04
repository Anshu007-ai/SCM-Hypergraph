"""
Cascade Engine: Disruption Propagation Simulator for Supply Chain Hypergraphs

Implements the Hypergraph Independent Cascade (HIC) model for simulating
how disruptions propagate through supply chain hypergraph structures.

HIC Algorithm:
    Step 0: Mark shock nodes as disrupted.
    Step 1: For each hyperedge e, compute the fraction of disrupted members.
    Step 2: If fraction > theta (default 0.5), mark remaining members "at-risk".
    Step 3: Each at-risk node activates with probability proportional to
            its cascade risk score.
    Step 4: Repeat until convergence or max steps.

Author: HT-HGNN v2.0 Project
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import copy


@dataclass
class CascadeResult:
    """
    Container for cascade simulation results.

    Attributes:
        timeline: List of step dictionaries, each containing step number,
            disrupted count, newly disrupted nodes, and at-risk nodes.
        total_disrupted: Total number of nodes disrupted by the end.
        critical_paths: List of hyperedge sequences through which
            disruption propagated.
        counterfactuals: Dictionary of counterfactual analysis results.
        shock_nodes: Original set of shock nodes.
        final_disrupted: Set of all disrupted node IDs at termination.
        converged: Whether the simulation reached a steady state.
        num_steps: Number of steps the simulation ran.
    """
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    total_disrupted: int = 0
    critical_paths: List[List[str]] = field(default_factory=list)
    counterfactuals: Dict[str, Any] = field(default_factory=dict)
    shock_nodes: List[str] = field(default_factory=list)
    final_disrupted: Set[str] = field(default_factory=set)
    converged: bool = False
    num_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            'timeline': self.timeline,
            'total_disrupted': self.total_disrupted,
            'critical_paths': self.critical_paths,
            'counterfactuals': self.counterfactuals,
            'shock_nodes': self.shock_nodes,
            'final_disrupted': list(self.final_disrupted),
            'converged': self.converged,
            'num_steps': self.num_steps,
        }


class CascadeEngine:
    """
    Simulates disruption cascades through a supply chain hypergraph.

    Uses the Hypergraph Independent Cascade (HIC) model where disruption
    spreads via hyperedges: if a sufficient fraction of a hyperedge's
    member nodes are disrupted, the remaining members become at-risk
    and may activate (become disrupted) with a probability proportional
    to their cascade risk score.

    Attributes:
        hypergraph: The supply chain hypergraph structure.
        cascade_threshold: Fraction of disrupted members in a hyperedge
            that triggers cascade propagation (theta).
        max_steps: Maximum number of propagation steps.
        incidence: Dictionary mapping hyperedge_id to set of node_ids.
        node_to_hyperedges: Dictionary mapping node_id to set of hyperedge_ids.
    """

    def __init__(
        self,
        hypergraph: Any,
        cascade_threshold: float = 0.5,
        max_steps: int = 100,
    ):
        """
        Initialize the cascade engine.

        Args:
            hypergraph: Hypergraph object with incidence, node_to_hyperedges,
                and nodes attributes.
            cascade_threshold: Fraction of disrupted members that triggers
                cascade in a hyperedge (default 0.5).
            max_steps: Maximum number of propagation steps before stopping.
        """
        self.hypergraph = hypergraph
        self.cascade_threshold = cascade_threshold
        self.max_steps = max_steps

        # Extract incidence structure
        self.incidence: Dict[str, Set[str]] = dict(hypergraph.incidence)
        self.node_to_hyperedges: Dict[str, Set[str]] = dict(
            hypergraph.node_to_hyperedges
        )

        # State tracked during simulation
        self._disrupted: Set[str] = set()
        self._at_risk: Set[str] = set()
        self._timeline: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._propagation_record: Dict[str, List[str]] = defaultdict(list)

    def _reset_state(self):
        """Reset all internal simulation state."""
        self._disrupted = set()
        self._at_risk = set()
        self._timeline = []
        self._step_count = 0
        self._propagation_record = defaultdict(list)

    def _get_node_risk_score(self, node_id: str, shock_magnitude: float) -> float:
        """
        Compute the cascade activation probability for an at-risk node.

        The risk score considers:
        - The fraction of the node's hyperedges that are affected.
        - The node's reliability (from node features).
        - The shock magnitude.

        Args:
            node_id: The at-risk node identifier.
            shock_magnitude: Intensity of the initial shock (0 to 1).

        Returns:
            Activation probability in [0, 1].
        """
        node = self.hypergraph.nodes.get(node_id)
        if node is None:
            return 0.5 * shock_magnitude

        # Count how many of this node's hyperedges have disrupted members
        affected_edges = 0
        total_edges = len(self.node_to_hyperedges.get(node_id, set()))
        if total_edges == 0:
            return 0.0

        for he_id in self.node_to_hyperedges.get(node_id, set()):
            members = self.incidence.get(he_id, set())
            if len(members) == 0:
                continue
            disrupted_fraction = len(members & self._disrupted) / len(members)
            if disrupted_fraction > 0:
                affected_edges += 1

        edge_exposure = affected_edges / total_edges

        # Lower reliability means higher risk of activation
        reliability = getattr(node, 'reliability', 0.5)
        vulnerability = 1.0 - reliability

        # Combined risk score
        risk_score = edge_exposure * vulnerability * shock_magnitude

        return min(max(risk_score, 0.0), 1.0)

    def step(self, shock_magnitude: float = 1.0) -> Dict[str, Any]:
        """
        Perform one propagation step of the HIC algorithm.

        Steps 1-3 of the algorithm:
        1. For each hyperedge, compute fraction of disrupted members.
        2. If fraction > theta, mark remaining members as at-risk.
        3. Each at-risk node activates with probability ~ risk score.

        Args:
            shock_magnitude: Intensity of the initial shock.

        Returns:
            Dictionary with step details:
                - step: Step number.
                - disrupted_count: Total currently disrupted.
                - newly_disrupted: List of newly disrupted node IDs.
                - at_risk: List of currently at-risk node IDs.
        """
        self._step_count += 1
        newly_at_risk = set()

        # Step 1 & 2: Check each hyperedge for cascade threshold
        for he_id, members in self.incidence.items():
            if len(members) == 0:
                continue

            disrupted_in_edge = members & self._disrupted
            disrupted_fraction = len(disrupted_in_edge) / len(members)

            if disrupted_fraction >= self.cascade_threshold:
                # Mark non-disrupted members as at-risk
                remaining = members - self._disrupted
                for node_id in remaining:
                    if node_id not in self._at_risk:
                        newly_at_risk.add(node_id)
                        # Record propagation path
                        self._propagation_record[node_id].append(he_id)

        self._at_risk.update(newly_at_risk)

        # Step 3: Activate at-risk nodes probabilistically
        newly_disrupted = []
        resolved_at_risk = set()

        for node_id in list(self._at_risk):
            if node_id in self._disrupted:
                resolved_at_risk.add(node_id)
                continue

            risk_score = self._get_node_risk_score(node_id, shock_magnitude)

            if np.random.random() < risk_score:
                self._disrupted.add(node_id)
                newly_disrupted.append(node_id)
                resolved_at_risk.add(node_id)

        # Remove activated nodes from at-risk pool
        self._at_risk -= resolved_at_risk

        step_info = {
            'step': self._step_count,
            'disrupted_count': len(self._disrupted),
            'newly_disrupted': newly_disrupted,
            'at_risk': list(self._at_risk),
            'newly_at_risk': list(newly_at_risk),
        }

        self._timeline.append(step_info)
        return step_info

    def simulate(
        self,
        shock_nodes: List[str],
        shock_magnitude: float = 1.0,
        seed: Optional[int] = None,
    ) -> CascadeResult:
        """
        Run a full cascade simulation from initial shock nodes.

        Args:
            shock_nodes: List of node IDs to initially disrupt (Step 0).
            shock_magnitude: Intensity of the shock (0 to 1).
            seed: Optional random seed for reproducibility.

        Returns:
            CascadeResult containing the full simulation timeline,
            total disrupted count, critical paths, and metadata.
        """
        if seed is not None:
            np.random.seed(seed)

        self._reset_state()

        # Step 0: Mark shock nodes as disrupted
        for node_id in shock_nodes:
            if node_id in self.hypergraph.nodes:
                self._disrupted.add(node_id)

        initial_step = {
            'step': 0,
            'disrupted_count': len(self._disrupted),
            'newly_disrupted': list(self._disrupted),
            'at_risk': [],
            'newly_at_risk': [],
        }
        self._timeline.append(initial_step)

        # Steps 1-4: Propagate until convergence or max steps
        converged = False
        for _ in range(self.max_steps):
            prev_disrupted = len(self._disrupted)
            self.step(shock_magnitude)

            # Convergence: no new disruptions and no at-risk nodes
            if (len(self._disrupted) == prev_disrupted
                    and len(self._at_risk) == 0):
                converged = True
                break

        # Identify critical paths
        critical_paths = self.identify_critical_paths()

        result = CascadeResult(
            timeline=self._timeline,
            total_disrupted=len(self._disrupted),
            critical_paths=critical_paths,
            counterfactuals={},
            shock_nodes=list(shock_nodes),
            final_disrupted=set(self._disrupted),
            converged=converged,
            num_steps=self._step_count,
        )

        return result

    def get_timeline(self) -> List[Dict[str, Any]]:
        """
        Retrieve the simulation timeline.

        Returns:
            List of step dictionaries with:
                - step: Step number.
                - disrupted_count: Total disrupted at this step.
                - newly_disrupted: Nodes disrupted in this step.
                - at_risk: Nodes currently at risk.
        """
        return self._timeline

    def identify_critical_paths(self) -> List[List[str]]:
        """
        Identify sequences of hyperedges through which disruption propagated.

        Traces back from each disrupted node (that was not an initial shock)
        through the propagation record to find the chain of hyperedges
        responsible for the cascade.

        Returns:
            List of hyperedge ID sequences, each representing a
            propagation pathway.
        """
        critical_paths = []

        for node_id, hyperedge_chain in self._propagation_record.items():
            if node_id in self._disrupted and len(hyperedge_chain) > 0:
                # Build path: trace through hyperedges
                path = list(dict.fromkeys(hyperedge_chain))  # deduplicate, preserve order
                critical_paths.append(path)

        # Sort by path length (longest cascades first)
        critical_paths.sort(key=len, reverse=True)

        return critical_paths

    def counterfactual_analysis(
        self,
        shock_nodes: List[str],
        protected_nodes: List[str],
        shock_magnitude: float = 1.0,
        n_simulations: int = 10,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform counterfactual analysis: what if certain nodes were protected?

        Runs the cascade twice -- once without protection and once with
        protected nodes immune to disruption -- and compares outcomes.

        Args:
            shock_nodes: Initial shock node IDs.
            protected_nodes: Node IDs to protect (make immune).
            shock_magnitude: Shock intensity.
            n_simulations: Number of Monte Carlo simulations for averaging.
            seed: Optional random seed.

        Returns:
            Dictionary with counterfactual results:
                - baseline_disrupted: Average disrupted without protection.
                - protected_disrupted: Average disrupted with protection.
                - disruption_reduction: Reduction in cascade size.
                - reduction_percentage: Percentage reduction.
                - protected_nodes: List of protected node IDs.
        """
        if seed is not None:
            np.random.seed(seed)

        # Baseline simulations (no protection)
        baseline_counts = []
        for _ in range(n_simulations):
            result = self.simulate(shock_nodes, shock_magnitude)
            baseline_counts.append(result.total_disrupted)

        # Protected simulations
        protected_set = set(protected_nodes)
        protected_counts = []

        for _ in range(n_simulations):
            self._reset_state()

            # Step 0: Mark shock nodes as disrupted (skip protected)
            for node_id in shock_nodes:
                if node_id in self.hypergraph.nodes and node_id not in protected_set:
                    self._disrupted.add(node_id)

            initial_step = {
                'step': 0,
                'disrupted_count': len(self._disrupted),
                'newly_disrupted': list(self._disrupted),
                'at_risk': [],
                'newly_at_risk': [],
            }
            self._timeline.append(initial_step)

            # Propagate with protection
            for _ in range(self.max_steps):
                prev_disrupted = len(self._disrupted)
                step_info = self.step(shock_magnitude)

                # Remove protected nodes from disrupted if they got activated
                self._disrupted -= protected_set

                if (len(self._disrupted) == prev_disrupted
                        and len(self._at_risk) == 0):
                    break

            protected_counts.append(len(self._disrupted))

        baseline_avg = np.mean(baseline_counts)
        protected_avg = np.mean(protected_counts)
        reduction = baseline_avg - protected_avg
        reduction_pct = (reduction / max(baseline_avg, 1)) * 100

        return {
            'baseline_disrupted': float(baseline_avg),
            'protected_disrupted': float(protected_avg),
            'disruption_reduction': float(reduction),
            'reduction_percentage': float(reduction_pct),
            'protected_nodes': protected_nodes,
            'baseline_95th': float(np.percentile(baseline_counts, 95)),
            'protected_95th': float(np.percentile(protected_counts, 95)),
        }

    def summary(self, result: CascadeResult) -> str:
        """
        Generate a text summary of cascade simulation results.

        Args:
            result: CascadeResult from simulate().

        Returns:
            Formatted multi-line string summarizing the cascade.
        """
        lines = [
            "=== Cascade Simulation Summary ===",
            f"Shock nodes:     {result.shock_nodes}",
            f"Total disrupted: {result.total_disrupted} / "
            f"{len(self.hypergraph.nodes)} nodes",
            f"Steps taken:     {result.num_steps}",
            f"Converged:       {result.converged}",
            "",
            "Timeline:",
        ]

        for step_info in result.timeline:
            lines.append(
                f"  Step {step_info['step']:3d}: "
                f"{step_info['disrupted_count']:4d} disrupted, "
                f"{len(step_info['newly_disrupted']):3d} new, "
                f"{len(step_info['at_risk']):3d} at-risk"
            )

        if result.critical_paths:
            lines.extend(["", f"Critical paths ({len(result.critical_paths)}):", ""])
            for i, path in enumerate(result.critical_paths[:5]):
                lines.append(f"  Path {i+1}: {' -> '.join(path)}")

        return "\n".join(lines)


if __name__ == "__main__":
    print("Cascade Engine Module - Disruption Propagation Simulator")
    print("=" * 55)
    print()
    print("Hypergraph Independent Cascade (HIC) Algorithm:")
    print("  Step 0: Mark shock nodes as disrupted")
    print("  Step 1: For each hyperedge e, compute fraction of disrupted members")
    print("  Step 2: If fraction > theta, mark remaining nodes 'at-risk'")
    print("  Step 3: Each at-risk node activates with prob ~ risk score")
    print("  Step 4: Repeat until convergence or max steps")
    print()
    print("Features:")
    print("  - Full cascade simulation with timeline tracking")
    print("  - Critical path identification")
    print("  - Counterfactual analysis (node protection)")
    print("  - Configurable cascade threshold and max steps")
    print()
    print("Usage example:")
    print("  engine = CascadeEngine(hypergraph, cascade_threshold=0.5)")
    print("  result = engine.simulate(shock_nodes=['S001'], shock_magnitude=0.8)")
    print("  print(engine.summary(result))")
    print()
    print("Module ready for integration with HT-HGNN v2.0.")
