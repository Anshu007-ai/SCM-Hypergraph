"""
Scenario Builder: What-If Scenario Construction for Supply Chain Hypergraphs

Provides a structured way to define, build, and manage what-if scenarios
for cascade simulations. Includes pre-built scenario templates for
common supply chain disruption patterns:
- Port shutdowns
- Supplier failures
- Demand shocks

Scenarios can be serialized, combined, and passed to the CascadeEngine
for simulation.

Author: HT-HGNN v2.0 Project
"""

import numpy as np
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Scenario:
    """
    Represents a what-if disruption scenario.

    Attributes:
        name: Human-readable scenario name.
        shock_nodes: List of node IDs that are initially disrupted.
        modified_features: Dictionary of {node_id: {feature: new_value}}
            defining feature modifications for the scenario.
        timestamp: When the scenario was created.
        description: Human-readable description of the scenario.
        severity: Overall severity level (0.0 to 1.0).
        scenario_type: Category of the scenario (e.g., 'port_shutdown').
        metadata: Additional arbitrary metadata.
    """
    name: str
    shock_nodes: List[str]
    modified_features: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ''
    severity: float = 1.0
    scenario_type: str = 'custom'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to a serializable dictionary."""
        return {
            'name': self.name,
            'shock_nodes': self.shock_nodes,
            'modified_features': self.modified_features,
            'timestamp': self.timestamp,
            'description': self.description,
            'severity': self.severity,
            'scenario_type': self.scenario_type,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scenario':
        """Reconstruct a Scenario from a dictionary."""
        return cls(
            name=data['name'],
            shock_nodes=data['shock_nodes'],
            modified_features=data.get('modified_features', {}),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            description=data.get('description', ''),
            severity=data.get('severity', 1.0),
            scenario_type=data.get('scenario_type', 'custom'),
            metadata=data.get('metadata', {}),
        )

    def __repr__(self) -> str:
        return (
            f"Scenario(name='{self.name}', type='{self.scenario_type}', "
            f"shock_nodes={len(self.shock_nodes)}, severity={self.severity:.2f})"
        )


class ScenarioBuilder:
    """
    Constructs and manages what-if scenarios for supply chain analysis.

    Provides both generic scenario creation and domain-specific templates
    for common supply chain disruption patterns (port shutdowns, supplier
    failures, demand shocks).

    Attributes:
        hypergraph: The supply chain hypergraph structure.
        scenarios: Dictionary of named scenarios that have been created.
    """

    def __init__(self, hypergraph: Any):
        """
        Initialize the scenario builder.

        Args:
            hypergraph: Hypergraph object with nodes, hyperedges, incidence,
                and node_to_hyperedges attributes.
        """
        self.hypergraph = hypergraph
        self.scenarios: Dict[str, Scenario] = {}

    def create_scenario(
        self,
        name: str,
        shock_nodes: List[str],
        modifications: Optional[Dict[str, Dict[str, float]]] = None,
        description: str = '',
        severity: float = 1.0,
        scenario_type: str = 'custom',
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Scenario:
        """
        Create a custom what-if scenario.

        Args:
            name: Unique human-readable scenario name.
            shock_nodes: List of node IDs to initially disrupt.
            modifications: Optional dict of {node_id: {feature: new_value}}
                for feature modifications to apply.
            description: Text description of the scenario.
            severity: Severity level (0.0 = mild, 1.0 = extreme).
            scenario_type: Category label for the scenario.
            metadata: Additional arbitrary key-value metadata.

        Returns:
            The constructed Scenario object.

        Raises:
            ValueError: If the scenario name already exists.
        """
        if name in self.scenarios:
            raise ValueError(
                f"Scenario '{name}' already exists. Use a unique name or "
                f"call remove_scenario('{name}') first."
            )

        # Validate shock nodes exist in the hypergraph
        valid_shock_nodes = []
        for node_id in shock_nodes:
            if node_id in self.hypergraph.nodes:
                valid_shock_nodes.append(node_id)

        scenario = Scenario(
            name=name,
            shock_nodes=valid_shock_nodes,
            modified_features=modifications or {},
            description=description,
            severity=severity,
            scenario_type=scenario_type,
            metadata=metadata or {},
        )

        self.scenarios[name] = scenario
        return scenario

    def build_port_shutdown_scenario(
        self,
        port_ids: List[str],
        description: Optional[str] = None,
    ) -> Scenario:
        """
        Build a scenario simulating port shutdowns.

        Identifies all nodes connected to the specified port nodes and
        marks them as disrupted. Reduces reliability and increases lead
        time for affected nodes.

        Args:
            port_ids: List of node IDs representing ports to shut down.
            description: Optional custom description.

        Returns:
            Scenario object for the port shutdown.
        """
        shock_nodes = list(port_ids)
        affected_nodes: Set[str] = set()

        # Find all nodes connected to ports via shared hyperedges
        for port_id in port_ids:
            if port_id not in self.hypergraph.node_to_hyperedges:
                continue
            for he_id in self.hypergraph.node_to_hyperedges[port_id]:
                members = self.hypergraph.incidence.get(he_id, set())
                affected_nodes.update(members)

        # Extend shock nodes to include directly connected nodes
        shock_nodes.extend([
            nid for nid in affected_nodes
            if nid not in shock_nodes and nid in self.hypergraph.nodes
        ])

        # Feature modifications: reduce reliability, increase lead time
        modifications = {}
        for node_id in affected_nodes:
            if node_id in self.hypergraph.nodes:
                node = self.hypergraph.nodes[node_id]
                modifications[node_id] = {
                    'reliability': max(getattr(node, 'reliability', 0.5) * 0.3, 0.0),
                    'lead_time': getattr(node, 'lead_time', 1.0) * 3.0,
                }

        if description is None:
            port_names = ', '.join(port_ids[:5])
            if len(port_ids) > 5:
                port_names += f' (+{len(port_ids) - 5} more)'
            description = (
                f"Port shutdown scenario affecting {len(port_ids)} port(s): "
                f"{port_names}. "
                f"Total affected nodes: {len(affected_nodes)}."
            )

        name = f"port_shutdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return self.create_scenario(
            name=name,
            shock_nodes=shock_nodes,
            modifications=modifications,
            description=description,
            severity=0.9,
            scenario_type='port_shutdown',
            metadata={
                'port_ids': port_ids,
                'affected_node_count': len(affected_nodes),
            },
        )

    def build_supplier_failure_scenario(
        self,
        supplier_ids: List[str],
        severity: float = 1.0,
        description: Optional[str] = None,
    ) -> Scenario:
        """
        Build a scenario simulating supplier failures.

        Marks specified supplier nodes as disrupted and degrades the
        features of nodes in downstream hyperedges proportionally to
        the severity.

        Args:
            supplier_ids: List of supplier node IDs that fail.
            severity: Failure severity (0.0 = partial, 1.0 = complete).
            description: Optional custom description.

        Returns:
            Scenario object for the supplier failure.
        """
        shock_nodes = [
            sid for sid in supplier_ids
            if sid in self.hypergraph.nodes
        ]

        # Find downstream affected nodes
        affected_hyperedges: Set[str] = set()
        for sid in shock_nodes:
            for he_id in self.hypergraph.node_to_hyperedges.get(sid, set()):
                affected_hyperedges.add(he_id)

        # Apply feature modifications based on severity
        modifications = {}

        # Directly failed suppliers: set reliability to 0
        for sid in shock_nodes:
            if sid in self.hypergraph.nodes:
                node = self.hypergraph.nodes[sid]
                modifications[sid] = {
                    'reliability': 0.0,
                    'lead_time': getattr(node, 'lead_time', 1.0) * (1 + 4 * severity),
                    'cost': getattr(node, 'cost', 1.0) * (1 + 2 * severity),
                }

        # Downstream nodes: degrade proportionally
        for he_id in affected_hyperedges:
            members = self.hypergraph.incidence.get(he_id, set())
            for node_id in members:
                if node_id not in shock_nodes and node_id in self.hypergraph.nodes:
                    node = self.hypergraph.nodes[node_id]
                    degradation = severity * 0.5  # Partial degradation
                    if node_id not in modifications:
                        modifications[node_id] = {}
                    modifications[node_id]['reliability'] = max(
                        getattr(node, 'reliability', 0.5) * (1 - degradation), 0.0
                    )

        if description is None:
            supplier_names = ', '.join(supplier_ids[:5])
            if len(supplier_ids) > 5:
                supplier_names += f' (+{len(supplier_ids) - 5} more)'
            description = (
                f"Supplier failure scenario: {len(supplier_ids)} supplier(s) "
                f"({supplier_names}) failing at severity {severity:.1f}. "
                f"Affected hyperedges: {len(affected_hyperedges)}."
            )

        name = f"supplier_failure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return self.create_scenario(
            name=name,
            shock_nodes=shock_nodes,
            modifications=modifications,
            description=description,
            severity=severity,
            scenario_type='supplier_failure',
            metadata={
                'supplier_ids': supplier_ids,
                'affected_hyperedge_count': len(affected_hyperedges),
            },
        )

    def build_demand_shock_scenario(
        self,
        product_ids: List[str],
        magnitude: float = 2.0,
        description: Optional[str] = None,
    ) -> Scenario:
        """
        Build a scenario simulating demand shocks on specific products.

        Demand shocks propagate upstream through the supply chain,
        placing stress on supplier nodes connected to the affected
        products via hyperedges.

        Args:
            product_ids: List of product/part node IDs experiencing demand shock.
            magnitude: Demand multiplier (e.g., 2.0 = double demand).
            description: Optional custom description.

        Returns:
            Scenario object for the demand shock.
        """
        # Product nodes that experience the shock
        valid_product_ids = [
            pid for pid in product_ids
            if pid in self.hypergraph.nodes
        ]

        # Demand shock: products are not "disrupted" but their features change
        # Upstream suppliers face increased stress
        shock_nodes = []  # Demand shocks don't directly disrupt
        modifications = {}

        # Identify upstream suppliers
        upstream_nodes: Set[str] = set()
        for pid in valid_product_ids:
            for he_id in self.hypergraph.node_to_hyperedges.get(pid, set()):
                members = self.hypergraph.incidence.get(he_id, set())
                upstream_nodes.update(members)

        # Product nodes: increase demand-related cost
        for pid in valid_product_ids:
            if pid in self.hypergraph.nodes:
                node = self.hypergraph.nodes[pid]
                modifications[pid] = {
                    'cost': getattr(node, 'cost', 1.0) * magnitude,
                }

        # Upstream suppliers: increase lead time and reduce reliability
        # proportional to demand surge
        stress_factor = min(magnitude / 5.0, 1.0)  # Cap at 1.0
        for node_id in upstream_nodes:
            if node_id not in valid_product_ids and node_id in self.hypergraph.nodes:
                node = self.hypergraph.nodes[node_id]
                if node_id not in modifications:
                    modifications[node_id] = {}
                modifications[node_id]['lead_time'] = (
                    getattr(node, 'lead_time', 1.0) * (1 + stress_factor)
                )
                modifications[node_id]['reliability'] = max(
                    getattr(node, 'reliability', 0.5) * (1 - stress_factor * 0.3),
                    0.0
                )

                # If stress is high enough, these become shock nodes
                if stress_factor > 0.7:
                    shock_nodes.append(node_id)

        if description is None:
            product_names = ', '.join(product_ids[:5])
            if len(product_ids) > 5:
                product_names += f' (+{len(product_ids) - 5} more)'
            description = (
                f"Demand shock scenario: {magnitude:.1f}x demand surge on "
                f"{len(product_ids)} product(s) ({product_names}). "
                f"Upstream stress propagated to {len(upstream_nodes)} nodes."
            )

        name = f"demand_shock_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return self.create_scenario(
            name=name,
            shock_nodes=shock_nodes,
            modifications=modifications,
            description=description,
            severity=min(magnitude / 5.0, 1.0),
            scenario_type='demand_shock',
            metadata={
                'product_ids': product_ids,
                'magnitude': magnitude,
                'upstream_node_count': len(upstream_nodes),
            },
        )

    def combine_scenarios(
        self,
        scenario_names: List[str],
        combined_name: str,
    ) -> Scenario:
        """
        Combine multiple scenarios into a single compound scenario.

        Merges shock nodes and feature modifications. Where modifications
        conflict (same node, same feature), the more severe value is used.

        Args:
            scenario_names: List of existing scenario names to combine.
            combined_name: Name for the new combined scenario.

        Returns:
            New combined Scenario object.

        Raises:
            ValueError: If any scenario name is not found.
        """
        combined_shocks: List[str] = []
        combined_mods: Dict[str, Dict[str, float]] = {}
        descriptions = []
        max_severity = 0.0

        for sname in scenario_names:
            if sname not in self.scenarios:
                raise ValueError(f"Scenario '{sname}' not found.")

            scenario = self.scenarios[sname]
            combined_shocks.extend(scenario.shock_nodes)
            descriptions.append(scenario.description)
            max_severity = max(max_severity, scenario.severity)

            # Merge modifications (take worst case for each feature)
            for node_id, features in scenario.modified_features.items():
                if node_id not in combined_mods:
                    combined_mods[node_id] = {}
                for feat, val in features.items():
                    if feat in combined_mods[node_id]:
                        # For reliability: take the lower value (worse)
                        if feat == 'reliability':
                            combined_mods[node_id][feat] = min(
                                combined_mods[node_id][feat], val
                            )
                        # For cost/lead_time: take the higher value (worse)
                        else:
                            combined_mods[node_id][feat] = max(
                                combined_mods[node_id][feat], val
                            )
                    else:
                        combined_mods[node_id][feat] = val

        # Deduplicate shock nodes
        combined_shocks = list(dict.fromkeys(combined_shocks))

        description = (
            f"Combined scenario from {len(scenario_names)} scenarios: "
            + " | ".join(descriptions)
        )

        return self.create_scenario(
            name=combined_name,
            shock_nodes=combined_shocks,
            modifications=combined_mods,
            description=description,
            severity=max_severity,
            scenario_type='combined',
            metadata={
                'source_scenarios': scenario_names,
            },
        )

    def remove_scenario(self, name: str) -> bool:
        """
        Remove a scenario by name.

        Args:
            name: Name of the scenario to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self.scenarios:
            del self.scenarios[name]
            return True
        return False

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """
        List all currently defined scenarios.

        Returns:
            List of scenario summary dictionaries.
        """
        summaries = []
        for name, scenario in self.scenarios.items():
            summaries.append({
                'name': name,
                'type': scenario.scenario_type,
                'shock_nodes': len(scenario.shock_nodes),
                'modified_nodes': len(scenario.modified_features),
                'severity': scenario.severity,
                'timestamp': scenario.timestamp,
                'description': scenario.description[:100],
            })
        return summaries

    def get_scenario(self, name: str) -> Optional[Scenario]:
        """
        Retrieve a scenario by name.

        Args:
            name: Scenario name.

        Returns:
            Scenario object if found, None otherwise.
        """
        return self.scenarios.get(name)

    def summary(self, scenario: Optional[Scenario] = None) -> str:
        """
        Generate a text summary of a scenario or all scenarios.

        Args:
            scenario: Specific scenario to summarize. If None,
                summarizes all registered scenarios.

        Returns:
            Formatted multi-line string.
        """
        if scenario is not None:
            return self._single_scenario_summary(scenario)

        lines = [
            "=== Scenario Builder Summary ===",
            f"Total scenarios: {len(self.scenarios)}",
            f"Total nodes in hypergraph: {len(self.hypergraph.nodes)}",
            "",
        ]

        for name, sc in self.scenarios.items():
            lines.append(
                f"  [{sc.scenario_type:<20s}] {name:<35s} "
                f"severity={sc.severity:.2f}  "
                f"shocks={len(sc.shock_nodes):3d}  "
                f"mods={len(sc.modified_features):3d}"
            )

        return "\n".join(lines)

    def _single_scenario_summary(self, scenario: Scenario) -> str:
        """Generate summary for a single scenario."""
        lines = [
            f"=== Scenario: {scenario.name} ===",
            f"Type:        {scenario.scenario_type}",
            f"Severity:    {scenario.severity:.2f}",
            f"Created:     {scenario.timestamp}",
            f"Description: {scenario.description}",
            "",
            f"Shock nodes ({len(scenario.shock_nodes)}):",
        ]

        for nid in scenario.shock_nodes[:10]:
            lines.append(f"  - {nid}")
        if len(scenario.shock_nodes) > 10:
            lines.append(f"  ... and {len(scenario.shock_nodes) - 10} more")

        lines.extend([
            "",
            f"Modified features ({len(scenario.modified_features)} nodes):",
        ])

        for nid, feats in list(scenario.modified_features.items())[:5]:
            feat_str = ', '.join(f"{k}={v:.3f}" for k, v in feats.items())
            lines.append(f"  {nid}: {feat_str}")
        if len(scenario.modified_features) > 5:
            lines.append(
                f"  ... and {len(scenario.modified_features) - 5} more nodes"
            )

        if scenario.metadata:
            lines.extend(["", "Metadata:"])
            for k, v in scenario.metadata.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)


if __name__ == "__main__":
    print("Scenario Builder Module - What-If Scenario Construction")
    print("=" * 55)
    print()
    print("Pre-built scenario templates:")
    print("  1. Port Shutdown     - Simulate port/hub closures")
    print("  2. Supplier Failure  - Simulate supplier disruptions")
    print("  3. Demand Shock      - Simulate demand surges")
    print("  4. Custom            - Build arbitrary scenarios")
    print("  5. Combined          - Merge multiple scenarios")
    print()
    print("Features:")
    print("  - Scenario creation with feature modifications")
    print("  - Automatic upstream/downstream impact analysis")
    print("  - Scenario serialization (to_dict / from_dict)")
    print("  - Scenario combination and comparison")
    print()
    print("Usage example:")
    print("  builder = ScenarioBuilder(hypergraph)")
    print("  scenario = builder.build_port_shutdown_scenario(['PORT_01'])")
    print("  result = cascade_engine.simulate(scenario.shock_nodes)")
    print()
    print("Module ready for integration with HT-HGNN v2.0.")
