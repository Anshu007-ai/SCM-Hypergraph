"""
D3-Ready Hypergraph Serialization

Converts the internal Hypergraph representation into JSON formats consumed
by popular browser-based graph visualisation libraries:
  - D3.js (force-directed)
  - Cytoscape.js
  - vis-network

Also provides helpers for risk-based node colouring and convex-hull
computation for rendering hyperedge regions.
"""

import json
import math
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple

from src.hypergraph.hypergraph import Hypergraph


# -----------------------------------------------------------------------
# Colour helpers
# -----------------------------------------------------------------------

_RISK_PALETTE = {
    "Critical": "#e74c3c",
    "High":     "#e67e22",
    "Medium":   "#f1c40f",
    "Low":      "#2ecc71",
    "Minimal":  "#3498db",
    "Unknown":  "#95a5a6",
}


def _risk_level_from_score(score: float) -> str:
    """Map a continuous risk score in [0, 1] to a categorical risk level."""
    if score >= 0.8:
        return "Critical"
    elif score >= 0.6:
        return "High"
    elif score >= 0.4:
        return "Medium"
    elif score >= 0.2:
        return "Low"
    else:
        return "Minimal"


# -----------------------------------------------------------------------
# Main serializer
# -----------------------------------------------------------------------

class HypergraphSerializer:
    """
    Serialize a :class:`Hypergraph` into JSON structures ready for use in
    browser-based graph visualisation libraries.

    Parameters
    ----------
    hypergraph : Hypergraph
        The hypergraph instance to serialize.
    """

    def __init__(self, hypergraph: Hypergraph):
        self.hg = hypergraph

    # ------------------------------------------------------------------
    # D3.js
    # ------------------------------------------------------------------

    def to_d3_json(self) -> Dict[str, Any]:
        """
        Produce a JSON-serializable dictionary for a D3.js force-directed layout.

        The output schema is::

            {
                "nodes": [
                    {"id": "...", "type": "...", "tier": 1, ...}, ...
                ],
                "links": [
                    {"source": "node_id", "target": "he_id", "value": 1}, ...
                ],
                "hyperedges": [
                    {"id": "...", "size": 4, "bom_weight": 0.5, ...}, ...
                ]
            }

        Hyperedges are modelled as auxiliary *hub* nodes so that D3 can
        render them in a bipartite force layout (nodes <-> hyperedge hubs).

        Returns
        -------
        dict
        """
        nodes: List[Dict[str, Any]] = []
        links: List[Dict[str, Any]] = []
        hyperedge_entries: List[Dict[str, Any]] = []

        # Serialise real nodes
        for nid, node in self.hg.nodes.items():
            nodes.append({
                "id": nid,
                "group": "node",
                "type": node.node_type,
                "tier": node.tier,
                "lead_time": node.lead_time,
                "reliability": float(node.reliability),
                "substitutability": float(node.substitutability),
                "cost": float(node.cost),
            })

        # Serialise hyperedges as hub nodes + links
        for hid, he in self.hg.hyperedges.items():
            nodes.append({
                "id": hid,
                "group": "hyperedge",
                "size": len(he.nodes),
                "bom_weight": float(he.bom_weight),
                "tolerance": float(he.tolerance),
                "critical_path": float(he.critical_path),
                "tier_level": he.tier_level,
            })

            hyperedge_entries.append({
                "id": hid,
                "members": sorted(he.nodes),
                "size": len(he.nodes),
                "bom_weight": float(he.bom_weight),
                "tolerance": float(he.tolerance),
                "critical_path": float(he.critical_path),
                "tier_level": he.tier_level,
            })

            for member_nid in he.nodes:
                links.append({
                    "source": member_nid,
                    "target": hid,
                    "value": 1,
                })

        # Add echelon dependency links
        for parent, children in self.hg.echelon_deps.items():
            for child in children:
                links.append({
                    "source": child,
                    "target": parent,
                    "value": 2,
                    "type": "echelon",
                })

        return {
            "nodes": nodes,
            "links": links,
            "hyperedges": hyperedge_entries,
        }

    # ------------------------------------------------------------------
    # Cytoscape.js
    # ------------------------------------------------------------------

    def to_cytoscape_json(self) -> Dict[str, Any]:
        """
        Produce a JSON-serializable dictionary for Cytoscape.js.

        Follows the ``cytoscape.js`` JSON graph format::

            {
                "elements": {
                    "nodes": [ {"data": {"id": ..., ...}}, ... ],
                    "edges": [ {"data": {"id": ..., "source": ..., "target": ...}}, ... ]
                }
            }

        Each hyperedge is represented as a compound (parent) node whose
        children are its member nodes, enabling Cytoscape's built-in
        compound-node rendering.

        Returns
        -------
        dict
        """
        cy_nodes: List[Dict[str, Any]] = []
        cy_edges: List[Dict[str, Any]] = []

        # Hyperedge compound nodes
        for hid, he in self.hg.hyperedges.items():
            cy_nodes.append({
                "data": {
                    "id": hid,
                    "label": hid,
                    "type": "hyperedge",
                    "size": len(he.nodes),
                    "bom_weight": float(he.bom_weight),
                    "tolerance": float(he.tolerance),
                    "critical_path": float(he.critical_path),
                    "tier_level": he.tier_level,
                },
                "classes": "hyperedge",
            })

        # Real nodes (parented to first hyperedge they belong to)
        for nid, node in self.hg.nodes.items():
            parent = None
            he_set = self.hg.node_to_hyperedges.get(nid, set())
            if he_set:
                parent = sorted(he_set)[0]

            data: Dict[str, Any] = {
                "id": nid,
                "label": nid,
                "type": node.node_type,
                "tier": node.tier,
                "lead_time": node.lead_time,
                "reliability": float(node.reliability),
                "substitutability": float(node.substitutability),
                "cost": float(node.cost),
            }
            if parent:
                data["parent"] = parent

            cy_nodes.append({"data": data, "classes": "supplier"})

        # Edges: node membership in hyperedges
        edge_counter = 0
        for hid, he in self.hg.hyperedges.items():
            for member_nid in he.nodes:
                cy_edges.append({
                    "data": {
                        "id": f"membership_{edge_counter}",
                        "source": member_nid,
                        "target": hid,
                        "type": "membership",
                    }
                })
                edge_counter += 1

        # Echelon dependencies
        for parent, children in self.hg.echelon_deps.items():
            for child in children:
                cy_edges.append({
                    "data": {
                        "id": f"echelon_{edge_counter}",
                        "source": child,
                        "target": parent,
                        "type": "echelon",
                    }
                })
                edge_counter += 1

        return {"elements": {"nodes": cy_nodes, "edges": cy_edges}}

    # ------------------------------------------------------------------
    # vis-network
    # ------------------------------------------------------------------

    def to_vis_network_json(self) -> Dict[str, Any]:
        """
        Produce a JSON-serializable dictionary for the **vis-network** library.

        Schema::

            {
                "nodes": [{"id": ..., "label": ..., "group": ..., ...}, ...],
                "edges": [{"from": ..., "to": ..., ...}, ...]
            }

        Returns
        -------
        dict
        """
        vis_nodes: List[Dict[str, Any]] = []
        vis_edges: List[Dict[str, Any]] = []

        for nid, node in self.hg.nodes.items():
            vis_nodes.append({
                "id": nid,
                "label": nid,
                "group": node.node_type,
                "title": (
                    f"Tier: {node.tier} | "
                    f"Reliability: {node.reliability:.2f} | "
                    f"Lead time: {node.lead_time:.1f}"
                ),
                "tier": node.tier,
                "reliability": float(node.reliability),
                "lead_time": node.lead_time,
                "cost": float(node.cost),
                "shape": "dot",
            })

        # Hyperedge hub nodes (larger, different shape)
        for hid, he in self.hg.hyperedges.items():
            vis_nodes.append({
                "id": hid,
                "label": hid,
                "group": "hyperedge",
                "title": (
                    f"Size: {len(he.nodes)} | "
                    f"BOM weight: {he.bom_weight:.2f} | "
                    f"Critical path: {he.critical_path:.2f}"
                ),
                "shape": "diamond",
                "size": 10 + len(he.nodes) * 3,
            })

            for member_nid in he.nodes:
                vis_edges.append({
                    "from": member_nid,
                    "to": hid,
                    "dashes": False,
                    "title": "membership",
                })

        # Echelon dependencies
        for parent, children in self.hg.echelon_deps.items():
            for child in children:
                vis_edges.append({
                    "from": child,
                    "to": parent,
                    "dashes": True,
                    "arrows": "to",
                    "title": "echelon dependency",
                    "color": {"color": "#e74c3c"},
                })

        return {"nodes": vis_nodes, "edges": vis_edges}

    # ------------------------------------------------------------------
    # Risk-coloured nodes
    # ------------------------------------------------------------------

    def get_risk_colored_nodes(
        self,
        risk_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Return node records augmented with colour attributes derived from
        continuous risk scores.

        Parameters
        ----------
        risk_scores : dict
            Mapping of ``node_id -> risk_score`` where risk_score is in [0, 1].

        Returns
        -------
        list of dict
            Each dict contains ``id``, ``risk_score``, ``risk_level``,
            ``color``, and all base node attributes.
        """
        coloured: List[Dict[str, Any]] = []

        for nid, node in self.hg.nodes.items():
            score = risk_scores.get(nid, 0.0)
            level = _risk_level_from_score(score)
            colour = _RISK_PALETTE.get(level, _RISK_PALETTE["Unknown"])

            coloured.append({
                "id": nid,
                "type": node.node_type,
                "tier": node.tier,
                "risk_score": float(score),
                "risk_level": level,
                "color": colour,
                "reliability": float(node.reliability),
                "lead_time": node.lead_time,
                "cost": float(node.cost),
            })

        return coloured

    # ------------------------------------------------------------------
    # Convex hulls for hyperedge rendering
    # ------------------------------------------------------------------

    def get_convex_hulls(
        self,
        hyperedge_ids: Optional[List[str]] = None,
        node_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: int = 42,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Compute convex hull polygons for selected hyperedges so they can be
        drawn as shaded regions in a 2-D visualisation.

        If ``node_positions`` are not supplied a deterministic pseudo-random
        layout is generated (useful for testing / server-side pre-computation).

        Parameters
        ----------
        hyperedge_ids : list of str or None
            Hyperedge IDs to compute hulls for.  ``None`` means all.
        node_positions : dict or None
            Mapping ``node_id -> (x, y)``.  If ``None`` a random layout is
            used.
        seed : int
            Random seed for the fallback layout.

        Returns
        -------
        dict
            Mapping ``hyperedge_id -> [(x1, y1), (x2, y2), ...]`` where the
            list of tuples is the ordered convex hull boundary.
        """
        if hyperedge_ids is None:
            hyperedge_ids = list(self.hg.hyperedges.keys())

        # Fallback positions
        if node_positions is None:
            rng = np.random.RandomState(seed)
            node_positions = {
                nid: (float(rng.uniform(-100, 100)), float(rng.uniform(-100, 100)))
                for nid in self.hg.nodes
            }

        hulls: Dict[str, List[Tuple[float, float]]] = {}

        for hid in hyperedge_ids:
            if hid not in self.hg.hyperedges:
                continue
            members = self.hg.hyperedges[hid].nodes
            points = [
                node_positions[nid]
                for nid in members
                if nid in node_positions
            ]
            if len(points) < 3:
                # Cannot form a hull; return the points themselves
                hulls[hid] = points
                continue

            hull_pts = self._convex_hull(points)
            hulls[hid] = hull_pts

        return hulls

    # ------------------------------------------------------------------
    # Utility: export helpers
    # ------------------------------------------------------------------

    def to_json_string(self, fmt: str = "d3", **kwargs) -> str:
        """
        Convenience wrapper that returns the chosen format as a JSON string.

        Parameters
        ----------
        fmt : str
            One of ``'d3'``, ``'cytoscape'``, ``'vis'``.

        Returns
        -------
        str
        """
        exporters = {
            "d3": self.to_d3_json,
            "cytoscape": self.to_cytoscape_json,
            "vis": self.to_vis_network_json,
        }
        if fmt not in exporters:
            raise ValueError(f"Unknown format '{fmt}'. Choose from {list(exporters)}")
        return json.dumps(exporters[fmt](), indent=2, default=str)

    # ------------------------------------------------------------------
    # Private: convex hull (Graham scan)
    # ------------------------------------------------------------------

    @staticmethod
    def _convex_hull(
        points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Compute the 2-D convex hull of a set of points using the Graham
        scan algorithm.  Returns the hull vertices in counter-clockwise
        order.
        """
        pts = sorted(set(points))
        if len(pts) <= 2:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Build lower hull
        lower: List[Tuple[float, float]] = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper: List[Tuple[float, float]] = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenation; last point of each half is omitted (duplicate)
        return lower[:-1] + upper[:-1]


if __name__ == "__main__":
    from src.hypergraph.hypergraph import Hypergraph, HypergraphNode, HypergraphEdge

    # Build a tiny demo hypergraph
    hg = Hypergraph()
    for i in range(6):
        hg.add_node(HypergraphNode(
            node_id=f"s{i}",
            node_type="supplier",
            tier=i % 3,
            lead_time=float(5 + i),
            reliability=0.85 + 0.02 * i,
            substitutability=0.5,
            cost=100.0 + i * 10,
        ))

    hg.add_hyperedge(HypergraphEdge(
        hyperedge_id="he_0", nodes={"s0", "s1", "s2"}, bom_weight=0.6,
    ))
    hg.add_hyperedge(HypergraphEdge(
        hyperedge_id="he_1", nodes={"s3", "s4", "s5"}, bom_weight=0.4,
    ))
    hg.add_echelon_dependency("he_1", "he_0")

    serializer = HypergraphSerializer(hg)

    print("--- D3 JSON (truncated) ---")
    d3 = serializer.to_d3_json()
    print(f"  nodes: {len(d3['nodes'])}, links: {len(d3['links'])}, hyperedges: {len(d3['hyperedges'])}")

    print("\n--- Cytoscape JSON (truncated) ---")
    cy = serializer.to_cytoscape_json()
    print(f"  nodes: {len(cy['elements']['nodes'])}, edges: {len(cy['elements']['edges'])}")

    print("\n--- vis-network JSON (truncated) ---")
    vis = serializer.to_vis_network_json()
    print(f"  nodes: {len(vis['nodes'])}, edges: {len(vis['edges'])}")

    print("\n--- Risk-coloured nodes ---")
    scores = {f"s{i}": 0.15 * i for i in range(6)}
    coloured = serializer.get_risk_colored_nodes(scores)
    for c in coloured:
        print(f"  {c['id']}: {c['risk_level']} ({c['color']})")

    print("\n--- Convex hulls ---")
    hulls = serializer.get_convex_hulls()
    for hid, pts in hulls.items():
        print(f"  {hid}: {len(pts)} hull vertices")

    print("\nVisualization utils module ready for integration")
