"""
STEP 2: Hypergraph Data Structure
Build and manage the core hypergraph representation

A hypergraph H = (V, E) where:
- V = nodes (suppliers/components)
- E = hyperedges (subassemblies) = sets of nodes
- Incidence relationship: which suppliers feed into which subassemblies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class HypergraphNode:
    """Node features (supplier/component)"""
    node_id: str
    node_type: str
    tier: int
    lead_time: float
    reliability: float
    substitutability: float
    cost: float
    raw_data_idx: int = 0


@dataclass
class HypergraphEdge:
    """Hyperedge features (subassembly)"""
    hyperedge_id: str
    nodes: Set[str] = field(default_factory=set)
    bom_weight: float = 0.0
    tolerance: float = 0.0
    critical_path: float = 0.0
    tier_level: int = 0


class Hypergraph:
    """
    Hypergraph representation and operations
    
    Core operations:
    - Build from incidence matrix
    - Query relationships
    - Compute graph properties
    - Message passing preparation
    """
    
    def __init__(self):
        self.nodes: Dict[str, HypergraphNode] = {}
        self.hyperedges: Dict[str, HypergraphEdge] = {}
        self.incidence: Dict[str, Set[str]] = {}  # hyperedge_id -> set of node_ids
        self.node_to_hyperedges: Dict[str, Set[str]] = {}  # node_id -> set of hyperedge_ids
        self.echelon_deps: Dict[str, List[str]] = {}  # parent -> [children]
        
    def add_node(self, node: HypergraphNode):
        """Add a node to the hypergraph"""
        self.nodes[node.node_id] = node
        if node.node_id not in self.node_to_hyperedges:
            self.node_to_hyperedges[node.node_id] = set()
    
    def add_hyperedge(self, hyperedge: HypergraphEdge):
        """Add a hyperedge to the hypergraph"""
        self.hyperedges[hyperedge.hyperedge_id] = hyperedge
        self.incidence[hyperedge.hyperedge_id] = hyperedge.nodes.copy()
        
        # Update node-to-hyperedges mapping
        for node_id in hyperedge.nodes:
            if node_id not in self.node_to_hyperedges:
                self.node_to_hyperedges[node_id] = set()
            self.node_to_hyperedges[node_id].add(hyperedge.hyperedge_id)
    
    def add_echelon_dependency(self, parent_id: str, child_id: str):
        """Add multi-echelon dependency: parent depends on child"""
        if parent_id not in self.echelon_deps:
            self.echelon_deps[parent_id] = []
        self.echelon_deps[parent_id].append(child_id)
    
    @staticmethod
    def from_dataframes(nodes_df: pd.DataFrame,
                       hyperedges_df: pd.DataFrame,
                       incidence_df: pd.DataFrame,
                       echelon_df: pd.DataFrame = None) -> 'Hypergraph':
        """
        Build hypergraph from DataFrames
        
        Args:
            nodes_df: Node features
            hyperedges_df: Hyperedge features
            incidence_df: Incidence relationships (hyperedge_id, node_id)
            echelon_df: Multi-echelon dependencies (parent, child)
        
        Returns:
            Constructed Hypergraph object
        """
        hg = Hypergraph()
        
        # Add nodes
        for _, row in nodes_df.iterrows():
            node = HypergraphNode(
                node_id=row['node_id'],
                node_type=row['node_type'],
                tier=row['tier'],
                lead_time=row['lead_time'],
                reliability=row['reliability'],
                substitutability=row['substitutability'],
                cost=row['cost']
            )
            hg.add_node(node)
        
        # Build hyperedges with incidence information
        hyperedge_nodes = {}
        for _, row in incidence_df.iterrows():
            hid = row['hyperedge_id']
            nid = row['node_id']
            if hid not in hyperedge_nodes:
                hyperedge_nodes[hid] = set()
            hyperedge_nodes[hid].add(nid)
        
        # Add hyperedges with their properties
        for _, row in hyperedges_df.iterrows():
            hid = row['hyperedge_id']
            hyperedge = HypergraphEdge(
                hyperedge_id=hid,
                nodes=hyperedge_nodes.get(hid, set()),
                bom_weight=row['bom_weight'],
                tolerance=row['tolerance'],
                critical_path=row['critical_path'],
                tier_level=row['tier_level']
            )
            hg.add_hyperedge(hyperedge)
        
        # Add echelon dependencies if provided
        if echelon_df is not None and len(echelon_df) > 0:
            for _, row in echelon_df.iterrows():
                hg.add_echelon_dependency(
                    row['parent_hyperedge'],
                    row['child_hyperedge']
                )
        
        return hg
    
    def get_downstream_hyperedges(self, hyperedge_id: str, 
                                  depth: int = None) -> Set[str]:
        """
        Get all downstream hyperedges in the supply chain
        
        Args:
            hyperedge_id: Starting hyperedge
            depth: Maximum search depth (None = unlimited)
        
        Returns:
            Set of downstream hyperedge IDs
        """
        if depth == 0:
            return set()
        
        downstream = set()
        queue = [(hyperedge_id, 0)]
        
        while queue:
            current, current_depth = queue.pop(0)
            
            # Find parents of current (hyperedges that depend on current)
            for parent, children in self.echelon_deps.items():
                if current in children and parent not in downstream:
                    downstream.add(parent)
                    if depth is None or current_depth + 1 < depth:
                        queue.append((parent, current_depth + 1))
        
        return downstream
    
    def get_upstream_hyperedges(self, hyperedge_id: str,
                               depth: int = None) -> Set[str]:
        """
        Get all upstream hyperedges in the supply chain
        
        Args:
            hyperedge_id: Starting hyperedge
            depth: Maximum search depth (None = unlimited)
        
        Returns:
            Set of upstream hyperedge IDs
        """
        if depth == 0:
            return set()
        
        upstream = set()
        queue = [(hyperedge_id, 0)]
        
        while queue:
            current, current_depth = queue.pop(0)
            
            if current in self.echelon_deps:
                for child in self.echelon_deps[current]:
                    if child not in upstream:
                        upstream.add(child)
                        if depth is None or current_depth + 1 < depth:
                            queue.append((child, current_depth + 1))
        
        return upstream
    
    def get_incidence_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Get incidence matrix H where H[e,v] = 1 if node v in hyperedge e
        
        Returns:
            Tuple of:
            - incidence_matrix: (n_hyperedges, n_nodes) binary matrix
            - hyperedge_ids: list of hyperedge IDs (row order)
            - node_ids: list of node IDs (column order)
        """
        hyperedge_ids = sorted(list(self.hyperedges.keys()))
        node_ids = sorted(list(self.nodes.keys()))
        
        n_he = len(hyperedge_ids)
        n_nodes = len(node_ids)
        
        H = np.zeros((n_he, n_nodes), dtype=np.int32)
        
        he_to_idx = {hid: i for i, hid in enumerate(hyperedge_ids)}
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        for hid, nodes in self.incidence.items():
            he_idx = he_to_idx[hid]
            for node_id in nodes:
                if node_id in node_to_idx:
                    node_idx = node_to_idx[node_id]
                    H[he_idx, node_idx] = 1
        
        return H, hyperedge_ids, node_ids
    
    def get_statistics(self) -> Dict:
        """Get basic hypergraph statistics"""
        
        # Node statistics
        node_degrees = [len(self.node_to_hyperedges[nid]) 
                       for nid in self.nodes.keys()]
        
        # Hyperedge sizes
        hyperedge_sizes = [len(self.incidence[hid]) 
                          for hid in self.hyperedges.keys()]
        
        # Dependency depth
        max_depth = 0
        for parent, children in self.echelon_deps.items():
            for child in children:
                depth = len(self.get_upstream_hyperedges(child))
                max_depth = max(max_depth, depth)
        
        return {
            'n_nodes': len(self.nodes),
            'n_hyperedges': len(self.hyperedges),
            'n_edges_in_incidence': sum(len(nodes) for nodes in self.incidence.values()),
            'avg_node_degree': np.mean(node_degrees) if node_degrees else 0,
            'min_node_degree': min(node_degrees) if node_degrees else 0,
            'max_node_degree': max(node_degrees) if node_degrees else 0,
            'avg_hyperedge_size': np.mean(hyperedge_sizes) if hyperedge_sizes else 0,
            'min_hyperedge_size': min(hyperedge_sizes) if hyperedge_sizes else 0,
            'max_hyperedge_size': max(hyperedge_sizes) if hyperedge_sizes else 0,
            'n_echelon_dependencies': len(sum(self.echelon_deps.values(), [])),
            'max_echelon_depth': max_depth
        }
    
    def to_dict(self) -> Dict:
        """Convert hypergraph to dictionary format for serialization"""
        return {
            'nodes': {nid: {
                'node_type': n.node_type,
                'tier': n.tier,
                'lead_time': n.lead_time,
                'reliability': float(n.reliability),
                'substitutability': float(n.substitutability),
                'cost': float(n.cost)
            } for nid, n in self.nodes.items()},
            'hyperedges': {hid: {
                'nodes': list(he.nodes),
                'bom_weight': float(he.bom_weight),
                'tolerance': float(he.tolerance),
                'critical_path': float(he.critical_path),
                'tier_level': he.tier_level
            } for hid, he in self.hyperedges.items()},
            'echelon_dependencies': {k: v for k, v in self.echelon_deps.items()}
        }


if __name__ == "__main__":
    # Example usage
    print("Hypergraph module ready for integration")
