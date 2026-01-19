"""
STEP 3: Risk Signal Generation
Compute ground truth labels for supervised learning

HCI (Hypergraph Critical Index) = α*P(joint_failure) + β*engineering_impact + γ*propagation_risk
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from src.hypergraph.hypergraph import Hypergraph


class RiskLabelGenerator:
    """Generate ground truth risk labels for hyperedges"""
    
    def __init__(self, hypergraph: Hypergraph,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 gamma: float = 0.2):
        """
        Initialize risk label generator
        
        Args:
            hypergraph: Constructed hypergraph object
            alpha: Weight for joint failure probability
            beta: Weight for engineering impact
            gamma: Weight for propagation risk
        """
        self.hg = hypergraph
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Validate weights
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    def compute_joint_failure_probability(self, hyperedge_id: str) -> float:
        """
        Compute P(joint failure) for a hyperedge
        
        Joint failure occurs when ANY supplier in the subassembly fails.
        Using logical AND constraint (all suppliers needed):
        
        P(joint_failure) = 1 - P(all_survive)
        P(all_survive) = ∏ P(supplier_i survives)
        
        Args:
            hyperedge_id: The hyperedge to compute for
        
        Returns:
            Joint failure probability [0, 1]
        """
        if hyperedge_id not in self.hg.hyperedges:
            return 0.0
        
        hyperedge = self.hg.hyperedges[hyperedge_id]
        nodes = hyperedge.nodes
        
        if not nodes:
            return 0.0
        
        # P(all survive) = product of individual reliabilities
        p_survive = 1.0
        for node_id in nodes:
            if node_id in self.hg.nodes:
                reliability = self.hg.nodes[node_id].reliability
                p_survive *= reliability
        
        # P(at least one fails) = 1 - P(all survive)
        joint_failure_prob = 1.0 - p_survive
        
        return joint_failure_prob
    
    def compute_engineering_impact(self, hyperedge_id: str) -> float:
        """
        Compute engineering impact score
        
        Combines:
        - BOM weight (importance in bill of materials)
        - Tolerance sensitivity (manufacturing precision needed)
        - Critical path (is on critical path to final assembly)
        
        Args:
            hyperedge_id: The hyperedge to compute for
        
        Returns:
            Engineering impact score [0, 1]
        """
        if hyperedge_id not in self.hg.hyperedges:
            return 0.0
        
        he = self.hg.hyperedges[hyperedge_id]
        
        # Geometric mean to balance factors
        impact = (he.bom_weight * he.tolerance * (1 + he.critical_path)) ** (1/3)
        
        return float(np.clip(impact, 0, 1))
    
    def compute_propagation_risk(self, hyperedge_id: str) -> float:
        """
        Compute propagation risk through supply chain
        
        Higher risk if:
        - Many downstream assemblies depend on this
        - Failure cascades through multiple tiers
        
        Args:
            hyperedge_id: The hyperedge to compute for
        
        Returns:
            Propagation risk score [0, 1]
        """
        # Get downstream hyperedges
        downstream = self.hg.get_downstream_hyperedges(hyperedge_id)
        
        # Normalize by total hyperedges
        if len(self.hg.hyperedges) == 0:
            return 0.0
        
        propagation = len(downstream) / len(self.hg.hyperedges)
        
        return float(np.clip(propagation, 0, 1))
    
    def compute_supplier_concentration_risk(self, hyperedge_id: str) -> float:
        """
        Compute risk from supplier concentration (single points of failure)
        
        Higher risk if:
        - Few suppliers
        - Low substitutability
        - Low lead time slack
        
        Args:
            hyperedge_id: The hyperedge to compute for
        
        Returns:
            Concentration risk score [0, 1]
        """
        if hyperedge_id not in self.hg.hyperedges:
            return 0.0
        
        hyperedge = self.hg.hyperedges[hyperedge_id]
        nodes = hyperedge.nodes
        
        if not nodes:
            return 0.0
        
        # Low supplier count = high risk
        avg_suppliers = len(nodes) / 6.0  # Normalize by max typical
        supplier_concentration = 1.0 - min(avg_suppliers, 1.0)
        
        # Low substitutability = high risk
        substitutabilities = [self.hg.nodes[nid].substitutability 
                            for nid in nodes if nid in self.hg.nodes]
        avg_substitutability = np.mean(substitutabilities) if substitutabilities else 0.5
        substitution_risk = 1.0 - avg_substitutability
        
        # High lead time = supply chain risk
        lead_times = [self.hg.nodes[nid].lead_time 
                     for nid in nodes if nid in self.hg.nodes]
        avg_lead_time = np.mean(lead_times) if lead_times else 10
        lead_time_risk = min(avg_lead_time / 30.0, 1.0)  # Normalize by 30 days
        
        concentration = (supplier_concentration + 
                        0.5 * substitution_risk + 
                        0.3 * lead_time_risk) / 2.3
        
        return float(np.clip(concentration, 0, 1))
    
    def compute_hci_label(self, hyperedge_id: str) -> Dict[str, float]:
        """
        Compute complete HCI label with component breakdown
        
        HCI = α*P(joint_failure) + β*engineering_impact + γ*propagation_risk
        
        Args:
            hyperedge_id: The hyperedge to compute for
        
        Returns:
            Dictionary with HCI and component scores
        """
        joint_failure = self.compute_joint_failure_probability(hyperedge_id)
        engineering = self.compute_engineering_impact(hyperedge_id)
        propagation = self.compute_propagation_risk(hyperedge_id)
        concentration = self.compute_supplier_concentration_risk(hyperedge_id)
        
        # Weighted combination
        hci = (self.alpha * joint_failure + 
               self.beta * engineering + 
               self.gamma * propagation)
        
        # Add concentration risk (weighted separately)
        hci = hci * 0.7 + concentration * 0.3
        
        return {
            'hyperedge_id': hyperedge_id,
            'HCI': float(np.clip(hci, 0, 1)),
            'joint_failure_prob': float(joint_failure),
            'engineering_impact': float(engineering),
            'propagation_risk': float(propagation),
            'concentration_risk': float(concentration),
            'risk_level': self._assign_risk_level(hci)
        }
    
    @staticmethod
    def _assign_risk_level(hci_score: float) -> str:
        """Map HCI score to risk level"""
        if hci_score >= 0.8:
            return "Critical"
        elif hci_score >= 0.6:
            return "High"
        elif hci_score >= 0.4:
            return "Medium"
        elif hci_score >= 0.2:
            return "Low"
        else:
            return "Minimal"
    
    def generate_all_labels(self) -> pd.DataFrame:
        """
        Generate HCI labels for all hyperedges
        
        Returns:
            DataFrame with labels for all hyperedges
        """
        labels_list = []
        
        for hyperedge_id in self.hg.hyperedges.keys():
            label = self.compute_hci_label(hyperedge_id)
            labels_list.append(label)
        
        return pd.DataFrame(labels_list)
    
    def get_risk_summary(self) -> Dict:
        """Get summary statistics of risk distribution"""
        labels_df = self.generate_all_labels()
        
        return {
            'total_hyperedges': len(labels_df),
            'mean_hci': labels_df['HCI'].mean(),
            'std_hci': labels_df['HCI'].std(),
            'min_hci': labels_df['HCI'].min(),
            'max_hci': labels_df['HCI'].max(),
            'critical_count': (labels_df['risk_level'] == 'Critical').sum(),
            'high_count': (labels_df['risk_level'] == 'High').sum(),
            'medium_count': (labels_df['risk_level'] == 'Medium').sum(),
            'low_count': (labels_df['risk_level'] == 'Low').sum(),
            'minimal_count': (labels_df['risk_level'] == 'Minimal').sum(),
        }


class FeatureAggregator:
    """Aggregate node features to hyperedge level for ML"""
    
    def __init__(self, hypergraph: Hypergraph):
        self.hg = hypergraph
    
    def aggregate_node_features(self, hyperedge_id: str) -> Dict[str, float]:
        """
        Aggregate node-level features to hyperedge level
        
        For hyperedge, compute:
        - Mean/min/max reliability
        - Mean/max lead time
        - Mean substitutability
        - Supplier count
        - Cost metrics
        
        Args:
            hyperedge_id: Target hyperedge
        
        Returns:
            Dictionary of aggregated features
        """
        if hyperedge_id not in self.hg.hyperedges:
            return {}
        
        hyperedge = self.hg.hyperedges[hyperedge_id]
        nodes = hyperedge.nodes
        
        if not nodes:
            return self._get_empty_features()
        
        # Extract node features
        node_objs = [self.hg.nodes[nid] for nid in nodes if nid in self.hg.nodes]
        
        if not node_objs:
            return self._get_empty_features()
        
        reliabilities = [n.reliability for n in node_objs]
        lead_times = [n.lead_time for n in node_objs]
        substitutabilities = [n.substitutability for n in node_objs]
        costs = [n.cost for n in node_objs]
        
        features = {
            'mean_reliability': np.mean(reliabilities),
            'min_reliability': np.min(reliabilities),
            'max_reliability': np.max(reliabilities),
            'std_reliability': np.std(reliabilities),
            'mean_lead_time': np.mean(lead_times),
            'max_lead_time': np.max(lead_times),
            'min_lead_time': np.min(lead_times),
            'mean_substitutability': np.mean(substitutabilities),
            'min_substitutability': np.min(substitutabilities),
            'supplier_count': len(nodes),
            'mean_cost': np.mean(costs),
            'total_cost': np.sum(costs),
        }
        
        return features
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return zero-valued feature dict"""
        return {
            'mean_reliability': 0.0,
            'min_reliability': 0.0,
            'max_reliability': 0.0,
            'std_reliability': 0.0,
            'mean_lead_time': 0.0,
            'max_lead_time': 0.0,
            'min_lead_time': 0.0,
            'mean_substitutability': 0.0,
            'min_substitutability': 0.0,
            'supplier_count': 0,
            'mean_cost': 0.0,
            'total_cost': 0.0,
        }
    
    def aggregate_all_features(self) -> pd.DataFrame:
        """
        Aggregate features for all hyperedges
        
        Returns:
            DataFrame with aggregated node features per hyperedge
        """
        features_list = []
        
        for hyperedge_id in self.hg.hyperedges.keys():
            features = self.aggregate_node_features(hyperedge_id)
            features['hyperedge_id'] = hyperedge_id
            
            # Add hyperedge-level features
            he = self.hg.hyperedges[hyperedge_id]
            features['bom_weight'] = he.bom_weight
            features['tolerance'] = he.tolerance
            features['critical_path'] = he.critical_path
            features['tier_level'] = he.tier_level
            
            # Add structural features
            features['downstream_degree'] = len(self.hg.get_downstream_hyperedges(hyperedge_id))
            features['upstream_degree'] = len(self.hg.get_upstream_hyperedges(hyperedge_id))
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)


if __name__ == "__main__":
    print("Risk label generation module ready")
