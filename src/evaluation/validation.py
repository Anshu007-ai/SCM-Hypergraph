"""
STEP 7: Validation & Ablation Testing
Prove that hypergraph structure matters

Tests:
1. Remove joint failure component → performance drops
2. Replace hyperedges with pairwise edges → performance degrades
3. Simulate supplier failure cascades
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Tuple, List
import pickle
from pathlib import Path


class AblationTester:
    """
    Perform ablation tests to validate hypergraph importance
    """
    
    def __init__(self, features_df: pd.DataFrame, 
                 labels_df: pd.DataFrame, 
                 model_path: str,
                 scaler_path: str):
        """
        Initialize ablation tester
        
        Args:
            features_df: Full feature matrix
            labels_df: Labels
            model_path: Path to trained baseline model
            scaler_path: Path to feature scaler
        """
        self.features_df = features_df
        self.labels_df = labels_df
        
        # Load model and scaler
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.results = {}
    
    def get_baseline_performance(self, X_test: np.ndarray, 
                                y_test: np.ndarray) -> Dict[str, float]:
        """
        Get baseline model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with baseline metrics
        """
        y_pred = self.model.predict(X_test)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred)
        }
    
    def ablate_feature_group(self, X_test: np.ndarray, 
                           y_test: np.ndarray,
                           feature_group: List[str]) -> Dict[str, float]:
        """
        Test performance when removing a feature group
        
        Args:
            X_test: Test features
            y_test: Test labels
            feature_group: List of feature names to remove
        
        Returns:
            Dictionary with performance metrics after removal
        """
        # Find indices of features to remove
        all_features = self.features_df.columns.tolist()
        all_features = [f for f in all_features if f != 'hyperedge_id']
        
        remove_indices = [i for i, f in enumerate(all_features) if f in feature_group]
        keep_indices = [i for i, f in enumerate(all_features) if f not in feature_group]
        
        # Create ablated feature matrix
        X_ablated = X_test[:, keep_indices]
        
        # Predict (note: this may not be ideal since scaler was fit on full features)
        try:
            y_pred = self.model.predict(X_ablated)
            return {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'n_features_removed': len(remove_indices)
            }
        except Exception as e:
            return {
                'error': str(e),
                'n_features_removed': len(remove_indices)
            }
    
    def test_joint_failure_importance(self, X_test: np.ndarray,
                                     y_test: np.ndarray) -> Dict:
        """
        Test if joint failure probability is important
        
        Tests:
        1. Baseline: full model
        2. Without: remove min_reliability (key for joint failure)
        """
        print("\n" + "="*60)
        print("ABLATION TEST: Joint Failure Importance")
        print("="*60)
        
        baseline = self.get_baseline_performance(X_test, y_test)
        print(f"\nBaseline Performance:")
        print(f"  RMSE: {baseline['rmse']:.4f}")
        print(f"  R²: {baseline['r2']:.4f}")
        
        # Remove min_reliability (core to joint failure calculation)
        ablated = self.ablate_feature_group(X_test, y_test, ['min_reliability'])
        print(f"\nWithout min_reliability (joint failure component):")
        print(f"  RMSE: {ablated['rmse']:.4f}")
        print(f"  R²: {ablated['r2']:.4f}")
        print(f"  RMSE Increase: {(ablated['rmse']/baseline['rmse']-1)*100:.2f}%")
        print(f"  R² Drop: {baseline['r2'] - ablated['r2']:.4f}")
        
        return {
            'baseline': baseline,
            'without_joint_failure': ablated,
            'rmse_impact': (ablated['rmse'] - baseline['rmse']) / baseline['rmse'] * 100
        }
    
    def test_hyperedge_structure_importance(self, X_test: np.ndarray,
                                           y_test: np.ndarray) -> Dict:
        """
        Test if hyperedge-specific features matter
        
        Hyperedge features:
        - bom_weight
        - tolerance
        - critical_path
        - downstream_degree
        """
        print("\n" + "="*60)
        print("ABLATION TEST: Hyperedge Structure Importance")
        print("="*60)
        
        baseline = self.get_baseline_performance(X_test, y_test)
        print(f"\nBaseline Performance:")
        print(f"  RMSE: {baseline['rmse']:.4f}")
        print(f"  R²: {baseline['r2']:.4f}")
        
        # Remove hyperedge-specific features
        hyperedge_features = ['bom_weight', 'tolerance', 'critical_path',
                            'downstream_degree', 'upstream_degree']
        ablated = self.ablate_feature_group(X_test, y_test, hyperedge_features)
        
        print(f"\nWithout hyperedge structure features:")
        print(f"  RMSE: {ablated['rmse']:.4f}")
        print(f"  R²: {ablated['r2']:.4f}")
        print(f"  RMSE Increase: {(ablated['rmse']/baseline['rmse']-1)*100:.2f}%")
        print(f"  R² Drop: {baseline['r2'] - ablated['r2']:.4f}")
        
        return {
            'baseline': baseline,
            'without_hyperedge_structure': ablated,
            'rmse_impact': (ablated['rmse'] - baseline['rmse']) / baseline['rmse'] * 100
        }
    
    def test_supplier_aggregation_importance(self, X_test: np.ndarray,
                                            y_test: np.ndarray) -> Dict:
        """
        Test if supplier-level aggregation matters
        
        Supplier aggregation features:
        - mean_reliability
        - min_reliability
        - max_lead_time
        - mean_substitutability
        - supplier_count
        """
        print("\n" + "="*60)
        print("ABLATION TEST: Supplier Aggregation Importance")
        print("="*60)
        
        baseline = self.get_baseline_performance(X_test, y_test)
        print(f"\nBaseline Performance:")
        print(f"  RMSE: {baseline['rmse']:.4f}")
        print(f"  R²: {baseline['r2']:.4f}")
        
        supplier_features = ['mean_reliability', 'min_reliability', 'max_reliability',
                           'mean_lead_time', 'max_lead_time', 'mean_substitutability',
                           'supplier_count', 'mean_cost', 'total_cost']
        ablated = self.ablate_feature_group(X_test, y_test, supplier_features)
        
        print(f"\nWithout supplier aggregation features:")
        print(f"  RMSE: {ablated['rmse']:.4f}")
        print(f"  R²: {ablated['r2']:.4f}")
        print(f"  RMSE Increase: {(ablated['rmse']/baseline['rmse']-1)*100:.2f}%")
        print(f"  R² Drop: {baseline['r2'] - ablated['r2']:.4f}")
        
        return {
            'baseline': baseline,
            'without_supplier_aggregation': ablated,
            'rmse_impact': (ablated['rmse'] - baseline['rmse']) / baseline['rmse'] * 100
        }


class FailureSimulator:
    """
    Simulate supplier failures and measure cascade impact
    """
    
    def __init__(self, hypergraph, features_df: pd.DataFrame,
                 labels_df: pd.DataFrame, model):
        """
        Initialize failure simulator
        
        Args:
            hypergraph: Hypergraph object
            features_df: Feature matrix
            labels_df: Labels
            model: Trained model for prediction
        """
        self.hypergraph = hypergraph
        self.features_df = features_df
        self.labels_df = labels_df
        self.model = model
    
    def simulate_supplier_failure(self, supplier_id: str, 
                                 failure_severity: float = 1.0) -> Dict:
        """
        Simulate a supplier failure and measure cascading impact
        
        Args:
            supplier_id: Supplier to fail
            failure_severity: How severely affected (0-1)
        
        Returns:
            Dictionary with cascade analysis
        """
        if supplier_id not in self.hypergraph.nodes:
            return {'error': f'Supplier {supplier_id} not found'}
        
        # Find all hyperedges that depend on this supplier
        affected_hyperedges = self.hypergraph.node_to_hyperedges.get(supplier_id, set())
        
        # For each affected hyperedge, find downstream impacts
        cascade = {}
        for hyperedge_id in affected_hyperedges:
            downstream = self.hypergraph.get_downstream_hyperedges(hyperedge_id)
            cascade[hyperedge_id] = {
                'direct_impact': True,
                'n_downstream': len(downstream),
                'downstream_ids': list(downstream)
            }
        
        return {
            'failed_supplier': supplier_id,
            'directly_affected_hyperedges': list(affected_hyperedges),
            'n_affected': len(affected_hyperedges),
            'cascade_details': cascade
        }
    
    def run_stress_test(self, n_failures: int = 10) -> Dict:
        """
        Run stress test: simulate multiple failures and measure impacts
        
        Args:
            n_failures: Number of random failures to simulate
        
        Returns:
            Summary of impacts
        """
        print("\n" + "="*60)
        print(f"STRESS TEST: Simulating {n_failures} Supplier Failures")
        print("="*60)
        
        supplier_ids = list(self.hypergraph.nodes.keys())
        failed_suppliers = np.random.choice(supplier_ids, size=min(n_failures, len(supplier_ids)), 
                                           replace=False)
        
        total_cascade = 0
        impacts = []
        
        for supplier_id in failed_suppliers:
            result = self.simulate_supplier_failure(supplier_id)
            if 'error' not in result:
                total_cascade += result['n_affected']
                impacts.append(result['n_affected'])
                print(f"\n  Supplier {supplier_id}:")
                print(f"    Directly affects {result['n_affected']} hyperedges")
                if result['cascade_details']:
                    max_downstream = max(len(v['downstream_ids']) 
                                       for v in result['cascade_details'].values())
                    print(f"    Max cascade depth: {max_downstream}")
        
        print(f"\n  Total hyperedges at risk: {total_cascade}")
        print(f"  Average impact: {np.mean(impacts):.2f} hyperedges")
        
        return {
            'n_failures_tested': len(failed_suppliers),
            'total_hyperedges_at_risk': total_cascade,
            'avg_cascade_size': np.mean(impacts) if impacts else 0,
            'max_cascade_size': max(impacts) if impacts else 0,
            'vulnerability_rate': total_cascade / len(self.hypergraph.hyperedges)
        }


if __name__ == "__main__":
    print("Validation module ready")
