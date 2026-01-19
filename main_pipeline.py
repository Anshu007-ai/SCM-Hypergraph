"""
MAIN PIPELINE: End-to-end supply chain hypergraph risk modeling

Steps:
1. Generate/load data
2. Build hypergraph
3. Compute risk labels
4. Train baseline ML models
5. Evaluate and save results
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from data.data_generator import SupplyChainDataGenerator, load_real_datasets
from hypergraph.hypergraph import Hypergraph
from hypergraph.risk_labels import RiskLabelGenerator, FeatureAggregator
from models.baseline_models import BaselineModelTrainer


def main():
    """Execute complete pipeline"""
    
    print("\n" + "="*70)
    print("SUPPLY CHAIN HYPERGRAPH RISK MODELING SYSTEM")
    print("="*70)
    
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== STEP 1: GENERATE DATA ==========
    print("\n[STEP 1] GENERATING SYNTHETIC DATASET")
    print("-" * 70)
    
    generator = SupplyChainDataGenerator(
        n_suppliers=150,
        n_assemblies=80,
        seed=42
    )
    
    synthetic_data = generator.generate_all()
    nodes_df = synthetic_data['nodes']
    hyperedges_df = synthetic_data['hyperedges']
    incidence_df = synthetic_data['incidence']
    echelon_df = synthetic_data['echelon_dependencies']
    
    # Save generated data
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    nodes_df.to_csv(datasets_dir / "nodes.csv", index=False)
    hyperedges_df.to_csv(datasets_dir / "hyperedges.csv", index=False)
    incidence_df.to_csv(datasets_dir / "incidence.csv", index=False)
    echelon_df.to_csv(datasets_dir / "echelon_dependencies.csv", index=False)
    
    print(f"\n✓ Datasets saved to {datasets_dir}")
    
    # ========== STEP 2: BUILD HYPERGRAPH ==========
    print("\n[STEP 2] BUILDING HYPERGRAPH DATA STRUCTURE")
    print("-" * 70)
    
    hypergraph = Hypergraph.from_dataframes(
        nodes_df=nodes_df,
        hyperedges_df=hyperedges_df,
        incidence_df=incidence_df,
        echelon_df=echelon_df
    )
    
    stats = hypergraph.get_statistics()
    print(f"\nHypergraph Statistics:")
    print(f"  Nodes (suppliers/components): {stats['n_nodes']}")
    print(f"  Hyperedges (subassemblies): {stats['n_hyperedges']}")
    print(f"  Incidence relationships: {stats['n_edges_in_incidence']}")
    print(f"  Echelon dependencies: {stats['n_echelon_dependencies']}")
    print(f"  Avg hyperedge size: {stats['avg_hyperedge_size']:.2f} suppliers")
    print(f"  Avg node degree: {stats['avg_node_degree']:.2f} hyperedges")
    print(f"  Max echelon depth: {stats['max_echelon_depth']}")
    
    # Save hypergraph
    hg_dict = hypergraph.to_dict()
    with open(output_dir / "hypergraph.json", 'w') as f:
        json.dump(hg_dict, f, indent=2)
    
    # ========== STEP 3: GENERATE RISK LABELS ==========
    print("\n[STEP 3] COMPUTING HYPERGRAPH CRITICAL INDEX (HCI) LABELS")
    print("-" * 70)
    
    risk_gen = RiskLabelGenerator(
        hypergraph=hypergraph,
        alpha=0.5,    # Joint failure weight
        beta=0.3,     # Engineering impact weight
        gamma=0.2     # Propagation risk weight
    )
    
    labels_df = risk_gen.generate_all_labels()
    risk_summary = risk_gen.get_risk_summary()
    
    print(f"\nRisk Distribution:")
    print(f"  Mean HCI: {risk_summary['mean_hci']:.4f}")
    print(f"  Std HCI: {risk_summary['std_hci']:.4f}")
    print(f"  Critical: {risk_summary['critical_count']} ({risk_summary['critical_count']/len(labels_df)*100:.1f}%)")
    print(f"  High: {risk_summary['high_count']} ({risk_summary['high_count']/len(labels_df)*100:.1f}%)")
    print(f"  Medium: {risk_summary['medium_count']} ({risk_summary['medium_count']/len(labels_df)*100:.1f}%)")
    print(f"  Low: {risk_summary['low_count']} ({risk_summary['low_count']/len(labels_df)*100:.1f}%)")
    print(f"  Minimal: {risk_summary['minimal_count']} ({risk_summary['minimal_count']/len(labels_df)*100:.1f}%)")
    
    # Save labels
    labels_df.to_csv(datasets_dir / "hci_labels.csv", index=False)
    
    # Convert risk summary to JSON-serializable format
    risk_summary_serializable = {k: int(v) if isinstance(v, (np.integer, int)) else float(v) 
                                for k, v in risk_summary.items()}
    with open(output_dir / "risk_summary.json", 'w') as f:
        json.dump(risk_summary_serializable, f, indent=2)
    
    # ========== STEP 4: AGGREGATE FEATURES ==========
    print("\n[STEP 4] AGGREGATING NODE FEATURES TO HYPEREDGE LEVEL")
    print("-" * 70)
    
    feature_agg = FeatureAggregator(hypergraph)
    features_df = feature_agg.aggregate_all_features()
    
    print(f"\nFeature Matrix:")
    print(f"  Shape: {features_df.shape}")
    print(f"  Features: {len(features_df.columns)}")
    print(f"\nFeature columns:")
    for col in features_df.columns:
        if col != 'hyperedge_id':
            print(f"    - {col}")
    
    # Save features
    features_df.to_csv(datasets_dir / "features.csv", index=False)
    
    # ========== STEP 5: TRAIN BASELINE MODELS ==========
    print("\n[STEP 5] TRAINING BASELINE ML MODELS")
    print("-" * 70)
    
    trainer = BaselineModelTrainer(random_state=42)
    X, y, feature_names = trainer.prepare_data(features_df, labels_df)
    data_split = trainer.split_and_scale(X, y)
    
    # Train models
    xgb_model, xgb_metrics = trainer.train_xgboost(data_split)
    rf_model, rf_metrics = trainer.train_random_forest(data_split)
    gb_model, gb_metrics = trainer.train_gradient_boosting(data_split)
    
    # Model comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    comparison = trainer.get_model_comparison()
    print(comparison.to_string(index=False))
    
    # Save models and results
    models_dir = output_dir / "models"
    trainer.save_models(str(models_dir))
    
    # ========== STEP 6: GENERATE REPORT ==========
    print("\n[STEP 6] GENERATING FINAL REPORT")
    print("-" * 70)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'n_suppliers': int(len(nodes_df)),
            'n_assemblies': int(len(hyperedges_df)),
            'n_incidence_relations': int(len(incidence_df)),
            'n_echelon_dependencies': int(len(echelon_df))
        },
        'hypergraph_stats': {k: int(v) if isinstance(v, (np.integer, int)) else float(v)
                            for k, v in stats.items()},
        'risk_distribution': {k: int(v) if isinstance(v, (np.integer, int)) else float(v)
                             for k, v in risk_summary.items()},
        'model_results': comparison.to_dict(orient='records'),
        'feature_importance': {
            'xgboost': trainer.results['xgboost']['feature_importance'].to_dict(),
            'random_forest': trainer.results['random_forest']['feature_importance'].to_dict(),
            'gradient_boosting': trainer.results['gradient_boosting']['feature_importance'].to_dict()
        }
    }
    
    with open(output_dir / "final_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    print(f"  📊 datasets/: nodes, hyperedges, incidence, features, labels")
    print(f"  📈 models/: trained baseline models and results")
    print(f"  📄 hypergraph.json: Full hypergraph structure")
    print(f"  📋 final_report.json: Complete system report")
    print(f"  ⚠️  risk_summary.json: Risk distribution analysis")
    
    return {
        'hypergraph': hypergraph,
        'features': features_df,
        'labels': labels_df,
        'models': trainer.models,
        'report': report
    }


if __name__ == "__main__":
    system = main()
