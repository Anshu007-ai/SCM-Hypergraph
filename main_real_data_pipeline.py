"""
Main Pipeline with Real Data Integration
Uses original datasets from Data set/ folder instead of synthetic data
"""

import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.real_data_loader import RealDataLoader
from src.hypergraph.hypergraph import Hypergraph
from src.hypergraph.risk_labels import RiskLabelGenerator, FeatureAggregator
from src.models.baseline_models import BaselineModelTrainer


def create_output_directories():
    """Create output directory structure if it doesn't exist."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "datasets").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    return output_dir


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def main():
    """Main pipeline execution with real data."""
    
    print("\n" + "=" * 80)
    print("SUPPLY CHAIN HYPERGRAPH RISK MODELING - REAL DATA PIPELINE")
    print("=" * 80)
    
    output_dir = create_output_directories()
    
    # ========================================================================
    # STEP 1: Load Real Data
    # ========================================================================
    print("\n[STEP 1] Loading Real Data from Data set/ folder...")
    print("-" * 80)
    
    loader = RealDataLoader("Data set")
    data = loader.load_all()
    
    nodes_df = data['nodes']
    hyperedges_df = data['hyperedges']
    incidence_df = data['incidence']
    reliability_dict = data['reliability']
    
    # Save extracted data
    nodes_df.to_csv(output_dir / "datasets" / "nodes.csv", index=False)
    hyperedges_df.to_csv(output_dir / "datasets" / "hyperedges.csv", index=False)
    incidence_df.to_csv(output_dir / "datasets" / "incidence.csv", index=False)
    
    print(f"\n✓ Saved extracted data to outputs/datasets/")
    
    # ========================================================================
    # STEP 2: Build Hypergraph
    # ========================================================================
    print("\n[STEP 2] Building Hypergraph from Real Data...")
    print("-" * 80)
    
    hypergraph = Hypergraph.from_dataframes(
        nodes_df=nodes_df,
        hyperedges_df=hyperedges_df,
        incidence_df=incidence_df
    )
    
    stats = hypergraph.get_statistics()
    print(f"\n✓ Hypergraph constructed successfully:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # ========================================================================
    # STEP 3: Compute Risk Labels (HCI)
    # ========================================================================
    print("\n[STEP 3] Computing Risk Labels (HCI)...")
    print("-" * 80)
    
    risk_generator = RiskLabelGenerator(
        hypergraph=hypergraph,
        alpha=0.5,      # Joint failure probability weight
        beta=0.3,       # Engineering impact weight
        gamma=0.2       # Propagation risk weight
    )
    
    labels_dict = {}
    for he_id in hypergraph.hyperedges.keys():
        hci_data = risk_generator.compute_hci_label(he_id)
        labels_dict[he_id] = hci_data
    
    # Convert to DataFrame
    labels_df = pd.DataFrame(list(labels_dict.values()))
    
    labels_df.to_csv(output_dir / "datasets" / "hci_labels.csv", index=False)
    
    print(f"\n✓ Risk labels computed for {len(labels_df)} hyperedges:")
    print(f"  Mean HCI: {labels_df['HCI'].mean():.4f}")
    print(f"  Std HCI: {labels_df['HCI'].std():.4f}")
    print(f"  HCI range: [{labels_df['HCI'].min():.4f}, {labels_df['HCI'].max():.4f}]")
    print(f"\nRisk Level Distribution:")
    print(labels_df['risk_level'].value_counts().to_string())
    
    # ========================================================================
    # STEP 4: Feature Aggregation
    # ========================================================================
    print("\n[STEP 4] Aggregating Features...")
    print("-" * 80)
    
    aggregator = FeatureAggregator(hypergraph)
    features_df = aggregator.aggregate_all_features()
    
    features_df.to_csv(output_dir / "datasets" / "features.csv", index=False)
    
    print(f"\n✓ Features aggregated for {len(features_df)} hyperedges:")
    print(f"  Total features: {features_df.shape[1] - 1}")  # -1 for hyperedge_id
    print(f"  Feature names:")
    for i, col in enumerate(features_df.columns[1:], 1):
        print(f"    {i}. {col}")
    
    # ========================================================================
    # STEP 5: Train Baseline Models
    # ========================================================================
    print("\n[STEP 5] Training Baseline ML Models...")
    print("-" * 80)
    
    trainer = BaselineModelTrainer(random_state=42)
    
    # Merge features with labels for model training
    features_with_labels = features_df.merge(
        labels_df[['hyperedge_id', 'HCI']], 
        on='hyperedge_id'
    )
    
    # Prepare data - this returns X, y, feature_names
    X, y, feature_names = trainer.prepare_data(features_df, labels_df)
    
    # Split data
    split_data = trainer.split_and_scale(
        X,
        y,
        test_size=0.2,
        val_size=0.2
    )
    
    print(f"\n✓ Data split:")
    print(f"  Training: {len(split_data['X_train'])} samples")
    print(f"  Validation: {len(split_data['X_val'])} samples")
    print(f"  Test: {len(split_data['X_test'])} samples")
    
    # Train models
    models_results = {}
    
    print("\n  Training XGBoost...")
    xgb_model, xgb_metrics = trainer.train_xgboost(split_data)
    models_results['xgboost'] = {'model': xgb_model, 'metrics': xgb_metrics}
    print(f"    R² Score: {xgb_metrics['test_r2']:.4f}")
    print(f"    RMSE: {xgb_metrics['test_rmse']:.4f}")
    
    print("\n  Training Random Forest...")
    rf_model, rf_metrics = trainer.train_random_forest(split_data)
    models_results['random_forest'] = {'model': rf_model, 'metrics': rf_metrics}
    print(f"    R² Score: {rf_metrics['test_r2']:.4f}")
    print(f"    RMSE: {rf_metrics['test_rmse']:.4f}")
    
    print("\n  Training Gradient Boosting...")
    gb_model, gb_metrics = trainer.train_gradient_boosting(split_data)
    models_results['gradient_boosting'] = {'model': gb_model, 'metrics': gb_metrics}
    print(f"    R² Score: {gb_metrics['test_r2']:.4f}")
    print(f"    RMSE: {gb_metrics['test_rmse']:.4f}")
    
    # Save models
    trainer.save_models(output_dir / "models")
    
    print(f"\n✓ Models saved to outputs/models/")
    
    # ========================================================================
    # STEP 6: Generate Final Report
    # ========================================================================
    print("\n[STEP 6] Generating Final Report...")
    print("-" * 80)
    
    # Create comprehensive report
    final_report = {
        'dataset_info': {
            'source': 'Real data from Data set/ folder',
            'nodes_count': len(nodes_df),
            'hyperedges_count': len(hyperedges_df),
            'incidence_count': len(incidence_df),
            'features_per_hyperedge': features_df.shape[1] - 1
        },
        'hypergraph_statistics': convert_numpy_types(stats),
        'risk_distribution': {
            'mean_hci': float(labels_df['HCI'].mean()),
            'std_hci': float(labels_df['HCI'].std()),
            'min_hci': float(labels_df['HCI'].min()),
            'max_hci': float(labels_df['HCI'].max()),
            'risk_level_distribution': labels_df['risk_level'].value_counts().to_dict()
        },
        'model_performance': {
            'xgboost': {
                'r2_score': float(xgb_metrics['test_r2']),
                'rmse': float(xgb_metrics['test_rmse']),
                'mae': float(xgb_metrics['test_mae']),
                'top_features': []
            },
            'random_forest': {
                'r2_score': float(rf_metrics['test_r2']),
                'rmse': float(rf_metrics['test_rmse']),
                'mae': float(rf_metrics['test_mae']),
                'top_features': []
            },
            'gradient_boosting': {
                'r2_score': float(gb_metrics['test_r2']),
                'rmse': float(gb_metrics['test_rmse']),
                'mae': float(gb_metrics['test_mae']),
                'top_features': []
            }
        },
        'best_model': max(
            [('xgboost', xgb_metrics), ('random_forest', rf_metrics), ('gradient_boosting', gb_metrics)],
            key=lambda x: x[1]['test_r2']
        )[0]
    }
    
    # Save report
    with open(output_dir / "final_report.json", 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n✓ Final report saved to outputs/final_report.json")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"\n✓ Real Data Loading: Complete")
    print(f"  Suppliers: {len(nodes_df)}")
    print(f"  Products: {len(hyperedges_df)}")
    print(f"  Relationships: {len(incidence_df)}")
    print(f"\n✓ Hypergraph Construction: Complete")
    print(f"\n✓ Risk Label Computation: Complete")
    print(f"  Mean HCI: {labels_df['HCI'].mean():.4f}")
    print(f"\n✓ Feature Engineering: Complete")
    print(f"  Features: {features_df.shape[1] - 1}")
    print(f"\n✓ Model Training: Complete")
    print(f"  Best Model: {final_report['best_model']}")
    print(f"  R² Score: {final_report['model_performance'][final_report['best_model']]['r2_score']:.4f}")
    print(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    print("\n" + "=" * 80)
    
    return final_report


if __name__ == "__main__":
    report = main()
