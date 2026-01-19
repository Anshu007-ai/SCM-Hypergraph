"""
ANALYSIS SCRIPT: Detailed analysis of results
Run after main_pipeline.py to generate insights
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import pickle
import sys

def load_results():
    """Load all results from pipeline"""
    output_dir = Path("outputs")
    
    results = {
        'nodes': pd.read_csv(output_dir / "datasets" / "nodes.csv"),
        'hyperedges': pd.read_csv(output_dir / "datasets" / "hyperedges.csv"),
        'incidence': pd.read_csv(output_dir / "datasets" / "incidence.csv"),
        'features': pd.read_csv(output_dir / "datasets" / "features.csv"),
        'labels': pd.read_csv(output_dir / "datasets" / "hci_labels.csv"),
    }
    
    with open(output_dir / "hypergraph.json") as f:
        results['hypergraph'] = json.load(f)
    
    with open(output_dir / "final_report.json") as f:
        results['report'] = json.load(f)
    
    return results, output_dir


def analyze_risk_distribution(labels_df):
    """Analyze risk distribution across hyperedges"""
    print("\n" + "="*70)
    print("RISK DISTRIBUTION ANALYSIS")
    print("="*70)
    
    print("\nHCI Statistics:")
    print(labels_df['HCI'].describe())
    
    print("\nRisk Level Breakdown:")
    print(labels_df['risk_level'].value_counts().sort_index())
    
    # Percentiles
    percentiles = labels_df['HCI'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
    print("\nPercentiles:")
    for p, v in percentiles.items():
        print(f"  {p*100:.0f}th: {v:.4f}")
    
    # Components
    print("\nComponent Scores (Mean):")
    print(f"  Joint Failure Prob: {labels_df['joint_failure_prob'].mean():.4f}")
    print(f"  Engineering Impact: {labels_df['engineering_impact'].mean():.4f}")
    print(f"  Propagation Risk: {labels_df['propagation_risk'].mean():.4f}")
    print(f"  Concentration Risk: {labels_df['concentration_risk'].mean():.4f}")
    
    # Highest risk
    print("\nTop 5 Highest Risk Hyperedges:")
    top5 = labels_df.nlargest(5, 'HCI')[['hyperedge_id', 'HCI', 'risk_level']]
    print(top5.to_string(index=False))
    
    # Lowest risk
    print("\nTop 5 Lowest Risk Hyperedges:")
    bottom5 = labels_df.nsmallest(5, 'HCI')[['hyperedge_id', 'HCI', 'risk_level']]
    print(bottom5.to_string(index=False))


def analyze_feature_importance(report):
    """Analyze feature importance across models"""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    models = ['xgboost', 'random_forest', 'gradient_boosting']
    
    for model_name in models:
        print(f"\n{model_name.upper()} - Top 10 Features:")
        features = report['feature_importance'][model_name]
        
        # Convert dict to DataFrame for sorting
        if isinstance(features, dict) and 'feature' in features:
            df = pd.DataFrame(features)
        else:
            # Handle case where feature_importance is already a dict of dicts
            df_dict = {}
            for feature_name, importance in features.items():
                if isinstance(importance, dict):
                    df_dict[feature_name] = importance
                else:
                    df_dict[feature_name] = {'importance': importance}
            df = pd.DataFrame(df_dict).T.reset_index()
            df.columns = ['feature', 'importance']
        
        df = df.sort_values('importance', ascending=False).head(10)
        print(df.to_string(index=False))


def analyze_hypergraph_structure(hypergraph_data, nodes_df, hyperedges_df):
    """Analyze hypergraph structural properties"""
    print("\n" + "="*70)
    print("HYPERGRAPH STRUCTURE ANALYSIS")
    print("="*70)
    
    nodes = hypergraph_data['nodes']
    hyperedges = hypergraph_data['hyperedges']
    
    # Node analysis
    print("\nNode (Supplier) Analysis:")
    print(f"  Total suppliers: {len(nodes)}")
    
    # Tier distribution
    tiers = [n.get('tier', 0) for n in nodes.values()]
    print(f"  Tier distribution: {dict(pd.Series(tiers).value_counts().sort_index())}")
    
    # Reliability distribution
    reliabilities = [n['reliability'] for n in nodes.values()]
    print(f"  Reliability - Mean: {np.mean(reliabilities):.4f}, Std: {np.std(reliabilities):.4f}")
    print(f"  Lead time - Mean: {np.mean([n['lead_time'] for n in nodes.values()]):.2f} days")
    
    # Hyperedge analysis
    print("\nHyperedge (Subassembly) Analysis:")
    print(f"  Total hyperedges: {len(hyperedges)}")
    
    # Size distribution
    sizes = [len(he['nodes']) for he in hyperedges.values()]
    print(f"  Hyperedge size - Mean: {np.mean(sizes):.2f}, Std: {np.std(sizes):.2f}")
    print(f"  Size range: [{min(sizes)}, {max(sizes)}]")
    
    # Tier level distribution
    tier_levels = [he.get('tier_level', 0) for he in hyperedges.values()]
    print(f"  Tier level distribution: {dict(pd.Series(tier_levels).value_counts().sort_index())}")
    
    # Critical path
    critical_count = sum(1 for he in hyperedges.values() if he['critical_path'] > 0)
    print(f"  On critical path: {critical_count}/{len(hyperedges)} ({critical_count/len(hyperedges)*100:.1f}%)")


def analyze_model_performance(report):
    """Analyze model performance metrics"""
    print("\n" + "="*70)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*70)
    
    results = report['model_results']
    df = pd.DataFrame(results)
    
    print("\nDetailed Metrics:")
    print(df.to_string(index=False))
    
    print("\nKey Insights:")
    best_r2 = df.loc[df['Test R²'].idxmax()]
    print(f"  Best model (R²): {best_r2['Model']} with R²={best_r2['Test R²']:.4f}")
    
    best_rmse = df.loc[df['Test RMSE'].idxmin()]
    print(f"  Best RMSE: {best_rmse['Model']} with RMSE={best_rmse['Test RMSE']:.4f}")
    
    # Overfitting analysis
    print("\nOverfitting Analysis:")
    for _, row in df.iterrows():
        overfit = row['Train R²'] - row['Test R²']
        print(f"  {row['Model']}: {overfit:.4f} (Train-Test R² gap)")


def analyze_correlations(features_df, labels_df):
    """Analyze feature-target correlations"""
    print("\n" + "="*70)
    print("FEATURE-TARGET CORRELATIONS")
    print("="*70)
    
    # Merge features and labels
    merged = features_df.merge(labels_df[['hyperedge_id', 'HCI']], on='hyperedge_id')
    
    # Remove non-numeric columns
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    
    # Calculate correlations with HCI
    correlations = merged[numeric_cols].corr()['HCI'].sort_values(ascending=False)
    
    print("\nTop 15 Positively Correlated Features:")
    print(correlations.head(15).to_string())
    
    print("\nTop 15 Negatively Correlated Features:")
    print(correlations.tail(15).to_string())


def generate_summary_report(results, output_dir):
    """Generate a summary text report"""
    print("\n" + "="*70)
    print("SAVING SUMMARY REPORT")
    print("="*70)
    
    report_path = output_dir / "analysis_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("SUPPLY CHAIN HYPERGRAPH RISK MODELING - ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. DATASET OVERVIEW\n")
        f.write("-"*70 + "\n")
        f.write(f"Suppliers: {len(results['nodes'])}\n")
        f.write(f"Subassemblies: {len(results['hyperedges'])}\n")
        f.write(f"Incidence relationships: {len(results['incidence'])}\n")
        f.write(f"Total features: {len(results['features'].columns)}\n\n")
        
        f.write("2. RISK DISTRIBUTION\n")
        f.write("-"*70 + "\n")
        labels = results['labels']
        f.write(f"Mean HCI: {labels['HCI'].mean():.4f}\n")
        f.write(f"Std HCI: {labels['HCI'].std():.4f}\n")
        f.write(f"Min HCI: {labels['HCI'].min():.4f}\n")
        f.write(f"Max HCI: {labels['HCI'].max():.4f}\n\n")
        
        f.write("3. MODEL PERFORMANCE\n")
        f.write("-"*70 + "\n")
        report = results['report']
        for model in report['model_results']:
            f.write(f"\n{model['Model']}:\n")
            f.write(f"  Test R²: {model['Test R²']:.4f}\n")
            f.write(f"  Test RMSE: {model['Test RMSE']:.4f}\n")
            f.write(f"  Test MAE: {model['Test MAE']:.4f}\n")
    
    print(f"✓ Saved: {report_path}")


def main():
    """Run complete analysis"""
    print("\n" + "="*70)
    print("SUPPLY CHAIN HYPERGRAPH - DETAILED ANALYSIS")
    print("="*70)
    
    # Load results
    results, output_dir = load_results()
    
    # Run analyses
    analyze_risk_distribution(results['labels'])
    analyze_feature_importance(results['report'])
    analyze_hypergraph_structure(results['hypergraph'], 
                                results['nodes'], 
                                results['hyperedges'])
    analyze_model_performance(results['report'])
    analyze_correlations(results['features'], results['labels'])
    
    # Generate summary
    generate_summary_report(results, output_dir)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
