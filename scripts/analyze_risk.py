import numpy as np
import pandas as pd
from pathlib import Path
import sys
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataco_loader import DataCoLoader
from src.hypergraph.risk_labels import RiskLabelGenerator
from src.models.ht_hgnn_model import HeterogeneousTemporalHypergraphNN
from train_ht_hgnn import get_data_and_config, prepare_tensors, _load_dataset_via_loader, DATASET_REGISTRY
from src.data.data_adapter import DataAdapter

def analyze_risk_distribution_and_features():
    """
    1. Calculates the distribution of risk scores.
    2. Analyzes the correlation between input features and risk scores.
    """
    print(f"\\n{'=' * 80}")
    print("ANALYZING RISK SCORE DISTRIBUTION AND FEATURE CORRELATION")
    print(f"{'=' * 80}")

    # --- Step 1: Load data and hypergraph ---
    print("\\n[Step 1] Loading DataCo data and building hypergraph...")
    loader = DataCoLoader()
    # Use the raw dataframe loaded by the loader
    df = loader.df
    if df is None:
        print("ERROR: Failed to load DataFrame from DataCoLoader.")
        return

    raw_data = loader.build_hypergraph()
    hypergraph = raw_data['hypergraph']
    risk_gen = RiskLabelGenerator(hypergraph)
    print(f"[OK] Loaded hypergraph with {len(hypergraph.nodes)} nodes and {len(hypergraph.hyperedges)} hyperedges.")

    # --- Step 2: Calculate risk scores for all hyperedges ---
    print("\\n[Step 2] Calculating risk score for every hyperedge...")
    risk_scores = []
    all_labels_data = []
    for he_id in hypergraph.hyperedges:
        details = risk_gen.compute_hci_label(he_id)
        risk_scores.append(details['HCI'])
        details['risk_score'] = details.pop('HCI') # Rename for consistency
        all_labels_data.append(details)
    
    risk_scores = np.array(risk_scores)
    labels_df = pd.DataFrame(all_labels_data)

    # --- Step 3: Print risk score distribution ---
    print("\\n[Step 3] Risk Score Distribution (across all 11,944 hyperedges):")
    min_score = np.min(risk_scores)
    max_score = np.max(risk_scores)
    mean_score = np.mean(risk_scores)
    std_score = np.std(risk_scores)
    p90 = np.percentile(risk_scores, 90)
    p95 = np.percentile(risk_scores, 95)
    p99 = np.percentile(risk_scores, 99)

    print(f"  - Min:    {min_score:.4f}")
    print(f"  - Max:    {max_score:.4f}")
    print(f"  - Mean:   {mean_score:.4f}")
    print(f"  - Std:    {std_score:.4f}")
    print(f"  - 90th percentile: {p90:.4f} <--- New proposed 'Critical' threshold")
    print(f"  - 95th percentile: {p95:.4f}")
    print(f"  - 99th percentile: {p99:.4f}")

    # --- Step 4: Analyze feature correlation ---
    print("\\n[Step 4] Analyzing correlation between node features and risk components...")
    
    # Aggregate node features for each hyperedge
    hyperedge_features = []
    for he_id, hyperedge in hypergraph.hyperedges.items():
        member_node_indices = [hypergraph.nodes[node_id].raw_data_idx 
                               for node_id in hyperedge.nodes 
                               if node_id in hypergraph.nodes and hasattr(hypergraph.nodes[node_id], 'raw_data_idx')]
        if member_node_indices:
            # Use the raw dataframe 'df' from the loader
            feature_cols = [c for c in DataCoLoader.FEATURE_COLUMNS if c in df.columns]
            avg_features = df.iloc[member_node_indices][feature_cols].mean().to_dict()
            avg_features['hyperedge_id'] = he_id
            hyperedge_features.append(avg_features)
    
    features_df = pd.DataFrame(hyperedge_features)
    
    # Merge with risk scores
    merged_df = pd.merge(features_df, labels_df, on='hyperedge_id')
    
    correlation_cols = [c for c in DataCoLoader.FEATURE_COLUMNS if c in merged_df.columns] + ['joint_failure_prob', 'engineering_impact', 'propagation_risk', 'risk_score']
    correlation_matrix = merged_df[correlation_cols].corr()

    print("\\nCorrelation of raw input features with final risk_score:")
    print(correlation_matrix['risk_score'].sort_values(ascending=False))

    # Visualize the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Node Features and Risk Components')
    fig_path = _PROJECT_ROOT / "outputs" / "figures" / "feature_risk_correlation.png"
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path)
    print(f"\\n[OK] Correlation matrix saved to {fig_path}")


if __name__ == "__main__":
    analyze_risk_distribution_and_features()
