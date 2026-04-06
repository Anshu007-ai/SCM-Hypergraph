import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.metrics import classification_report

# Ensure the project root is in the Python path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(_PROJECT_ROOT.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize

from src.data.dataco_loader import DataCoLoader
from src.data.data_adapter import DataAdapter
from src.hypergraph.risk_labels import RiskLabelGenerator
from src.models.ht_hgnn_model import HeterogeneousTemporalHypergraphNN
from src.evaluation.utils import flatten_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Low', 'Medium', 'High', 'Critical']
RISK_LEVEL_MAP = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}

def get_data_and_config():
    """
    Loads data and extracts configuration needed for model initialization.
    This is a self-contained function to ensure the figure generation script
    is independent of the training script's data processing.
    """
    print("Loading DataCo dataset for configuration and evaluation...")
    loader = DataCoLoader()
    raw_data = loader.build_hypergraph()

    # Generate risk labels on the fly
    print("Generating risk labels...")
    risk_gen = RiskLabelGenerator(raw_data['hypergraph'])
    labels_df = risk_gen.generate_all_labels()

    num_nodes = raw_data['node_features'].shape[0]
    node_labels_numeric = np.zeros(num_nodes, dtype=int)
    node_id_to_idx = {node.node_id: i for i, node in enumerate(raw_data['hypergraph'].nodes.values())}

    for _, row in labels_df.iterrows():
        he_id = row['hyperedge_id']
        risk_level_str = row['risk_level']
        risk_level_numeric = RISK_LEVEL_MAP.get(risk_level_str, 0)
        
        if he_id in raw_data['hypergraph'].hyperedges:
            member_nodes = raw_data['hypergraph'].hyperedges[he_id].nodes
            for node_id in member_nodes:
                if node_id in node_id_to_idx:
                    node_idx = node_id_to_idx[node_id]
                    # Use max risk level for a node involved in multiple hyperedges
                    if risk_level_numeric > node_labels_numeric[node_idx]:
                        node_labels_numeric[node_idx] = risk_level_numeric

    # Map [0,1,2,3,4] to [0,0,1,2,3] for 4-class classification
    final_node_labels = np.select(
        [node_labels_numeric <= 1, node_labels_numeric == 2, node_labels_numeric == 3, node_labels_numeric == 4],
        [0, 1, 2, 3],
        default=0
    )
    raw_data['criticality_labels'] = final_node_labels

    # Use the adapter to get the final processed data and config
    adapter = DataAdapter()
    data = adapter.transform(raw_data)
    data['criticality_labels'] = final_node_labels # Ensure labels are in the final dict

    # Extract config needed for model initialization
    model_config = {
        'node_types': data['node_types'],
        'edge_types': data['edge_types'],
        'in_channels': data['node_features'].shape[1],
        'num_hgt_heads': 4,
        'num_hgnn_layers': 2,
        'use_spectral_conv': True,
        'time_window': 10,
        'num_nodes': len(data['node_types']),
        'num_hyperedges': data['incidence_matrix'].shape[0]
    }
    
    # For evaluation, we use all data
    indices = np.arange(len(final_node_labels))
    
    snaps = {
        'node_features': data['node_features'][indices],
        'incidence_matrix': data['incidence_matrix'] # Pass the full matrix
    }
    labels = data['criticality_labels'][indices]

    return (snaps, labels), model_config, data

def plot_confusion_matrices(cms, titles, class_names):
    """Plots side-by-side normalized confusion matrices."""
    fig, axes = plt.subplots(1, len(cms), figsize=(8 * len(cms), 6))
    if len(cms) == 1:
        axes = [axes]
        
    for i, (cm, title) in enumerate(zip(cms, titles)):
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1 # Avoid division by zero
        cm_normalized = cm.astype('float') / row_sums
        
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[i])
        axes[i].set_title(title)
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
        
    fig.tight_layout()
    return fig

def plot_roc_curves(y_true, y_prob, class_names, model_name):
    """Plots ROC curves for each class for a single model."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    n_classes = y_true_bin.shape[1]

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve for {class_names[i]} (area = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Multi-class ROC for {model_name}')
    ax.legend(loc="lower right")
    return fig

def main():
    """Main figure generation pipeline for HT-HGNN vs. Baseline."""
    print(f"\n{'=' * 80}")
    print("PUBLICATION VISUALIZATION GENERATION (HT-HGNN vs. RF Baseline)")
    print(f"{'=' * 80}")

    output_dir = _PROJECT_ROOT / "outputs" / "figures"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")
    
    checkpoint_path = _PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        print(f"FATAL: Checkpoint not found at {checkpoint_path}. Cannot proceed.")
        return

    # --- Step 1: Data Loading and Config Extraction ---
    print("\n[Step 1] Loading data and extracting model configuration...")
    (test_snaps, test_labels), model_config, full_data = get_data_and_config()
    print("[OK] Data loaded and config extracted.")
    print("Model Config:", model_config)

    # --- Step 2: Load HT-HGNN Model ---
    print("\n[Step 2] Loading HT-HGNN model from checkpoint...")
    hgnn_model = HeterogeneousTemporalHypergraphNN(**model_config).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Check for config in checkpoint (good practice)
    if 'config' in checkpoint:
        print("Found 'config' in checkpoint, but using config derived from current data loader for consistency.")
        # In a real scenario, you might validate they match. For now, we trust the loader.
    
    try:
        hgnn_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        hgnn_model.eval()
        print("[OK] HT-HGNN model state_dict loaded successfully.")
    except RuntimeError as e:
        print(f"FATAL: Error loading state_dict. This indicates a mismatch between the model architecture defined here and the one in the checkpoint.")
        print("Please ensure the parameters in `get_data_and_config` match the training script.")
        print(f"Error details: {e}")
        return

    # --- Step 3: Generate HT-HGNN Predictions ---
    print("\n[Step 3] Generating predictions with HT-HGNN...")
    with torch.no_grad():
        node_features_tensor = torch.tensor(test_snaps['node_features'], dtype=torch.float32).to(DEVICE)
        incidence_matrix_tensor = test_snaps['incidence_matrix'].to(DEVICE)
        
        # The model expects a list of snapshots for the temporal dimension, even if it's just one
        mock_temporal_data = [{
            'node_features': node_features_tensor,
            'incidence_matrix': incidence_matrix_tensor,
            'timestamps': torch.arange(node_features_tensor.shape[0]).to(DEVICE) # Dummy timestamps
        }]
        
        outputs = hgnn_model(mock_temporal_data)
        y_prob_hgnn_tensor = torch.softmax(outputs['criticality'], dim=1)
        y_pred_hgnn_tensor = torch.argmax(y_prob_hgnn_tensor, dim=1)
        
        y_pred_hgnn = y_pred_hgnn_tensor.cpu().numpy()
        y_prob_hgnn = y_prob_hgnn_tensor.cpu().numpy()
    print("[OK] HT-HGNN predictions generated.")

    # --- Step 3.5: Analyze Predictions and True Labels ---
    print("\n[Step 3.5] Analyzing true vs. predicted labels...")
    unique_true = np.unique(test_labels)
    unique_pred = np.unique(y_pred_hgnn)
    print(f"Unique TRUE labels in test set:    {unique_true}")
    print(f"Unique PREDICTED labels from model: {unique_pred}")

    class_names_in_data = [CLASS_NAMES[i] for i in unique_true]
    
    print("\nClassification Report (HT-HGNN):")
    # Note: We specify labels to see the report for all possible classes, even those not present.
    report = classification_report(test_labels, y_pred_hgnn, target_names=class_names_in_data, labels=unique_true, zero_division=0)
    print(report)

    print("Confusion Matrix (HT-HGNN) - Raw Counts:")
    cm_hgnn = confusion_matrix(test_labels, y_pred_hgnn, labels=unique_true)
    print(cm_hgnn)
    
    # --- Step 4: Train and Predict with Random Forest Baseline ---
    print("\n[Step 4] Training and predicting with Random Forest baseline...")
    # Flatten data for scikit-learn
    X_flat, y_flat = flatten_data(np.expand_dims(test_snaps['node_features'], axis=0), np.expand_dims(test_labels, axis=0))
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_flat, y_flat) # Train on the full dataset for this viz
    y_pred_rf = rf_model.predict(X_flat)
    y_prob_rf = rf_model.predict_proba(X_flat)
    print("[OK] Random Forest baseline trained and predictions generated.")

    # --- Step 5: Generate and Save Figures ---
    print("\n[Step 5] Generating and saving comparison figures...")
    
    # Confusion Matrices
    cm_rf = confusion_matrix(test_labels, y_pred_rf)
    cm_hgnn = confusion_matrix(test_labels, y_pred_hgnn)
    
    cm_fig = plot_confusion_matrices(
        [cm_rf, cm_hgnn],
        ['Random Forest Baseline', 'HT-HGNN v2.0'],
        CLASS_NAMES
    )
    cm_png_path = output_dir / "comparison_confusion_matrix.png"
    cm_fig.savefig(cm_png_path, dpi=300, bbox_inches='tight')
    print(f"  Saved comparison confusion matrix to {cm_png_path}")

    # ROC Curve for HT-HGNN
    roc_fig_hgnn = plot_roc_curves(test_labels, y_prob_hgnn, CLASS_NAMES, 'HT-HGNN v2.0')
    roc_hgnn_png_path = output_dir / "hgnn_roc_curves.png"
    roc_fig_hgnn.savefig(roc_hgnn_png_path, dpi=300)
    print(f"  Saved HT-HGNN ROC curve plot to {roc_hgnn_png_path}")

    print(f"\n{'=' * 80}")
    print("FIGURE GENERATION COMPLETE")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()

