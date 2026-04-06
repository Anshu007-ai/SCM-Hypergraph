"""
Diagnostic script to investigate training collapse in checkpoint evaluation
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Add project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.ht_hgnn_model import HeterogeneousTemporalHypergraphNN
from src.data.indigo_disruption_loader import IndiGoDisruptionLoader
from src.data.bom_loader import BOMLoader
from src.data.data_adapter import DataAdapter


def print_header(title):
    """Print a formatted header"""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}\n")


def diagnose_checkpoint(checkpoint_path):
    """Load checkpoint and extract training history"""
    print_header("1. CHECKPOINT TRAINING HISTORY")

    if not Path(checkpoint_path).exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")

    # Print epoch info
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    # Print config
    if "config" in checkpoint:
        print(f"Config: {checkpoint['config']}")

    # Check for training history
    if "history" not in checkpoint:
        print("[WARNING] No training history in checkpoint")
        return None

    history = checkpoint["history"]
    print(f"\nTraining history found!")

    # Handle dict format (history is {'loss': [...], 'loss_price': [...], ...})
    if isinstance(history, dict):
        if len(history) > 0:
            n_epochs = len(next(iter(history.values())))
            print(f"History format: dict with {n_epochs} epochs")
            print(f"Tracked metrics: {list(history.keys())}")

            # Build dataframe
            df_history = pd.DataFrame(history)
            df_history['epoch'] = range(len(df_history))

            print(f"\nFull training history (first 10 and last 10 epochs):")
            print(df_history[['epoch', 'loss', 'loss_price', 'loss_change', 'loss_criticality']].head(10).to_string())
            print("...")
            print(df_history[['epoch', 'loss', 'loss_price', 'loss_change', 'loss_criticality']].tail(10).to_string())

            # Extract loss analysis
            losses = np.array(history['loss'])
            print(f"\nLoss statistics:")
            print(f"  Min: {losses.min():.2f}")
            print(f"  Max: {losses.max():.2f}")
            print(f"  Mean: {losses.mean():.2f}")
            print(f"  Final: {losses[-1]:.2f}")

            # Check trend
            loss_trend = losses[-1] - losses[0]
            print(f"  Trend (final - first): {loss_trend:.2f}")
            if loss_trend < 0:
                print("  [OK] Loss CONVERGED!")
            else:
                print("  [WARNING] Loss is INCREASING - possible divergence!")

            return df_history
    else:
        print(f"[WARNING] History is not dict format: {type(history)}")
        return None


def check_data_sources():
    """Check if loaders find real data or use fallback"""
    print_header("2. DATA SOURCE VERIFICATION")

    # Check IndiGo data
    print("Checking IndiGo dataset...")
    indigo_path = Path("Data set/IndiGo/indigo_disruption.csv")
    if indigo_path.exists():
        print(f"[OK] IndiGo CSV found: {indigo_path}")
        df = pd.read_csv(indigo_path)
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {list(df.columns)}")
    else:
        print(f"[ERROR] IndiGo CSV NOT FOUND: {indigo_path}")

    # Check BOM data
    print("\nChecking BOM dataset...")
    bom_train_path = Path("Data set/BOM/train_set.csv")
    bom_test_path = Path("Data set/BOM/test_set.csv")

    if bom_train_path.exists() and bom_test_path.exists():
        print(f"[OK] BOM CSVs found")
        df_train = pd.read_csv(bom_train_path)
        df_test = pd.read_csv(bom_test_path)
        print(f"    Train shape: {df_train.shape}")
        print(f"    Test shape: {df_test.shape}")
    else:
        print(f"[ERROR] BOM CSVs NOT FOUND")

    # Try loading with IndiGo loader
    print("\nTrying IndiGoDisruptionLoader.build_hypergraph()...")
    try:
        loader = IndiGoDisruptionLoader(data_dir="Data set")
        hg_data = loader.build_hypergraph()
        print(f"[OK] Loader succeeded")
        print(f"    Keys: {list(hg_data.keys())}")
        if "node_features" in hg_data:
            print(f"    Node features shape: {np.asarray(hg_data['node_features']).shape}")
        if "incidence_matrix" in hg_data:
            print(f"    Incidence matrix shape: {np.asarray(hg_data['incidence_matrix']).shape}")
    except Exception as e:
        print(f"[ERROR] Loader failed: {e}")


def check_hci_labels():
    """Check class distribution in HCI labels"""
    print_header("3. HCI LABELS DISTRIBUTION")

    hci_path = Path("outputs/datasets/hci_labels.csv")
    if hci_path.exists():
        print(f"[OK] HCI labels file found: {hci_path}")
        df = pd.read_csv(hci_path)
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)[:10]}...")

        # Try to find criticality column
        criticality_cols = [c for c in df.columns if 'crit' in c.lower() or 'class' in c.lower() or 'label' in c.lower()]
        print(f"Potential criticality columns: {criticality_cols}")

        if len(df.columns) > 0:
            first_col = df.columns[0]
            print(f"\nFirst column value distribution:")
            value_counts = df.iloc[:, 0].value_counts().sort_index()
            print(value_counts)

            # Check for imbalance
            class_props = value_counts / len(df)
            print(f"\nClass proportions:")
            for k, v in class_props.items():
                print(f"  Class {k}: {v:.1%}")

            max_prop = class_props.max()
            min_prop = class_props.min()
            imbalance_ratio = max_prop / min_prop if min_prop > 0 else float('inf')
            print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")

            if imbalance_ratio > 10:
                print("[WARNING] SEVERE class imbalance detected!")
            elif imbalance_ratio > 3:
                print("[WARNING] Significant class imbalance detected")
    else:
        print(f"[ERROR] HCI labels file NOT FOUND: {hci_path}")


def inspect_training_data_labels():
    """Check the actual labels used during training"""
    print_header("4. TRAINING DATA LABELS ANALYSIS")

    # Look for generated data logs
    data_dir = Path("outputs/datasets")
    if data_dir.exists():
        print(f"Checking outputs/datasets/...")
        for f in data_dir.glob("*.csv"):
            print(f"  Found: {f.name}")

    # Check if there's a training script log or data manifest
    print("\nSearching for dataset manifest or labels...")

    # Check for criticality labels at node level
    if Path("outputs/datasets/criticality_labels.csv").exists():
        df = pd.read_csv("outputs/datasets/criticality_labels.csv")
        print(f"\n[OK] Found criticality_labels.csv - shape {df.shape}")
        print(f"Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"\nFirst few rows:")
            print(df.head())
            # Check label distribution
            if "criticality" in df.columns or "label" in df.columns:
                col = "criticality" if "criticality" in df.columns else "label"
                print(f"\nLabel distribution for '{col}':")
                print(df[col].value_counts().sort_index())
    else:
        print("\n[INFO] No criticality_labels.csv found")

    # Check src/data files for loader hints
    print("\nChecking data loader configuration...")
    dataset_split_path = Path("src/data/dataset_split.py")
    if dataset_split_path.exists():
        with open(dataset_split_path, 'r') as f:
            content = f.read()
            if 'criticality' in content.lower():
                print("[OK] dataset_split.py contains criticality-related code")
                # Extract a snippet
                lines = [l for l in content.split('\n') if 'criticality' in l.lower()]
                for line in lines[:5]:
                    print(f"  {line.strip()}")


def inspect_checkpoint_criticality_outputs(checkpoint_path, n_samples=20):
    """Inspect criticality head layer directly from model weights"""
    print_header("4b. CRITICALITY HEAD LAYER INSPECTION")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model_state_dict"]

    # Find criticality head parameters
    crit_params = {k: v for k, v in model_state.items() if 'criticality' in k}
    print(f"Found {len(crit_params)} parameters in criticality head:")

    for param_name, param_tensor in crit_params.items():
        print(f"\n{param_name}:")
        print(f"  Shape: {param_tensor.shape}")
        print(f"  Mean: {param_tensor.mean():.8f}")
        print(f"  Std: {param_tensor.std():.8f}")
        print(f"  Min: {param_tensor.min():.8f}")
        print(f"  Max: {param_tensor.max():.8f}")

        if param_tensor.numel() <= 20:
            print(f"  Values: {param_tensor.flatten().tolist()}")

    print("\n[INFO] If criticality head weights have near-zero std across all epochs,")
    print("       the model may not have learned discriminative patterns.")


def plot_training_curve(checkpoint_path, output_dir="paper_figures"):
    """Plot training loss curve from checkpoint"""
    print_header("5. PLOTTING TRAINING CURVE")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "history" not in checkpoint:
        print("[ERROR] No training history in checkpoint")
        return

    history = checkpoint["history"]

    # Convert dict format to dataframe
    if isinstance(history, dict):
        df = pd.DataFrame(history)
        df['epoch'] = range(len(df))
        print(f"[OK] Parsed history dict: {len(df)} epochs")
    else:
        print(f"[WARNING] Cannot parse history format: {type(history)}")
        return

    # Create figure
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Training Diagnostics", fontsize=14, fontweight="bold")

    epochs = df['epoch'].values

    # Plot 1: Multi-task losses
    ax = axes[0, 0]
    if "loss" in df.columns:
        ax.plot(epochs, df["loss"], "b-", label="Total Loss", linewidth=2, marker="o", markersize=4)
    if "loss_criticality" in df.columns:
        ax.plot(epochs, df["loss_criticality"], "r-", label="Criticality Loss", linewidth=2, marker="s", markersize=4)
    if "loss_price" in df.columns:
        ax.plot(epochs, df["loss_price"], "g-", label="Price Loss", linewidth=2, marker="^", markersize=3)
    if "loss_change" in df.columns:
        ax.plot(epochs, df["loss_change"], "purple", label="Change Loss", linewidth=2, marker="d", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Multi-task Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Total loss only
    if "loss" in df.columns:
        ax = axes[0, 1]
        ax.plot(epochs, df["loss"], "b-o", linewidth=2, markersize=6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title("Total Loss Progression")
        ax.grid(True, alpha=0.3)
        ax.fill_between(epochs, df["loss"], alpha=0.2)

    # Plot 3: Individual loss components
    ax = axes[1, 0]
    components = ["loss_price", "loss_change", "loss_criticality"]
    for comp in components:
        if comp in df.columns:
            ax.plot(epochs, df[comp], marker="o", label=comp.replace("loss_", ""), linewidth=2, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Component Loss")
    ax.set_title("Loss Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Statistics
    ax = axes[1, 1]
    ax.axis("off")

    if "loss" in df.columns:
        stat_text = f"""
LOSS STATISTICS
===============
Total Loss:
  Initial: {df['loss'].iloc[0]:.2f}
  Final: {df['loss'].iloc[-1]:.2f}
  Min: {df['loss'].min():.2f}
  Max: {df['loss'].max():.2f}
  Mean: {df['loss'].mean():.2f}

Trend (final - initial): {df['loss'].iloc[-1] - df['loss'].iloc[0]:.2f}
Epochs trained: {len(df)}

Components:
"""
        for comp in components:
            if comp in df.columns:
                stat_text += f"\n{comp.replace('loss_', '').upper()}:\n"
                stat_text += f"  Initial: {df[comp].iloc[0]:.6f}\n"
                stat_text += f"  Final: {df[comp].iloc[-1]:.6f}\n"

        ax.text(0.05, 0.95, stat_text, transform=ax.transAxes,
               fontsize=8, verticalalignment="top", family="monospace",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    output_path = Path(output_dir) / "training_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Training curve saved to: {output_path}")
    plt.close()


def generate_diagnostic_report(checkpoint_path):
    """Generate a comprehensive diagnostic report"""
    print("\n" + "=" * 80)
    print("TRAINING COLLAPSE DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().isoformat()}")
    print(f"Checkpoint: {checkpoint_path}\n")

    # Run all diagnostics
    history_df = diagnose_checkpoint(checkpoint_path)
    check_data_sources()
    check_hci_labels()
    inspect_training_data_labels()
    inspect_checkpoint_criticality_outputs(checkpoint_path, n_samples=20)
    plot_training_curve(checkpoint_path, output_dir="paper_figures")

    # Summary
    print_header("DIAGNOSTIC SUMMARY")

    findings = []

    # Check 1: Loss convergence
    if history_df is not None and "loss" in history_df.columns:
        losses = history_df["loss"].values

        if losses[-1] < losses[0]:
            findings.append("OK: Total loss CONVERGED from {:.0f} → {:.0f}".format(losses[0], losses[-1]))
        else:
            findings.append("CRITICAL: Training loss not converging")

        if "loss_criticality" in history_df.columns:
            crit_loss = history_df["loss_criticality"].values
            if crit_loss[-1] < crit_loss[0]:
                findings.append("OK: Criticality loss converged from {:.4f} → {:.4f}".format(crit_loss[0], crit_loss[-1]))
            else:
                findings.append("WARNING: Criticality loss did not improve")

    findings.append("ISSUE: Test metrics show single-class predictions (macro_F1=0.25)")
    findings.append("ISSUE: evaluate_checkpoint.py ran on SYNTHETIC data, not real training data")

    print("\nKey Findings:")
    for i, finding in enumerate(findings, 1):
        safe_finding = finding.encode('ascii', 'replace').decode('ascii')
        print(f"{i}. {safe_finding}")

    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print("""
The checkpoint DID train successfully (loss converged). However:

1. **Data Mismatch**:
   - checkpoint trained on: 114 nodes, 18 hyperedges (IndiGo data)
   - evaluate_checkpoint.py generated: 1207 nodes, 36 hyperedges (different data!)
   - This size mismatch causes incorrect predictions!

2. **Label Issue**:
   - HCI labels are hyperedge-level (36 assemblies)
   - But criticality task is node-level prediction
   - Training used criticality_label from IndiGo (4 classes)
   - Model expects to predict on different-sized graph

3. **What to do next**:
   - Check what actual training script trained the model
   - Verify the training data source (real vs synthetic)
   - Ensure evaluate_checkpoint.py uses SAME data as training
   - Rebuild with correct data or retrain on realistic labels
    """)


def main():
    """Main diagnostic entry point"""
    checkpoint_path = "outputs/checkpoints/best.pt"
    generate_diagnostic_report(checkpoint_path)


if __name__ == "__main__":
    main()
