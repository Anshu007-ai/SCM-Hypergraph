import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path to import project modules
# This assumes the script is run from the root of the project
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# It seems there might be an issue with relative paths, so let's try to construct paths from the script's location.
# However, for simplicity and given the workspace structure, I will assume the script is run from the root.
# If not, paths might need adjustment.
sys.path.append('src')
from data.bom_loader import BOMLoader
from data.indigo_disruption_loader import IndiGoDisruptionLoader
from data.data_adapter import DataAdapter
from models.ht_hgnn_model import HeterogeneousTemporalHypergraphNN as HT_HGNN

def diagnose_training_collapse():
    """
    Diagnoses the training collapse by inspecting the checkpoint, data, and model outputs.
    """
    print("--- Starting Training Collapse Diagnosis ---")
    
    # Correctly determine the project root
    project_root = Path(__file__).parent.resolve()
    print(f"Project Root detected as: {project_root}")


    # Define paths relative to the project root
    checkpoint_path = project_root / "outputs/checkpoints/best.pt"
    hci_labels_path = project_root / "outputs/datasets/hci_labels.csv"
    output_fig_path = project_root / "paper_figures/training_curve.png"
    output_fig_path.parent.mkdir(exist_ok=True, parents=True)

    # --- 1. Load best.pt and analyze ---
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        # Let's try to find it in a common alternative location from the instructions
        alt_checkpoint_path = project_root / "outputs/checkpoints/best_dataco.pt"
        if alt_checkpoint_path.exists():
            checkpoint_path = alt_checkpoint_path
            print(f"Found checkpoint at alternative path: {checkpoint_path}")
        else:
            return

    print(f"\n1. Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    history = checkpoint.get('history')
    
    if not history:
        print("ERROR: `history` not found in checkpoint.")
        print("Available keys in checkpoint:", list(checkpoint.keys()))
    else:
        # Print training history
        print("\n--- Training History (Loss) ---")
        print("Available keys in history:", list(history.keys()))
        train_loss = history.get('loss', [])
        # There is no validation loss in the history, so we'll plot only train loss
        val_loss = [] # history.get('val_losses', history.get('val_loss', [])) 
        
        if train_loss:
            print("\n--- Training History (Total Loss) ---")
            for epoch, train_l in enumerate(train_loss):
                print(f"Epoch {epoch+1:02d}: Train Loss = {train_l:.4f}")
        else:
            print("Could not find 'loss' in history dict.")

        # --- 3. Plot training loss curve ---
        print(f"\n3. Plotting training loss curve and saving to {output_fig_path}...")
        plt.figure(figsize=(12, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_fig_path)
        print(f"   Plot saved successfully to {output_fig_path}")

    # --- 2. Check data source and distribution ---
    print("\n2. Checking data sources and distributions...")
    
    # Check BOM Loader
    print("\n--- BOM Loader ---")
    try:
        # The loaders have hardcoded paths, so we don't pass file_path
        bom_loader = BOMLoader()
        if bom_loader.df is not None and not bom_loader.df.empty:
            print(f"   BOM Loader successfully loaded data from its internal path.")
            print("   Synthetic data generation was NOT triggered.")
            adapter = DataAdapter(bom_loader)
            data = adapter.get_data()
            labels = data['criticality_labels']
            unique, counts = np.unique(labels, return_counts=True)
            print("\n   --- Training Data Class Distribution (BOM) ---")
            print("      Class mapping: {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}")
            for label, count in zip(unique, counts):
                print(f"      Class {label}: {count} samples")
        else:
            print(f"   BOM Loader did NOT find or failed to load CSV from {bom_path}.")
            # The loader has a fallback to generate synthetic data.
            print("   The 'CSV not found, generating synthetic' fallback branch was likely hit if data was processed.")

    except Exception as e:
        print(f"   Could not inspect BOMLoader: {e}")

    # Check IndiGo Loader
    print("\n--- IndiGo Loader ---")
    try:
        # The loader has a hardcoded path
        indigo_loader = IndiGoDisruptionLoader()
        if indigo_loader.df is not None and not indigo_loader.df.empty:
            print(f"   IndiGo Loader successfully loaded data from its internal path.")
            print("   Synthetic data generation was NOT triggered.")
            adapter = DataAdapter(indigo_loader)
            data = adapter.get_data()
            labels = data['criticality_labels']
            unique, counts = np.unique(labels, return_counts=True)
            print("\n   --- Training Data Class Distribution (IndiGo) ---")
            print("      Class mapping: {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}")
            for label, count in zip(unique, counts):
                print(f"      Class {label}: {count} samples")
        else:
            print(f"   IndiGo Loader did NOT find or failed to load CSV from {indigo_path}.")
            print("   The 'CSV not found, generating synthetic' fallback branch was likely hit if data was processed.")
    except FileNotFoundError:
        print(f"   IndiGo Loader file not found at {indigo_path}. The loader would use synthetic data.")
    except Exception as e:
        print(f"   Could not inspect IndiGoDisruptionLoader: {e}")


    # --- 4. Check hci_labels.csv distribution ---
    if hci_labels_path.exists():
        print(f"\n4. Checking criticality distribution in {hci_labels_path}...")
        try:
            hci_df = pd.read_csv(hci_labels_path)
            if 'risk_level' in hci_df.columns:
                print("--- Class Distribution in hci_labels.csv ---")
                print(hci_df['risk_level'].value_counts())
            else:
                print(f"   'risk_level' column not found in {hci_labels_path}. Available columns: {hci_df.columns.tolist()}")
        except Exception as e:
            print(f"   Could not read or process {hci_labels_path}: {e}")
    else:
        print(f"\n4. {hci_labels_path} not found. Skipping analysis.")

    # --- 1. (cont.) Check model outputs ---
    print("\n1. (cont.) Checking model outputs...")
    model_state_dict = checkpoint.get('model_state_dict')
    if model_state_dict:
        try:
            # The keys suggest the head is nn.Sequential, let's look for the final layer
            crit_head_weights = model_state_dict.get('criticality_head.2.weight')
            crit_head_bias = model_state_dict.get('criticality_head.2.bias')

            if crit_head_weights is not None and crit_head_bias is not None:
                print("\n--- Criticality Head Output Layer Analysis ---")
                print("   Weights:\n", crit_head_weights)
                print("   Bias:\n", crit_head_bias)
                
                if torch.argmax(crit_head_bias) == torch.argmin(crit_head_bias):
                     print("   Bias terms are uniform.")
                else:
                     print(f"   Bias term for class {torch.argmax(crit_head_bias)} is the largest.")
                     print("   This could indicate a bias towards predicting one class.")

            else:
                print("   Could not find 'criticality_head.2.weight' or 'criticality_head.2.bias' in model state_dict.")
                print("   Available keys:", [k for k in model_state_dict.keys() if 'criticality' in k])

        except Exception as e:
            print(f"   Error inspecting model weights: {e}")
    else:
        print("   `model_state_dict` not found, cannot analyze model outputs.")


    print("\n--- Diagnosis Complete ---")

if __name__ == "__main__":
    # This is a bit of a hack to make sure we can import from src
    # when running from the command line.
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the absolute path of the project root (assuming the script is in the root)
    project_root = script_dir
    # Add the 'src' directory to the Python path
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Now we can import our modules
    from data.bom_loader import BOMLoader
    # A little risky, but let's try to guess the name based on convention
    try:
        from data.indigo_disruption_loader import IndiGoDisruptionLoader
    except ImportError:
        print("Warning: Could not import IndiGoDisruptionLoader. Assuming a different name or it doesn't exist.")
        # Create a dummy class to avoid crashing
        class IndiGoDisruptionLoader:
            def __init__(self, file_path):
                self.df = None
                print(f"Synthetic IndiGoDisruptionLoader used for path: {file_path}")


    diagnose_training_collapse()
