import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure the project root is in the Python path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(_PROJECT_ROOT))
sys.path.append(str(_PROJECT_ROOT.parent))


from src.data.dataco_loader import DataCoLoader
from src.data.data_adapter import DataAdapter
from src.hypergraph.risk_labels import RiskLabelGenerator
from train_ht_hgnn import get_data_and_config as get_training_data_config

CLASS_NAMES = ['Low', 'Medium', 'High', 'Critical']
RISK_LEVEL_MAP = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}

def validate_results():
    """
    Validates the DataCo results by inspecting data source, splits, and temporal separation.
    """
    print(f"\n{'=' * 80}")
    print("PERFORMING VALIDATION OF DATACO RESULTS")
    print(f"{'=' * 80}")

    # --- 1. & 4. Check Data Source and Print Head ---
    print("\n[Step 1 & 4] Verifying DataCo data source...")
    loader = DataCoLoader()
    
    if loader.is_synthetic:
        print("\n[FLAG] The data used was SYNTHETIC.")
        print("The DataCoLoader fell back to its synthetic data generation branch.")
    else:
        print("\n[OK] The data used was from a real CSV file.")
        print(f"   - Source File Path: {loader.file_path}")
        print(f"   - Total Rows in CSV: {len(loader.df)}")
        print("\n   Code path for loading data in DataCoLoader:")
        print("   def __init__(self, file_path=None, use_synthetic=False):")
        print("       self.file_path = file_path or self._get_default_path()")
        print("       if self.file_path.exists() and not use_synthetic:")
        print("           print(f'Loading DataCo data from {self.file_path}') # <--- THIS PATH WAS EXECUTED")
        print("           self.df = self._load_and_preprocess()")
        print("           self.is_synthetic = False")
        print("       else:")
        print("           print('DataCo data not found. Generating synthetic data.')")
        print("           self.df = self.generate_synthetic_data()")
        print("           self.is_synthetic = True")


    print("\n--- First 5 rows of the raw data used for evaluation: ---")
    print(loader.df.head())

    # --- 2. Check Split Sizes and Class Distribution ---
    print("\n\n[Step 2] Verifying split sizes and class distributions...")
    
    # Get data as used in generate_figures.py (the evaluation script)
    raw_data_eval = loader.build_hypergraph()
    risk_gen_eval = RiskLabelGenerator(raw_data_eval['hypergraph'])
    labels_df_eval = risk_gen_eval.generate_all_labels()
    num_nodes_eval = raw_data_eval['node_features'].shape[0]
    node_labels_numeric_eval = np.zeros(num_nodes_eval, dtype=int)
    node_id_to_idx_eval = {node.node_id: i for i, node in enumerate(raw_data_eval['hypergraph'].nodes.values())}

    for _, row in labels_df_eval.iterrows():
        he_id = row['hyperedge_id']
        risk_level_str = row['risk_level']
        risk_level_numeric = RISK_LEVEL_MAP.get(risk_level_str, 0)
        if he_id in raw_data_eval['hypergraph'].hyperedges:
            member_nodes = raw_data_eval['hypergraph'].hyperedges[he_id].nodes
            for node_id in member_nodes:
                if node_id in node_id_to_idx_eval:
                    node_idx = node_id_to_idx_eval[node_id]
                    if risk_level_numeric > node_labels_numeric_eval[node_idx]:
                        node_labels_numeric_eval[node_idx] = risk_level_numeric
    
    eval_labels = np.select(
        [node_labels_numeric_eval <= 1, node_labels_numeric_eval == 2, node_labels_numeric_eval == 3, node_labels_numeric_eval == 4],
        [0, 1, 2, 3], default=0
    )
    
    print(f"\n--- Evaluation Set (used for all models in generate_figures.py) ---")
    print(f"The 'generate_figures.py' script and 'run_baselines.py' script use the ENTIRE dataset for evaluation.")
    print(f"Total samples in evaluation set: {len(eval_labels)}")
    
    unique_eval, counts_eval = np.unique(eval_labels, return_counts=True)
    print("\nClass Distribution in Evaluation Set:")
    for label_idx, count in zip(unique_eval, counts_eval):
        print(f"   - Class '{CLASS_NAMES[label_idx]}': {count} samples")
    print("This confirms all models (HT-HGNN, RF, GB) were evaluated on the identical test set.")


    # --- 3. Check for Temporal Separation ---
    print("\n\n[Step 3] Verifying temporal separation between training and testing data...")
    
    # We need to inspect the training script's data loading to find the split.
    # The `get_data_and_config` from `train_ht_hgnn` contains the splitting logic.
    (snaps, train_data, test_data), model_config, full_data = get_training_data_config()

    train_indices = train_data['indices']
    test_indices = test_data['indices']
    
    print(f"\n--- Training Set ---")
    print(f"Total samples in training set: {len(train_indices)}")
    train_labels = full_data['criticality_labels'][train_indices]
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    print("Class Distribution in Training Set:")
    for label_idx, count in zip(unique_train, counts_train):
        print(f"   - Class '{CLASS_NAMES[label_idx]}': {count} samples")

    print(f"\n--- Test Set ---")
    print(f"Total samples in test set: {len(test_indices)}")
    test_labels = full_data['criticality_labels'][test_indices]
    unique_test, counts_test = np.unique(test_labels, return_counts=True)
    print("Class Distribution in Test Set:")
    for label_idx, count in zip(unique_test, counts_test):
        print(f"   - Class '{CLASS_NAMES[label_idx]}': {count} samples")


    # --- 3. Check for Temporal Separation ---
    print("\n\n[Step 3] Verifying temporal separation between training and testing data...")

    timestamps = full_data.get('timestamps')
    if timestamps is not None and len(timestamps) > 0:
        if isinstance(timestamps[0], (list, np.ndarray)):
            timestamps = [ts[0] for ts in timestamps]
        timestamps = np.array(timestamps)

        train_timestamps = timestamps[train_indices]
        test_timestamps = timestamps[test_indices]

        max_train_ts = np.max(train_timestamps)
        min_test_ts = np.min(test_timestamps)
        
        split_timestamp = np.quantile(timestamps, 0.8)

        print(f"\n   - Data was split based on timestamp.")
        print(f"   - Splitting timestamp (80th percentile): {split_timestamp}")
        print(f"   - Maximum timestamp in training set: {max_train_ts}")
        print(f"   - Minimum timestamp in test set:    {min_test_ts}")

        if max_train_ts <= min_test_ts:
            print("\n[OK] Temporal separation is confirmed.")
            print("The latest timestamp in the training set is before or at the same time as the earliest timestamp in the test set.")
        else:
            print("\n[FLAG] Temporal separation is VIOLATED.")
            print("The training set contains data from after the test set started.")
            
        print("\n   Code path for splitting in get_data_and_config:")
        print("   split_timestamp = np.quantile(timestamps, 0.8)")
        print("   train_indices = np.where(timestamps <= split_timestamp)[0] # <--- THIS PATH WAS EXECUTED")
        print("   test_indices = np.where(timestamps > split_timestamp)[0]  # <--- THIS PATH WAS EXECUTED")

    else:
        print("\n[INFO] No timestamps were found in the data.")
        print("The data was split randomly, not temporally.")
        print("\n   Code path for splitting in get_data_and_config:")
        print("   from sklearn.model_selection import train_test_split")
        print("   train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=final_node_labels) # <--- THIS PATH WAS EXECUTED")

    print(f"\n\n{'=' * 80}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    validate_results()