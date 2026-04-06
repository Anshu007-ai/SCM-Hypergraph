import numpy as np

def flatten_data(snapshots, labels):
    """
    Flattens temporal data snapshots for use with traditional ML models.
    
    Args:
        snapshots (list of dicts): List of snapshots, where each snapshot 
                                   contains at least 'node_features'.
        labels (list of np.ndarray): List of label arrays corresponding to snapshots.

    Returns:
        (np.ndarray, np.ndarray): Flattened features and labels.
    """
    # If snapshots is a dictionary (single snapshot), wrap it in a list
    if isinstance(snapshots, dict):
        snapshots = [snapshots]
        labels = [labels]

    all_features = []
    all_labels = []

    # Check if we have temporal snapshots or just a single feature matrix
    if isinstance(snapshots, list) and len(snapshots) > 0 and isinstance(snapshots[0], dict):
        # Case 1: List of snapshot dictionaries
        for i, snap in enumerate(snapshots):
            features = snap['node_features']
            # Ensure features are 2D
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            
            num_nodes = features.shape[0]
            
            # Ensure labels for this snapshot are correctly sized
            snap_labels = labels[i]
            if len(snap_labels) != num_nodes:
                # This can happen if labels are passed for the whole dataset
                # and we are iterating through temporal slices.
                # This logic might need adjustment based on how data is structured.
                # For now, we assume a 1-to-1 mapping is intended.
                print(f"Warning: Mismatch between features ({num_nodes}) and labels ({len(snap_labels)}) in snapshot {i}. Truncating labels.")
                snap_labels = snap_labels[:num_nodes]

            all_features.append(features)
            all_labels.append(snap_labels)
    
    elif isinstance(snapshots, np.ndarray):
        # Case 2: A single numpy array of features
        all_features.append(snapshots)
        all_labels.append(labels)

    else:
        raise TypeError(f"Unsupported type for snapshots: {type(snapshots)}")

    # Concatenate all features and labels
    if not all_features:
        return np.array([]), np.array([])

    X_flat = np.vstack(all_features)
    y_flat = np.concatenate(all_labels)
    
    print(f"  Flattened data shape: {X_flat.shape}")
    print(f"  Flattened labels shape: {y_flat.shape}")
    
    return X_flat, y_flat
