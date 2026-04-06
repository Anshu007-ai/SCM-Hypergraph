"""
Temporal Dataset Splitting for HT-HGNN

Replaces random train/test splits with strict temporal splitting to prevent data leakage
in time series prediction tasks. Designed specifically for the IndiGo aviation 2025
dataset with 84 nodes, 18 hyperedges, and ~8640 hourly snapshots over 12 months.

Key features:
- Temporal split with 72-hour gaps between splits to prevent crisis bleed-through
- Hard test set creation focusing on criticality class transition boundaries
- Compatible with existing DataLoader/Dataset classes

Usage:
    from src.data.dataset_split import temporal_split, create_hard_test_set

    # Replace random split
    train_data, val_data, test_data = temporal_split(snapshots, labels)

    # Create challenging test set
    hard_test = create_hard_test_set(test_snapshots, test_labels)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def temporal_split(
    snapshots: Union[List, np.ndarray, pd.DataFrame],
    labels: Union[List, np.ndarray, pd.DataFrame],
    test_months: int = 2,
    val_months: int = 1,
    gap_hours: int = 72,
    hours_per_month: int = 720  # 30 days * 24 hours
) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any]]:
    """
    Split time series data with strict temporal ordering to prevent data leakage.

    Splits the LAST 2 months as test, months 9-10 as validation, rest as training.
    Adds 72-hour gaps between splits so the model cannot see crisis buildup.

    Timeline:
        Months 1-8: Train (5760 snapshots)
        [72h gap]
        Months 9-10: Validation (1440 snapshots)
        [72h gap]
        Months 11-12: Test (1440 snapshots)

    Args:
        snapshots: Chronologically ordered list/array of snapshots
        labels: Corresponding labels for each snapshot
        test_months: Number of final months for test set (default: 2)
        val_months: Number of months before test for validation (default: 1)
        gap_hours: Hours gap between train/val and val/test (default: 72)
        hours_per_month: Average hours per month for calculation (default: 720)

    Returns:
        Tuple of ((train_snaps, train_labels), (val_snaps, val_labels), (test_snaps, test_labels))

    Example:
        # Replace sklearn random split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # With temporal split
        train_data, val_data, test_data = temporal_split(snapshots, labels)
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
    """

    # Convert inputs to numpy arrays for consistent indexing
    if isinstance(snapshots, pd.DataFrame):
        snapshots_arr = snapshots.values
    else:
        snapshots_arr = np.array(snapshots)

    if isinstance(labels, pd.DataFrame):
        labels_arr = labels.values
    else:
        labels_arr = np.array(labels)

    total_snapshots = len(snapshots_arr)

    # Calculate split indices
    test_size = test_months * hours_per_month
    val_size = val_months * hours_per_month

    # Test set: final test_months (e.g., months 11-12)
    test_start_idx = total_snapshots - test_size

    # Validation set: val_months before test, with gap
    val_end_idx = test_start_idx - gap_hours
    val_start_idx = val_end_idx - val_size

    # Training set: everything before validation, with gap
    train_end_idx = val_start_idx - gap_hours

    # Ensure valid indices
    if train_end_idx < 0 or val_start_idx < 0 or test_start_idx < 0:
        raise ValueError(
            f"Dataset too small for temporal split. Need at least "
            f"{(test_months + val_months) * hours_per_month + 2 * gap_hours} snapshots, "
            f"got {total_snapshots}"
        )

    # Extract splits
    train_snaps = snapshots_arr[:train_end_idx]
    train_labels = labels_arr[:train_end_idx]

    val_snaps = snapshots_arr[val_start_idx:val_end_idx]
    val_labels = labels_arr[val_start_idx:val_end_idx]

    test_snaps = snapshots_arr[test_start_idx:]
    test_labels = labels_arr[test_start_idx:]

    # Print split information
    print("=" * 80)
    print("TEMPORAL DATA SPLIT - PREVENTING LEAKAGE")
    print("=" * 80)
    print(f"[DATA] Total snapshots: {total_snapshots:,}")
    print(f"[TIME] Gap between splits: {gap_hours} hours")
    print()
    print(f"[TRAIN] TRAIN:  indices [0:{train_end_idx}] = {len(train_snaps):,} snapshots ({len(train_snaps)/total_snapshots*100:.1f}%)")
    print(f"          Covers ~months 1-{int(train_end_idx/hours_per_month)}")
    print()
    print(f"[VAL] VAL:    indices [{val_start_idx}:{val_end_idx}] = {len(val_snaps):,} snapshots ({len(val_snaps)/total_snapshots*100:.1f}%)")
    print(f"          Covers ~months {int(val_start_idx/hours_per_month)+1}-{int(val_end_idx/hours_per_month)}")
    print()
    print(f"[TEST] TEST:   indices [{test_start_idx}:{total_snapshots}] = {len(test_snaps):,} snapshots ({len(test_snaps)/total_snapshots*100:.1f}%)")
    print(f"          Covers ~months {int(test_start_idx/hours_per_month)+1}-12")
    print()
    print(f"[OK] NO LEAKAGE: {gap_hours}h gaps prevent crisis buildup bleeding between sets")
    print("=" * 80)

    return (train_snaps, train_labels), (val_snaps, val_labels), (test_snaps, test_labels)


def create_hard_test_set(
    snapshots: Union[List, np.ndarray, pd.DataFrame],
    labels: Union[List, np.ndarray, pd.DataFrame],
    hard_ratio: float = 0.7,
    easy_ratio: float = 0.3,
    criticality_classes: int = 4
) -> Tuple[Any, Any, Dict[str, int]]:
    """
    Create a challenging test set with 70% "hard" boundary cases and 30% "easy" stable cases.

    "Hard" samples are where a node changed criticality class between snapshot t-1 and t.
    "Easy" samples are where the class remained stable.

    This tests the model's ability to predict transitions vs steady states.

    Args:
        snapshots: Test snapshots (chronologically ordered)
        labels: Test labels (N_snapshots, N_nodes) or (N_snapshots,)
        hard_ratio: Proportion of hard transition cases in final test set
        easy_ratio: Proportion of easy stable cases in final test set
        criticality_classes: Number of criticality classes (default: 4 for Low/Medium/High/Critical)

    Returns:
        Tuple of (hard_test_snapshots, hard_test_labels, stats_dict)

    Example:
        test_snaps, test_labels = test_data
        hard_test_snaps, hard_test_labels, stats = create_hard_test_set(test_snaps, test_labels)
    """

    # Convert to numpy
    if isinstance(snapshots, pd.DataFrame):
        snapshots_arr = snapshots.values
    else:
        snapshots_arr = np.array(snapshots)

    if isinstance(labels, pd.DataFrame):
        labels_arr = labels.values
    else:
        labels_arr = np.array(labels)

    # Ensure we have at least 2 snapshots for transition detection
    if len(snapshots_arr) < 2:
        logger.warning("Need at least 2 snapshots for hard/easy split. Returning original data.")
        return snapshots_arr, labels_arr, {"hard_samples": 0, "easy_samples": len(snapshots_arr)}

    # Handle different label shapes
    if labels_arr.ndim == 1:
        # Single node or already flattened
        n_snapshots = len(labels_arr)
        n_nodes = 1
        labels_reshaped = labels_arr.reshape(-1, 1)
    else:
        # Multi-node: (n_snapshots, n_nodes)
        n_snapshots, n_nodes = labels_arr.shape
        labels_reshaped = labels_arr

    hard_indices = []
    easy_indices = []

    # Check each snapshot from t=1 onwards (need t-1 for comparison)
    for t in range(1, n_snapshots):
        snapshot_is_hard = False

        # Check if ANY node changed criticality class between t-1 and t
        for node in range(n_nodes):
            prev_class = labels_reshaped[t-1, node]
            curr_class = labels_reshaped[t, node]

            # Class transition = hard sample
            if prev_class != curr_class:
                snapshot_is_hard = True
                break

        if snapshot_is_hard:
            hard_indices.append(t)
        else:
            easy_indices.append(t)

    n_hard_total = len(hard_indices)
    n_easy_total = len(easy_indices)

    print("=" * 80)
    print("HARD TEST SET CREATION")
    print("=" * 80)
    print(f"[STATS] Found {n_hard_total:,} HARD samples (criticality transitions)")
    print(f"[DATA] Found {n_easy_total:,} EASY samples (stable states)")
    print()

    # Calculate target sizes
    total_target = int(len(snapshots_arr) * 0.8)  # Use 80% of test set
    hard_target = int(total_target * hard_ratio)
    easy_target = int(total_target * easy_ratio)

    # Sample indices
    if hard_target > n_hard_total:
        logger.warning(f"Requested {hard_target} hard samples, but only {n_hard_total} available. Using all hard samples.")
        hard_selected = hard_indices
    else:
        hard_selected = np.random.choice(hard_indices, size=hard_target, replace=False).tolist()

    if easy_target > n_easy_total:
        logger.warning(f"Requested {easy_target} easy samples, but only {n_easy_total} available. Using all easy samples.")
        easy_selected = easy_indices
    else:
        easy_selected = np.random.choice(easy_indices, size=easy_target, replace=False).tolist()

    # Combine and sort to maintain temporal order
    selected_indices = sorted(hard_selected + easy_selected)

    # Extract final test set
    final_snapshots = snapshots_arr[selected_indices]
    final_labels = labels_arr[selected_indices]

    stats = {
        "hard_samples": len(hard_selected),
        "easy_samples": len(easy_selected),
        "total_samples": len(selected_indices),
        "hard_ratio_actual": len(hard_selected) / len(selected_indices),
        "easy_ratio_actual": len(easy_selected) / len(selected_indices),
        "hard_available": n_hard_total,
        "easy_available": n_easy_total
    }

    print(f"[TARGET] TARGET: {hard_target} hard + {easy_target} easy = {hard_target + easy_target} total")
    print(f"[OK] ACTUAL: {stats['hard_samples']} hard + {stats['easy_samples']} easy = {stats['total_samples']} total")
    print(f"[DATA] RATIO:  {stats['hard_ratio_actual']:.1%} hard, {stats['easy_ratio_actual']:.1%} easy")
    print()
    print("[BRAIN] This challenging test set will reveal if the model can:")
    print("   • Predict crisis transitions (hard samples)")
    print("   • Maintain stability predictions (easy samples)")
    print("=" * 80)

    return final_snapshots, final_labels, stats


class TemporalDataLoader:
    """
    Drop-in replacement for existing DataLoader classes that uses temporal splitting.

    Maintains the same interface as existing loaders while ensuring no data leakage.
    """

    def __init__(self,
                 snapshots: Any,
                 labels: Any,
                 batch_size: int = 32,
                 shuffle: bool = False,  # Disabled for temporal data
                 test_months: int = 2,
                 val_months: int = 1,
                 gap_hours: int = 72):
        """
        Initialize temporal data loader.

        Args:
            snapshots: Chronologically ordered snapshots
            labels: Corresponding labels
            batch_size: Batch size for model training
            shuffle: Ignored - temporal order must be preserved
            test_months: Months for test set (default: 2)
            val_months: Months for validation (default: 1)
            gap_hours: Gap between splits (default: 72)
        """
        self.snapshots = snapshots
        self.labels = labels
        self.batch_size = batch_size

        if shuffle:
            logger.warning("Shuffle disabled for temporal data to prevent leakage")

        # Perform temporal split
        self.train_data, self.val_data, self.test_data = temporal_split(
            snapshots, labels, test_months, val_months, gap_hours
        )

    def get_train_loader(self):
        """Get training data"""
        return self.train_data

    def get_val_loader(self):
        """Get validation data"""
        return self.val_data

    def get_test_loader(self):
        """Get test data"""
        return self.test_data

    def get_hard_test_loader(self, hard_ratio: float = 0.7):
        """Get challenging test set with hard/easy samples"""
        test_snaps, test_labels = self.test_data
        return create_hard_test_set(test_snaps, test_labels, hard_ratio)


# Compatibility function to replace sklearn train_test_split
def train_test_split_temporal(
    X: Any,
    y: Any,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    **kwargs
) -> Tuple[Any, Any, Any, Any]:
    """
    Drop-in replacement for sklearn.model_selection.train_test_split that uses temporal splitting.

    This function maintains the same interface as sklearn but prevents data leakage
    by respecting temporal order instead of random shuffling.

    Args:
        X: Features/snapshots (chronologically ordered)
        y: Labels (corresponding to X)
        test_size: Proportion of data for test set
        random_state: Ignored (temporal order preserved)
        **kwargs: Additional sklearn arguments (mostly ignored)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) - same as sklearn

    Example:
        # Replace this:
        # from sklearn.model_selection import train_test_split

        # With this:
        # from src.data.dataset_split import train_test_split_temporal as train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    """

    if random_state is not None:
        logger.warning("random_state ignored in temporal split to preserve chronological order")

    # Convert test_size to months (assuming test_size=0.2 means ~2 months of 12)
    test_months = max(1, int(12 * test_size))
    val_months = 1  # Fixed validation size

    train_data, val_data, test_data = temporal_split(X, y, test_months, val_months)

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # sklearn interface returns 4-tuple (no validation)
    # Combine train + val to match expected interface
    if isinstance(X_train, np.ndarray) and isinstance(X_val, np.ndarray):
        X_train_combined = np.concatenate([X_train, X_val])
        y_train_combined = np.concatenate([y_train, y_val])
    else:
        X_train_combined = list(X_train) + list(X_val)
        y_train_combined = list(y_train) + list(y_val)

    logger.info(f"Temporal split: {len(X_train_combined)} train, {len(X_test)} test")

    return X_train_combined, X_test, y_train_combined, y_test


if __name__ == "__main__":
    print("=" * 80)
    print("TEMPORAL DATASET SPLITTING MODULE")
    print("HT-HGNN Data Leakage Prevention")
    print("=" * 80)
    print()
    print("[TARGET] Purpose: Replace random train_test_split with temporal splits")
    print("📅 Dataset: IndiGo Aviation 2025 (84 nodes, 18 hyperedges, ~8640 snapshots)")
    print("[TIME] Method:  Strict chronological order with 72-hour gaps")
    print()
    print("Functions:")
    print("  • temporal_split() - Main temporal splitting function")
    print("  • create_hard_test_set() - Create challenging test set")
    print("  • TemporalDataLoader - Drop-in DataLoader replacement")
    print("  • train_test_split_temporal() - sklearn compatibility")
    print()
    print("Timeline:")
    print("  Months 1-8:   Training (5760 snapshots)")
    print("  [72h gap]")
    print("  Months 9-10:  Validation (1440 snapshots)")
    print("  [72h gap]")
    print("  Months 11-12: Test (1440 snapshots)")
    print()

    # Demo with synthetic data
    print("[TEST] DEMO: Synthetic 12-month hourly dataset")
    np.random.seed(42)

    # Create synthetic temporal data (8640 snapshots = 12 months * 720 hours)
    n_snapshots = 8640
    n_nodes = 84
    n_features = 10

    # Synthetic snapshots and labels with temporal trends
    snapshots = np.random.randn(n_snapshots, n_nodes, n_features)

    # Add temporal trend to make transitions more realistic
    time_trend = np.linspace(0, 1, n_snapshots).reshape(-1, 1, 1)
    snapshots = snapshots + time_trend * np.random.randn(1, n_nodes, n_features)

    # Synthetic criticality labels (4 classes) with some transitions
    labels = np.zeros((n_snapshots, n_nodes), dtype=int)
    for node in range(n_nodes):
        # Start with random class
        labels[0, node] = np.random.randint(0, 4)

        # Add transitions over time
        for t in range(1, n_snapshots):
            if np.random.random() < 0.001:  # 0.1% chance of transition per hour
                labels[t, node] = np.random.randint(0, 4)
            else:
                labels[t, node] = labels[t-1, node]  # Stay same

    # Test temporal split
    print("\n[DATA] Running temporal_split()...")
    train_data, val_data, test_data = temporal_split(snapshots, labels)

    # Test hard test set
    print("\n[TARGET] Running create_hard_test_set()...")
    test_snaps, test_labels = test_data
    hard_snaps, hard_labels, stats = create_hard_test_set(test_snaps, test_labels)

    print(f"\n[OK] Demo completed successfully!")
    print(f"   Train: {len(train_data[0]):,} snapshots")
    print(f"   Val:   {len(val_data[0]):,} snapshots")
    print(f"   Test:  {len(test_data[0]):,} snapshots")
    print(f"   Hard:  {len(hard_snaps):,} challenging samples")

    print("\n🚀 Module ready for use!")
    print("=" * 80)