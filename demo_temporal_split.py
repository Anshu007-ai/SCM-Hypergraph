"""
Temporal Split Demo - Data Leakage Prevention

This script demonstrates the difference between random splitting (which causes
data leakage) and temporal splitting (which prevents leakage) for the IndiGo
aviation disruption dataset.

Run this to see:
1. How random split causes leakage in time series data
2. How temporal split prevents leakage with chronological order
3. Performance difference between approaches
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_split import temporal_split, create_hard_test_set, train_test_split_temporal
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def create_synthetic_crisis_data(n_months=12, hours_per_month=720, n_nodes=84):
    """
    Create synthetic temporal data mimicking the IndiGo crisis scenario.

    Crisis starts in month 10 and affects months 10-12, so a model trained
    on months 1-11 with random split would see future crisis data.
    """
    n_snapshots = n_months * hours_per_month  # 8640 total

    # Base features (stable over time)
    base_features = np.random.randn(n_snapshots, n_nodes, 5)

    # Crisis signal: starts in month 10 (hour 7200)
    crisis_start = 10 * hours_per_month  # Hour 7200
    crisis_intensity = np.zeros(n_snapshots)

    # Gradual crisis buildup in months 10-12
    for t in range(crisis_start, n_snapshots):
        # Exponential crisis growth
        months_into_crisis = (t - crisis_start) / hours_per_month
        crisis_intensity[t] = min(2.0, 0.1 * np.exp(months_into_crisis))

    # Add crisis signal to features (affects all nodes but with different intensity)
    crisis_features = np.zeros((n_snapshots, n_nodes, 5))
    for t in range(n_snapshots):
        crisis_features[t, :, :] = crisis_intensity[t] * np.random.randn(n_nodes, 5)

    # Combined features
    features = base_features + crisis_features

    # Target: criticality index (higher during crisis)
    targets = np.zeros(n_snapshots)
    for t in range(n_snapshots):
        base_target = np.random.normal(0.5, 0.1)  # Base criticality
        crisis_target = crisis_intensity[t] * 0.3  # Crisis boost
        targets[t] = np.clip(base_target + crisis_target, 0, 1)

    # Add some noise
    targets += np.random.normal(0, 0.05, n_snapshots)
    targets = np.clip(targets, 0, 1)

    return features.reshape(n_snapshots, -1), targets, crisis_start


def evaluate_split_method(X, y, crisis_start, method_name, split_func, **kwargs):
    """Evaluate a splitting method and return performance metrics."""

    # Apply split
    if method_name == "Random (leaky)":
        X_train, X_test, y_train, y_test = split_func(X, y, **kwargs)
    else:
        train_data, val_data, test_data = split_func(X, y, **kwargs)
        X_train, y_train = train_data
        X_test, y_test = test_data

    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Check data leakage
    if method_name == "Random (leaky)":
        # Find which training samples are from the crisis period
        train_indices = kwargs.get('train_indices', range(len(X_train)))
        crisis_in_train = sum(1 for i in train_indices if i >= crisis_start)
        leakage_pct = (crisis_in_train / len(train_indices)) * 100
    else:
        leakage_pct = 0  # Temporal split prevents leakage

    return {
        'method': method_name,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'rmse': rmse,
        'leakage_pct': leakage_pct
    }


def main():
    print("="*80)
    print("TEMPORAL VS RANDOM SPLITTING DEMO")
    print("IndiGo Aviation Crisis Dataset")
    print("="*80)

    # Create synthetic data
    print("📊 Creating synthetic IndiGo crisis data...")
    X, y, crisis_start = create_synthetic_crisis_data()

    print(f"   • {len(X):,} snapshots (12 months @ 720 hours/month)")
    print(f"   • {X.shape[1]} features (84 nodes × 5 features each)")
    print(f"   • Crisis starts at snapshot {crisis_start:,} (month 10)")
    print(f"   • Crisis affects snapshots {crisis_start:,}+ (months 10-12)")
    print()

    # Test both methods
    results = []

    # 1. Random split (causes leakage)
    print("🔴 Testing RANDOM SPLIT (sklearn default)...")
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = sklearn_train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Simulate getting train indices for leakage calculation
    indices = list(range(len(X)))
    train_indices, _ = sklearn_train_test_split(indices, test_size=0.2, random_state=42)

    result_random = evaluate_split_method(
        X, y, crisis_start, "Random (leaky)",
        sklearn_train_test_split,
        test_size=0.2, random_state=42, train_indices=train_indices
    )
    results.append(result_random)

    # 2. Temporal split (prevents leakage)
    print("\n🟢 Testing TEMPORAL SPLIT (our method)...")
    result_temporal = evaluate_split_method(
        X, y, crisis_start, "Temporal (safe)",
        temporal_split,
        test_months=2, val_months=1, gap_hours=72
    )
    results.append(result_temporal)

    # Display results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format='%.3f'))

    print("\n📊 KEY INSIGHTS:")
    print(f"   • Random split: {result_random['leakage_pct']:.1f}% training data from crisis period")
    print(f"   • Temporal split: {result_temporal['leakage_pct']:.1f}% training data from crisis period")
    print()
    print(f"   • Random RMSE: {result_random['rmse']:.3f}")
    print(f"   • Temporal RMSE: {result_temporal['rmse']:.3f}")
    print()

    if result_random['rmse'] < result_temporal['rmse']:
        print("❌ Random split seems 'better' but THIS IS MISLEADING!")
        print("   Lower error is due to DATA LEAKAGE, not better modeling")
        print("   The model has seen future crisis data during training")
    else:
        print("✅ Temporal split provides honest performance estimate")
        print("   Higher error reflects true difficulty of crisis prediction")

    print("\n🎯 RECOMMENDED ACTION:")
    print("   • ALWAYS use temporal_split() for time series data")
    print("   • NEVER use sklearn train_test_split for temporal data")
    print("   • Accept higher but honest error rates")

    print("\n" + "="*80)
    print("TEMPORAL SPLIT DEMO COMPLETE")
    print("="*80)

    # Test hard test set creation
    print("\n🔥 Testing hard test set creation...")
    train_data, val_data, test_data = temporal_split(X, y)
    test_X, test_y = test_data

    # Create synthetic multi-node labels for hard test demo
    test_y_multinode = np.random.randint(0, 4, size=(len(test_y), 10))  # 10 nodes, 4 classes

    hard_X, hard_y, stats = create_hard_test_set(test_X, test_y_multinode)

    print(f"\n📈 Hard test set stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()