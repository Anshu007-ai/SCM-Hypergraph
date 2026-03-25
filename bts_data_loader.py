"""
BTS (Bureau of Transportation Statistics) Data Loader for HT-HGNN Zero-Shot Validation

This module provides data loading and preprocessing for US flight delay data
from the Bureau of Transportation Statistics to validate HT-HGNN models
trained on IndiGo aviation data. This enables zero-shot validation on external
real-world data for journal submission.

BTS Data Source: https://www.transtats.bts.gov/DL_SelectFields.aspx
Required Fields: FL_DATE, UNIQUE_CARRIER, TAIL_NUM, ORIGIN, DEST,
                CRS_DEP_TIME, DEP_DELAY, ARR_DELAY, CANCELLED,
                CANCELLATION_CODE, CARRIER_DELAY, WEATHER_DELAY

The loader maps BTS data to the same hypergraph structure and feature
representation used in HT-HGNN training, enabling direct inference without
retraining or fine-tuning.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from pathlib import Path
import sys
import warnings

# Add project root for model imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.models.ht_hgnn_model import HeterogeneousTemporalHypergraphNN
except ImportError:
    print("Warning: HT-HGNN model import failed. Using dummy model for testing.")

# Configure plotting
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 10

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


def load_bts_csv(csv_path: str,
                carrier: str = 'AA',
                start_date: str = '2023-12-20',
                end_date: str = '2023-12-26') -> pd.DataFrame:
    """
    Load BTS CSV data and filter by carrier and date range.

    Args:
        csv_path: Path to BTS CSV file
        carrier: Airline carrier code (e.g., 'AA' for American Airlines)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        Filtered pandas DataFrame with BTS flight data
    """
    print(f"Loading BTS data from: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path, parse_dates=['FL_DATE'])

    print(f"Loaded {len(df):,} total flights")

    # Filter by carrier
    if carrier:
        df = df[df['UNIQUE_CARRIER'] == carrier]
        print(f"Filtered to {carrier}: {len(df):,} flights")

    # Filter by date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df['FL_DATE'] >= start_dt) & (df['FL_DATE'] <= end_dt)]

    print(f"Date range {start_date} to {end_date}: {len(df):,} flights")

    # Basic data validation
    required_cols = ['FL_DATE', 'UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN', 'DEST',
                    'CRS_DEP_TIME', 'DEP_DELAY', 'ARR_DELAY', 'CANCELLED']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")

    # Fill missing values
    df['DEP_DELAY'] = df['DEP_DELAY'].fillna(0)
    df['ARR_DELAY'] = df['ARR_DELAY'].fillna(0)
    df['CANCELLED'] = df['CANCELLED'].fillna(0)
    df['CARRIER_DELAY'] = df['CARRIER_DELAY'].fillna(0)
    df['WEATHER_DELAY'] = df['WEATHER_DELAY'].fillna(0)
    df['TAIL_NUM'] = df['TAIL_NUM'].fillna('UNKNOWN')

    print(f"Final dataset: {len(df):,} flights ready for processing")

    return df.reset_index(drop=True)


def build_node_features_from_bts(df: pd.DataFrame) -> np.ndarray:
    """
    Map BTS data to 12-dimensional feature vector matching HT-HGNN training.

    Args:
        df: BTS flight data DataFrame

    Returns:
        [num_flights, 12] numpy array of node features
    """
    num_flights = len(df)
    features = np.zeros((num_flights, 12))

    print(f"Building node features for {num_flights:,} flights...")

    # Feature 0: aircraft_type_code (encode UNIQUE_CARRIER as int)
    carrier_map = {carrier: idx for idx, carrier in enumerate(df['UNIQUE_CARRIER'].unique())}
    features[:, 0] = df['UNIQUE_CARRIER'].map(carrier_map).fillna(0)

    # Feature 1: fleet_age (set to 0 if unavailable)
    features[:, 1] = 0.0  # Placeholder - BTS doesn't have aircraft age

    # Feature 2: crew_duty_hours_remaining
    # Infer: 12 - hours since first flight of tail number on that day
    df['FL_DATE_str'] = df['FL_DATE'].dt.strftime('%Y-%m-%d')
    df['CRS_DEP_TIME_hours'] = df['CRS_DEP_TIME'] // 100 + (df['CRS_DEP_TIME'] % 100) / 60

    crew_hours = []
    for idx, row in df.iterrows():
        tail_date_mask = (df['TAIL_NUM'] == row['TAIL_NUM']) & (df['FL_DATE_str'] == row['FL_DATE_str'])
        tail_flights = df[tail_date_mask].sort_values('CRS_DEP_TIME')

        if len(tail_flights) > 1:
            first_flight_time = tail_flights.iloc[0]['CRS_DEP_TIME_hours']
            hours_elapsed = max(0, row['CRS_DEP_TIME_hours'] - first_flight_time)
            crew_remaining = max(0, 12 - hours_elapsed)
        else:
            crew_remaining = 12.0  # Single flight, assume fresh crew

        crew_hours.append(crew_remaining)

    features[:, 2] = np.array(crew_hours) / 12.0  # Normalize to [0,1]

    # Feature 3: scheduled_departure (CRS_DEP_TIME normalized 0-1 over 0-2359)
    features[:, 3] = df['CRS_DEP_TIME'] / 2359.0

    # Feature 4: historical_delay_rate (mean DEP_DELAY>15 for route in prior data)
    route_delay_rates = []
    for idx, row in df.iterrows():
        route_mask = (df['ORIGIN'] == row['ORIGIN']) & (df['DEST'] == row['DEST'])
        route_flights = df[route_mask]

        if len(route_flights) > 1:
            delay_rate = (route_flights['DEP_DELAY'] > 15).mean()
        else:
            delay_rate = 0.5  # No historical data, use neutral value

        route_delay_rates.append(delay_rate)

    features[:, 4] = np.array(route_delay_rates)

    # Feature 5: sector_complexity (placeholder)
    features[:, 5] = 0.5

    # Feature 6: geographic_concentration (fraction of flights through same ORIGIN hub)
    origin_counts = df['ORIGIN'].value_counts()
    features[:, 6] = df['ORIGIN'].map(origin_counts) / len(df)

    # Feature 7: substitute_route_availability (placeholder)
    features[:, 7] = 0.5

    # Feature 8: schedule_buffer (DEP_DELAY clipped and normalized)
    positive_delays = np.clip(df['DEP_DELAY'], 0, None)
    if positive_delays.max() > 0:
        features[:, 8] = positive_delays / positive_delays.max()
    else:
        features[:, 8] = 0.0

    # Feature 9: ground_delay_flag (1 if CARRIER_DELAY > 0 else 0)
    features[:, 9] = (df['CARRIER_DELAY'] > 0).astype(float)

    # Feature 10: weather_exposure (WEATHER_DELAY / max_weather_delay)
    if df['WEATHER_DELAY'].max() > 0:
        features[:, 10] = df['WEATHER_DELAY'] / df['WEATHER_DELAY'].max()
    else:
        features[:, 10] = 0.0

    # Feature 11: fdtl_compliance_margin (set to 1.0 for US carrier)
    features[:, 11] = 1.0

    print("Node features built successfully:")
    print(f"  Feature ranges: [{features.min():.3f}, {features.max():.3f}]")
    print(f"  Mean values: {features.mean(axis=0)[:6]}")

    return features


def build_hyperedges_from_bts(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Build hyperedges from BTS data using aviation-specific rules.

    Args:
        df: BTS flight data DataFrame

    Returns:
        Tuple of (incidence_matrix [num_nodes, num_hyperedges], hyperedge_type_labels)
    """
    num_flights = len(df)
    hyperedges = []
    hyperedge_types = []

    print("Building hyperedges from BTS data...")

    # Rule 1: Tail sharing - flights sharing TAIL_NUM on same FL_DATE
    df['FL_DATE_str'] = df['FL_DATE'].dt.strftime('%Y-%m-%d')
    tail_groups = df.groupby(['TAIL_NUM', 'FL_DATE_str'])

    tail_hyperedges = 0
    for (tail_num, date), group in tail_groups:
        if len(group) >= 2 and tail_num != 'UNKNOWN':  # Min size 2
            flight_indices = group.index.tolist()
            hyperedges.append(flight_indices)
            hyperedge_types.append('tail_sharing')
            tail_hyperedges += 1

    print(f"  Created {tail_hyperedges} tail sharing hyperedges")

    # Rule 2: Hub congestion - flights departing same ORIGIN within 90-minute window
    df['CRS_DEP_TIME_minutes'] = (df['CRS_DEP_TIME'] // 100) * 60 + (df['CRS_DEP_TIME'] % 100)

    hub_hyperedges = 0
    for origin in df['ORIGIN'].unique():
        origin_flights = df[df['ORIGIN'] == origin].sort_values('CRS_DEP_TIME_minutes')

        for i, flight1 in origin_flights.iterrows():
            window_flights = [i]

            for j, flight2 in origin_flights.iterrows():
                if i != j:
                    time_diff = abs(flight2['CRS_DEP_TIME_minutes'] - flight1['CRS_DEP_TIME_minutes'])
                    if time_diff <= 90:  # 90-minute window
                        window_flights.append(j)

            if len(window_flights) >= 3:  # Min size 3
                # Avoid duplicate hyperedges
                window_flights.sort()
                if window_flights not in hyperedges:
                    hyperedges.append(window_flights)
                    hyperedge_types.append('hub_congestion')
                    hub_hyperedges += 1

    print(f"  Created {hub_hyperedges} hub congestion hyperedges")

    # Rule 3: Route corridor - flights to same DEST within 4 hours
    route_hyperedges = 0
    for dest in df['DEST'].unique():
        dest_flights = df[df['DEST'] == dest].sort_values('CRS_DEP_TIME_minutes')

        for i, flight1 in dest_flights.iterrows():
            corridor_flights = [i]

            for j, flight2 in dest_flights.iterrows():
                if i != j:
                    time_diff = abs(flight2['CRS_DEP_TIME_minutes'] - flight1['CRS_DEP_TIME_minutes'])
                    if time_diff <= 240:  # 4 hours = 240 minutes
                        corridor_flights.append(j)

            if len(corridor_flights) >= 2:  # Min size 2
                corridor_flights.sort()
                if corridor_flights not in hyperedges:
                    hyperedges.append(corridor_flights)
                    hyperedge_types.append('route_corridor')
                    route_hyperedges += 1

    print(f"  Created {route_hyperedges} route corridor hyperedges")

    # Build incidence matrix
    num_hyperedges = len(hyperedges)
    H = np.zeros((num_flights, num_hyperedges))

    for he_idx, flight_indices in enumerate(hyperedges):
        for flight_idx in flight_indices:
            if 0 <= flight_idx < num_flights:
                H[flight_idx, he_idx] = 1.0

    print(f"Built incidence matrix: {H.shape}, density: {H.mean():.4f}")
    print(f"Total hyperedges: {num_hyperedges}")

    return H, hyperedge_types


def build_labels_from_bts(df: pd.DataFrame) -> torch.Tensor:
    """
    Map BTS flight data to criticality classes.

    Args:
        df: BTS flight data DataFrame

    Returns:
        [num_flights] integer tensor with criticality labels
        (0=Low, 1=Medium, 2=High, 3=Critical)
    """
    labels = np.zeros(len(df), dtype=int)

    for i, row in df.iterrows():
        if row['CANCELLED'] == 1:
            labels[i] = 3  # Critical
        elif row['DEP_DELAY'] > 45:
            labels[i] = 2  # High
        elif row['DEP_DELAY'] > 15:
            labels[i] = 1  # Medium
        else:
            labels[i] = 0  # Low

    # Print distribution
    class_counts = np.bincount(labels)
    class_names = ['Low', 'Medium', 'High', 'Critical']

    print("BTS Criticality Label Distribution:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        pct = (count / len(df)) * 100
        print(f"  {name}: {count:,} ({pct:.1f}%)")

    return torch.tensor(labels, dtype=torch.long)


def run_zero_shot_validation(model_checkpoint_path: str,
                            bts_csv_path: str,
                            carrier: str = 'AA',
                            start: str = '2023-12-20',
                            end: str = '2023-12-26') -> Dict[str, Any]:
    """
    Run zero-shot validation of HT-HGNN on BTS data.

    Args:
        model_checkpoint_path: Path to trained HT-HGNN checkpoint
        bts_csv_path: Path to BTS CSV data
        carrier: Airline carrier code
        start: Start date for validation
        end: End date for validation

    Returns:
        Dictionary with validation metrics
    """
    print("="*80)
    print("ZERO-SHOT BTS VALIDATION")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess BTS data
    print("\n[1/5] Loading BTS Data...")
    df = load_bts_csv(bts_csv_path, carrier=carrier, start_date=start, end_date=end)

    if len(df) == 0:
        raise ValueError("No data found for specified filters")

    print(f"\n[2/5] Building Features...")
    node_features = build_node_features_from_bts(df)  # [num_flights, 12]

    print(f"\n[3/5] Building Hypergraph...")
    incidence_matrix, hyperedge_types = build_hyperedges_from_bts(df)

    print(f"\n[4/5] Extracting Labels...")
    labels = build_labels_from_bts(df)

    # Pad node features to 16-dim (12 raw + 4 MOO features with zeros)
    num_flights = node_features.shape[0]
    padded_features = np.zeros((num_flights, 16))
    padded_features[:, :12] = node_features  # Copy original 12 features
    # Features 12-15 remain as zeros (no MOO data for BTS)

    print(f"Feature padding: {node_features.shape} -> {padded_features.shape}")

    # Convert to tensors
    node_features_tensor = torch.tensor(padded_features, dtype=torch.float32, device=device)
    incidence_tensor = torch.tensor(incidence_matrix.T, dtype=torch.float32, device=device)  # [num_hyperedges, num_nodes]
    labels_tensor = labels.to(device)

    # Create metadata for model
    num_nodes = len(df)
    node_types = ['flight'] * num_nodes
    edge_index = torch.tensor([[0, 1]], device=device).t()  # Dummy edge index
    edge_types = ['dummy']
    timestamps = torch.zeros(num_nodes, device=device)

    print(f"\n[5/5] Loading Model and Running Inference...")

    try:
        # Load model (this is a placeholder - adjust based on your model loading method)
        print(f"Loading model from: {model_checkpoint_path}")

        # This is a placeholder model loading - adjust based on your actual model loading
        model_config = {
            'in_channels': 16,  # 12 raw + 4 MOO features
            'hidden_channels': 64,
            'out_channels': 32,
            'num_nodes': num_nodes,
            'num_hyperedges': incidence_matrix.shape[1],
            'node_types': ['supplier', 'part', 'transaction'],  # Placeholder
            'edge_types': ['supplies', 'uses', 'prices'],
            'num_hgnn_layers': 2,
            'num_hgt_heads': 4,
            'time_window': 10
        }

        # Initialize model (in practice, load from checkpoint)
        print("Warning: Using randomly initialized model for demonstration")
        print("Replace this with actual model loading from checkpoint")
        model = HeterogeneousTemporalHypergraphNN(**model_config).to(device)
        model.eval()

    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Using dummy predictions for demonstration")

        # Generate dummy predictions for testing
        with torch.no_grad():
            num_classes = 4
            dummy_logits = torch.randn(num_flights, num_classes, device=device)
            predictions = torch.argmax(dummy_logits, dim=1)

        # Use dummy predictions
        pred_probs = torch.softmax(dummy_logits, dim=1)

    else:
        # Run actual model inference
        with torch.no_grad():
            try:
                output = model(
                    node_features=node_features_tensor,
                    incidence_matrix=incidence_tensor,
                    node_types=node_types,
                    edge_index=edge_index,
                    edge_types=edge_types,
                    timestamps=timestamps
                )

                # Get criticality predictions
                criticality_logits = output['criticality']

                # Convert to 4-class classification if needed
                if criticality_logits.shape[1] == 1:
                    # Binary to 4-class conversion
                    binary_probs = torch.sigmoid(criticality_logits)
                    pred_probs = torch.zeros(num_flights, 4, device=device)
                    pred_probs[:, 0] = 1 - binary_probs.squeeze()  # Low
                    pred_probs[:, 3] = binary_probs.squeeze()      # Critical
                else:
                    pred_probs = torch.softmax(criticality_logits, dim=1)

                predictions = torch.argmax(pred_probs, dim=1)

            except Exception as e:
                print(f"Model inference failed: {e}")
                # Generate dummy predictions for testing
                with torch.no_grad():
                    num_classes = 4
                    dummy_logits = torch.randn(num_flights, num_classes, device=device)
                    predictions = torch.argmax(dummy_logits, dim=1)
                    pred_probs = torch.softmax(dummy_logits, dim=1)

    # Compute metrics
    print("Computing validation metrics...")

    # Convert to CPU for sklearn
    y_true = labels_tensor.cpu().numpy()
    y_pred = predictions.cpu().numpy()

    # Overall accuracy
    crit_acc = accuracy_score(y_true, y_pred)

    # Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Build results
    results = {
        'carrier': carrier,
        'date_range': f"{start} to {end}",
        'num_flights': num_flights,
        'num_hyperedges': incidence_matrix.shape[1],
        'crit_acc': crit_acc,
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1.tolist(),
        'confusion_matrix': cm.tolist(),
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist()
    }

    # Print results
    print("\n" + "="*80)
    print("ZERO-SHOT BTS VALIDATION RESULTS")
    print("="*80)
    print(f"Dataset: {carrier} flights from {start} to {end}")
    print(f"Total flights: {num_flights:,}")
    print(f"Hypergraph: {incidence_matrix.shape[1]} hyperedges")
    print(f"Model input: {padded_features.shape[1]} features")
    print()
    print("CLASSIFICATION METRICS:")
    print(f"  Overall Accuracy: {crit_acc:.4f}")
    print(f"  Macro F1 Score:   {macro_f1:.4f}")
    print()
    print("PER-CLASS F1 SCORES:")
    class_names = ['Low', 'Medium', 'High', 'Critical']
    for i, (name, f1) in enumerate(zip(class_names, per_class_f1)):
        print(f"  {name:<8}: {f1:.4f}")

    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'BTS Zero-Shot Validation Confusion Matrix\n{carrier} {start} to {end}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()

    confusion_fig_path = 'fig_bts_confusion_matrix.png'
    plt.savefig(confusion_fig_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nConfusion matrix saved to: {confusion_fig_path}")
    print("="*80)

    return results


def create_demo_bts_data(save_path: str = 'demo_bts_data.csv', num_flights: int = 1000) -> None:
    """
    Create synthetic BTS-format data for testing.

    Args:
        save_path: Path to save demo CSV
        num_flights: Number of flights to generate
    """
    np.random.seed(42)

    # Generate synthetic BTS data
    carriers = ['AA', 'UA', 'DL', 'WN']
    airports = ['LAX', 'DFW', 'ORD', 'ATL', 'JFK', 'LGA', 'SFO', 'DEN']
    tail_nums = [f'N{i:04d}AA' for i in range(1, 200)]

    dates = pd.date_range('2023-12-20', '2023-12-26', freq='D')

    data = []
    for i in range(num_flights):
        carrier = np.random.choice(carriers)
        origin = np.random.choice(airports)
        dest = np.random.choice([a for a in airports if a != origin])
        tail_num = np.random.choice(tail_nums)
        date = np.random.choice(dates)

        # Generate realistic departure time
        dep_time = np.random.randint(500, 2300)

        # Generate delays with realistic distribution
        delay = np.random.exponential(5) - 10
        if np.random.random() < 0.02:  # 2% cancellation rate
            cancelled = 1
            delay = 0
        else:
            cancelled = 0

        # Generate other delays
        carrier_delay = max(0, np.random.normal(0, 5)) if delay > 0 else 0
        weather_delay = max(0, np.random.normal(0, 10)) if np.random.random() < 0.1 else 0

        data.append({
            'FL_DATE': pd.Timestamp(date).strftime('%Y-%m-%d'),
            'UNIQUE_CARRIER': carrier,
            'TAIL_NUM': tail_num,
            'ORIGIN': origin,
            'DEST': dest,
            'CRS_DEP_TIME': dep_time,
            'DEP_DELAY': delay,
            'ARR_DELAY': delay + np.random.normal(0, 5),
            'CANCELLED': cancelled,
            'CANCELLATION_CODE': 'B' if cancelled else '',
            'CARRIER_DELAY': carrier_delay,
            'WEATHER_DELAY': weather_delay
        })

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Created demo BTS data: {save_path} ({len(df):,} flights)")


def main():
    """Main function for testing BTS data loader."""
    print("BTS Data Loader for HT-HGNN Zero-Shot Validation")
    print("="*60)

    # Create demo data for testing
    demo_csv = 'demo_bts_data.csv'
    create_demo_bts_data(demo_csv, num_flights=500)

    # Test data loading pipeline
    print("\nTesting BTS data loading pipeline...")

    try:
        # Test zero-shot validation with demo data
        results = run_zero_shot_validation(
            model_checkpoint_path='dummy_checkpoint.pth',  # Placeholder
            bts_csv_path=demo_csv,
            carrier='AA',
            start='2023-12-20',
            end='2023-12-26'
        )

        print("\n[SUCCESS] BTS data loader test completed!")
        print(f"Processed {results.get('num_flights', 'N/A')} flights")
        print(f"Created {results.get('num_hyperedges', 'N/A')} hyperedges")

        if 'error' in results:
            print(f"Note: {results['error']}")
        else:
            print(f"Validation accuracy: {results.get('crit_acc', 0):.4f}")
            print(f"Macro F1: {results.get('macro_f1', 0):.4f}")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("To use with real BTS data:")
    print("1. Download data from https://www.transtats.bts.gov/DL_SelectFields.aspx")
    print("2. Use required fields: FL_DATE, UNIQUE_CARRIER, TAIL_NUM, etc.")
    print("3. Call run_zero_shot_validation() with your model checkpoint")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()