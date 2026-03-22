"""
Complete pipeline: Extract model outputs → Generate plot data

Run this script to get all DataFrames ready for Claude Web visualization.
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.extract_model_outputs import ModelOutputExtractor
from scripts.generate_plot_data import generate_all_plot_data


def main():
    print("="*80)
    print("HT-HGNN COMPLETE PIPELINE: Model Outputs → Plot Data")
    print("="*80)

    # Step 1: Extract model outputs
    print("\n[1/2] Extracting model outputs...")
    extractor = ModelOutputExtractor(checkpoint_path='outputs/checkpoints/best.pt')
    extractor.load_model()
    data = extractor.load_data()
    model_outputs = extractor.run_inference(data)

    # Extract test subset (84 nodes for plots)
    test_indices = np.arange(84)
    test_outputs = {
        'y_true': model_outputs['y_true'][test_indices],
        'y_pred': model_outputs['y_pred'][test_indices],
        'y_reg': model_outputs['y_reg'][test_indices],
        'y_reg_pred': model_outputs['y_reg_pred'][test_indices],
        'node_types': [model_outputs['node_types'][i] for i in test_indices],
        'Z': model_outputs['Z'][test_indices],
        'hyperedge_degree': model_outputs['hyperedge_degree'][test_indices],
        'cascade_depth': model_outputs['cascade_depth'][test_indices],
        'pareto_position': model_outputs['pareto_position'][test_indices],
        'classification_accuracy': model_outputs['classification_accuracy'][test_indices]
    }

    # Step 2: Generate plot data
    print("\n[2/2] Generating plot DataFrames...")
    class_names = ['Low', 'Medium', 'High', 'Critical']

    plot_data = generate_all_plot_data(
        y_true=test_outputs['y_true'],
        y_pred=test_outputs['y_pred'],
        y_reg=test_outputs['y_reg'],
        y_reg_pred=test_outputs['y_reg_pred'],
        node_types=test_outputs['node_types'],
        Z=test_outputs['Z'],
        hyperedge_degree=test_outputs['hyperedge_degree'],
        cascade_depth=test_outputs['cascade_depth'],
        pareto_position=test_outputs['pareto_position'],
        classification_accuracy=test_outputs['classification_accuracy'],
        class_names=class_names
    )

    # Step 3: Display results
    print("\n" + "="*80)
    print("PLOT DATA GENERATED - READY FOR CLAUDE WEB")
    print("="*80)

    print("\n" + "-"*80)
    print("Figure 6.5 — CONFUSION MATRIX")
    print("-"*80)
    print(plot_data['confusion_matrix'].to_string())

    print("\n" + "-"*80)
    print("Figure 6.6 — DELAY RMSE PER NODE TYPE")
    print("-"*80)
    print(plot_data['rmse_per_node_type'].to_string(index=False))

    print("\n" + "-"*80)
    print("Figure 6.9 — CASCADE DEPTH VS CENTRALITY (first 10 rows)")
    print("-"*80)
    print(plot_data['cascade_centrality'].head(10).to_string(index=False))
    print(f"... ({len(plot_data['cascade_centrality'])} total rows)")

    print("\n" + "-"*80)
    print("Figure 6.10 — ACCURACY VS PARETO POSITION (first 10 rows)")
    print("-"*80)
    print(plot_data['accuracy_pareto'].head(10).to_string(index=False))
    print(f"... ({len(plot_data['accuracy_pareto'])} total rows)")

    print("\n" + "-"*80)
    print("Figure 6.11 — PHASE 1 EMBEDDING CLUSTERS (first 10 rows)")
    print("-"*80)
    print(plot_data['embedding_clusters'].head(10).to_string(index=False))
    print(f"... ({len(plot_data['embedding_clusters'])} total rows)")

    # Step 4: Save to CSV for easy copy-paste
    output_dir = Path('outputs/plot_data')
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_data['confusion_matrix'].to_csv(output_dir / 'confusion_matrix.csv')
    plot_data['rmse_per_node_type'].to_csv(output_dir / 'rmse_per_node_type.csv', index=False)
    plot_data['cascade_centrality'].to_csv(output_dir / 'cascade_centrality.csv', index=False)
    plot_data['accuracy_pareto'].to_csv(output_dir / 'accuracy_pareto.csv', index=False)
    plot_data['embedding_clusters'].to_csv(output_dir / 'embedding_clusters.csv', index=False)

    print("\n" + "="*80)
    print(f"✓ All plot data saved to: {output_dir}")
    print("="*80)
    print("\nFiles created:")
    print("  • confusion_matrix.csv")
    print("  • rmse_per_node_type.csv")
    print("  • cascade_centrality.csv")
    print("  • accuracy_pareto.csv")
    print("  • embedding_clusters.csv")
    print("\n📋 Copy these CSV files into Claude Web for visualization!")

    return plot_data


if __name__ == "__main__":
    try:
        plot_data = main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
