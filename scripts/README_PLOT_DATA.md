# HT-HGNN Model Output Extraction & Plot Data Generation

This directory contains scripts to extract outputs from your trained HT-HGNN model and generate plot data for your thesis figures.

## 📁 Files

- **`extract_model_outputs.py`** - Loads trained model and runs inference
- **`generate_plot_data.py`** - Generates DataFrames for plots (modular functions)
- **`run_full_pipeline.py`** - Complete end-to-end pipeline (run this!)

## 🚀 Quick Start

### Step 1: Run the Pipeline

```bash
python scripts/run_full_pipeline.py
```

This will:
1. Load your trained model from `outputs/checkpoints/best.pt`
2. Run inference on all nodes
3. Extract test subset (84 nodes)
4. Generate plot DataFrames
5. Save CSV files to `outputs/plot_data/`

### Step 2: View Results

The script will print all plot data to console and save to CSV:

```
outputs/plot_data/
├── confusion_matrix.csv          → Figure 6.5
├── rmse_per_node_type.csv        → Figure 6.6
├── cascade_centrality.csv        → Figure 6.9
├── accuracy_pareto.csv           → Figure 6.10
└── embedding_clusters.csv        → Figure 6.11
```

### Step 3: Copy to Claude Web

Open the CSV files and copy-paste the data into Claude Web for visualization.

## 📊 Plots Generated

| Figure | Description | Output File |
|--------|-------------|-------------|
| 6.5 | Confusion Matrix (4-class) | `confusion_matrix.csv` |
| 6.6 | Delay RMSE per Node Type | `rmse_per_node_type.csv` |
| 6.9 | Cascade Depth vs Centrality | `cascade_centrality.csv` |
| 6.10 | Accuracy vs Pareto Position | `accuracy_pareto.csv` |
| 6.11 | t-SNE Embedding Clusters | `embedding_clusters.csv` |

## 🔧 Advanced Usage

### Extract Model Outputs Only

```python
from scripts.extract_model_outputs import ModelOutputExtractor

extractor = ModelOutputExtractor('outputs/checkpoints/best.pt')
extractor.load_model()
data = extractor.load_data()
outputs = extractor.run_inference(data)
```

### Generate Plot Data Only

```python
from scripts.generate_plot_data import generate_all_plot_data

plot_data = generate_all_plot_data(
    y_true=y_true,
    y_pred=y_pred,
    y_reg=y_reg,
    y_reg_pred=y_reg_pred,
    node_types=node_types,
    Z=embeddings,
    hyperedge_degree=hyperedge_degree,
    cascade_depth=cascade_depth,
    pareto_position=pareto_position,
    classification_accuracy=classification_accuracy,
    class_names=['Low', 'Medium', 'High', 'Critical']
)
```

### Individual Plot Functions

```python
from scripts.generate_plot_data import (
    compute_confusion_matrix,
    compute_rmse_per_node_type,
    create_cascade_centrality_data,
    create_accuracy_pareto_data,
    create_embedding_clusters
)

# Confusion matrix only
cm_df = compute_confusion_matrix(y_true, y_pred, class_names)

# RMSE by node type
rmse_df = compute_rmse_per_node_type(y_reg, y_reg_pred, node_types)

# And so on...
```

## ⚠️ Important Notes

### Synthetic vs Real Data

The current implementation includes some **synthetic/placeholder data** for:
- `y_true` - Uses y_pred as proxy (replace with actual test labels)
- `y_reg` - Uses y_reg_pred with noise (replace with actual delay targets)
- `pareto_position` - Random values (requires MOO optimization)
- `classification_accuracy` - Random values (requires cross-validation or ensemble)

### To Use Real Data:

1. **Ground Truth Labels**: Load actual test set labels
   ```python
   # In extract_model_outputs.py, line ~165
   y_true = load_actual_test_labels()  # Your implementation
   ```

2. **Regression Targets**: Load actual delay values
   ```python
   # In extract_model_outputs.py, line ~172
   y_reg = load_actual_delay_targets()  # Your implementation
   ```

3. **Pareto Positions**: Run MOO and extract positions
   ```python
   # After training, run Pareto optimization
   pareto_position = run_pareto_optimization(model, data)
   ```

4. **Per-Node Accuracy**: Use k-fold CV or test set accuracy
   ```python
   classification_accuracy = compute_per_node_accuracy(model, test_loader)
   ```

## 🛠️ Dependencies

```
torch
numpy
pandas
scikit-learn
```

## 📝 Model Mapping

Your HT-HGNN model outputs are mapped to plot requirements as follows:

| Model Output | Plot Requirement | Transformation |
|--------------|------------------|----------------|
| `criticality` | 4-class classification | Binning: [0,1] → {0,1,2,3} |
| `change_pred` | Delay regression | Scale by 100× |
| `cascade_scores` | Cascade depth | Scale to [0-10] |
| `incidence_matrix` | Hyperedge degree | Sum per node |
| `embeddings` | t-SNE clusters | Direct use (64D → 2D) |

## 🐛 Troubleshooting

**Issue**: Checkpoint not found
```
Solution: Check that outputs/checkpoints/best.pt exists
Run training first: python train_ht_hgnn_safe_v2.py
```

**Issue**: CSV files not found
```
Solution: Ensure data files exist:
- outputs/datasets/features.csv
- outputs/datasets/hci_labels.csv
- outputs/datasets/incidence.csv
```

**Issue**: CUDA out of memory
```
Solution: The script runs in eval mode (no gradients)
If still OOM, reduce batch size or use CPU:
device = torch.device('cpu')
```

## 📧 Questions?

Check the inline comments in each script for detailed explanations.
