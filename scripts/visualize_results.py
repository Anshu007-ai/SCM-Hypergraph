"""
HT-HGNN Model Outputs: Data Generation + Visualization

This script provides:
1. DataFrame generation for each plot
2. Publication-quality plotting functions
3. Modular, reusable code

Usage:
    python scripts/visualize_results.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


# ============================================================================
# 1. CONFUSION MATRIX
# ============================================================================

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute confusion matrix for 4-class classification.

    Args:
        y_true: True class labels (N,)
        y_pred: Predicted class labels (N,)
        class_names: Names for each class

    Returns:
        pd.DataFrame: Confusion matrix with labeled rows/columns
    """
    if class_names is None:
        class_names = ['Low', 'Medium', 'High', 'Critical']

    # Ensure we get a 4x4 matrix even if some classes are missing
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    return cm_df


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.

    Args:
        cm_df: Confusion matrix DataFrame
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        plt.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap with annotations
    sns.heatmap(
        cm_df,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Confusion Matrix: 4-Class Criticality Prediction',
                 fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[SAVED] {save_path}")

    if show:
        plt.show()

    return fig


# ============================================================================
# 2. RMSE PER NODE TYPE
# ============================================================================

def compute_rmse_per_node_type(
    y_reg: np.ndarray,
    y_reg_pred: np.ndarray,
    node_types: List[str]
) -> pd.DataFrame:
    """
    Compute RMSE per node type (aircraft, sector, airport).

    Args:
        y_reg: True delay values (N,)
        y_reg_pred: Predicted delay values (N,)
        node_types: Node type for each sample

    Returns:
        pd.DataFrame: RMSE for each node type
    """
    df = pd.DataFrame({
        'y_true': y_reg,
        'y_pred': y_reg_pred,
        'node_type': node_types
    })

    rmse_results = []
    for node_type in ['aircraft', 'sector', 'airport']:
        subset = df[df['node_type'] == node_type]
        if len(subset) > 0:
            rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
            rmse_results.append({
                'node_type': node_type,
                'rmse': rmse,
                'count': len(subset)
            })

    return pd.DataFrame(rmse_results)


def plot_rmse_per_node_type(
    rmse_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot RMSE per node type as bar chart.

    Args:
        rmse_df: DataFrame with columns ['node_type', 'rmse', 'count']
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        plt.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar plot
    bars = ax.bar(
        rmse_df['node_type'],
        rmse_df['rmse'],
        color=['#3498db', '#e74c3c', '#2ecc71'],
        edgecolor='black',
        linewidth=1.2
    )

    # Add value labels on bars
    for bar, rmse, count in zip(bars, rmse_df['rmse'], rmse_df['count']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.5,
            f'{rmse:.2f}\n(n={count})',
            ha='center',
            va='bottom',
            fontsize=9
        )

    ax.set_xlabel('Node Type', fontweight='bold')
    ax.set_ylabel('RMSE (Delay Prediction)', fontweight='bold')
    ax.set_title('Delay RMSE by Node Type', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[SAVED] {save_path}")

    if show:
        plt.show()

    return fig


# ============================================================================
# 3. CASCADE DEPTH VS CENTRALITY
# ============================================================================

def create_cascade_centrality_data(
    cascade_depth: np.ndarray,
    hyperedge_degree: np.ndarray
) -> pd.DataFrame:
    """
    Create DataFrame for cascade depth vs centrality scatter plot.

    Args:
        cascade_depth: Cascade depth for each node (84,)
        hyperedge_degree: Hyperedge degree for each node (84,)

    Returns:
        pd.DataFrame: columns = ['cascade_depth', 'hyperedge_degree']
    """
    return pd.DataFrame({
        'cascade_depth': cascade_depth,
        'hyperedge_degree': hyperedge_degree
    })


def plot_cascade_centrality(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot cascade depth vs hyperedge degree scatter plot.

    Args:
        df: DataFrame with columns ['cascade_depth', 'hyperedge_degree']
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        plt.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    scatter = ax.scatter(
        df['hyperedge_degree'],
        df['cascade_depth'],
        c=df['cascade_depth'],
        cmap='viridis',
        s=80,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cascade Depth', rotation=270, labelpad=20, fontweight='bold')

    # Add trend line
    z = np.polyfit(df['hyperedge_degree'], df['cascade_depth'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['hyperedge_degree'].min(), df['hyperedge_degree'].max(), 100)
    ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8, label=f'Trend (R²={np.corrcoef(df["hyperedge_degree"], df["cascade_depth"])[0,1]**2:.3f})')

    ax.set_xlabel('Hyperedge Degree (Centrality)', fontweight='bold')
    ax.set_ylabel('Cascade Depth', fontweight='bold')
    ax.set_title('Cascade Depth vs Network Centrality', fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[SAVED] {save_path}")

    if show:
        plt.show()

    return fig


# ============================================================================
# 4. ACCURACY VS PARETO POSITION
# ============================================================================

def create_accuracy_pareto_data(
    pareto_position: np.ndarray,
    classification_accuracy: np.ndarray
) -> pd.DataFrame:
    """
    Create DataFrame for accuracy vs Pareto position.

    Args:
        pareto_position: Position on Pareto front (N,)
        classification_accuracy: Per-node classification accuracy (N,)

    Returns:
        pd.DataFrame: columns = ['pareto_position', 'accuracy']
    """
    return pd.DataFrame({
        'pareto_position': pareto_position,
        'accuracy': classification_accuracy
    })


def plot_accuracy_pareto(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot accuracy vs Pareto position.

    Args:
        df: DataFrame with columns ['pareto_position', 'accuracy']
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        plt.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Sort by pareto position for cleaner line plot
    df_sorted = df.sort_values('pareto_position')

    # Scatter + line plot
    ax.scatter(
        df_sorted['pareto_position'],
        df_sorted['accuracy'],
        c=df_sorted['accuracy'],
        cmap='RdYlGn',
        s=60,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    ax.plot(
        df_sorted['pareto_position'],
        df_sorted['accuracy'],
        color='navy',
        alpha=0.3,
        linewidth=1,
        label='Accuracy Trend'
    )

    ax.set_xlabel('Pareto Position (MOO Front)', fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontweight='bold')
    ax.set_title('Classification Accuracy vs Pareto Position', fontweight='bold', pad=20)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[SAVED] {save_path}")

    if show:
        plt.show()

    return fig


# ============================================================================
# 5. EMBEDDING CLUSTERS (t-SNE)
# ============================================================================

def create_embedding_clusters(
    Z: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    random_state: int = 42,
    perplexity: int = 30
) -> pd.DataFrame:
    """
    Run t-SNE on node embeddings and create 2D visualization data.

    Args:
        Z: Node embeddings (84, 256)
        y_pred: Predicted class labels (84,)
        class_names: Names for each class
        random_state: Random seed for reproducibility
        perplexity: t-SNE perplexity parameter

    Returns:
        pd.DataFrame: columns = ['x', 'y', 'label', 'label_name']
    """
    if class_names is None:
        class_names = ['Low', 'Medium', 'High', 'Critical']

    # Adjust perplexity if needed
    n_samples = Z.shape[0]
    if perplexity >= n_samples:
        perplexity = max(5, n_samples // 3)

    # Run t-SNE
    print(f"Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    Z_2d = tsne.fit_transform(Z)

    # Create DataFrame
    df = pd.DataFrame({
        'x': Z_2d[:, 0],
        'y': Z_2d[:, 1],
        'label': y_pred,
        'label_name': [class_names[int(l)] for l in y_pred]
    })

    print("[OK] t-SNE complete")
    return df


def plot_embedding_clusters(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot t-SNE embedding clusters with color-coded classes.

    Args:
        df: DataFrame with columns ['x', 'y', 'label', 'label_name']
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        plt.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color palette for classes
    class_colors = {
        'Low': '#2ecc71',
        'Medium': '#f39c12',
        'High': '#e67e22',
        'Critical': '#e74c3c'
    }

    # Plot each class separately for legend
    for label_name in df['label_name'].unique():
        subset = df[df['label_name'] == label_name]
        ax.scatter(
            subset['x'],
            subset['y'],
            label=label_name,
            color=class_colors.get(label_name, 'gray'),
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax.set_title('Phase 1 Embedding Clusters (t-SNE)', fontweight='bold', pad=20)
    ax.legend(title='Criticality Class', title_fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[SAVED] {save_path}")

    if show:
        plt.show()

    return fig


# ============================================================================
# 6. ALL-IN-ONE PIPELINE
# ============================================================================

def generate_all_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_reg: np.ndarray,
    y_reg_pred: np.ndarray,
    node_types: List[str],
    Z: np.ndarray,
    hyperedge_degree: np.ndarray,
    cascade_depth: np.ndarray,
    pareto_position: np.ndarray,
    classification_accuracy: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_dir: str = 'outputs/figures',
    show_plots: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Generate all plots and DataFrames.

    Args:
        All model outputs (see individual function docstrings)
        class_names: Names for classification classes
        output_dir: Directory to save figures
        show_plots: Whether to display plots interactively

    Returns:
        dict: All DataFrames generated
    """
    if class_names is None:
        class_names = ['Low', 'Medium', 'High', 'Critical']

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING ALL PLOTS AND DATA")
    print("="*80)

    results = {}

    # 1. Confusion Matrix
    print("\n[1/5] Confusion Matrix...")
    cm_df = compute_confusion_matrix(y_true, y_pred, class_names)
    plot_confusion_matrix(cm_df, save_path=str(output_path / 'confusion_matrix.png'), show=show_plots)
    results['confusion_matrix'] = cm_df

    # 2. RMSE per Node Type
    print("\n[2/5] RMSE per Node Type...")
    rmse_df = compute_rmse_per_node_type(y_reg, y_reg_pred, node_types)
    plot_rmse_per_node_type(rmse_df, save_path=str(output_path / 'rmse_per_node_type.png'), show=show_plots)
    results['rmse_per_node_type'] = rmse_df

    # 3. Cascade Depth vs Centrality
    print("\n[3/5] Cascade Depth vs Centrality...")
    cascade_df = create_cascade_centrality_data(cascade_depth, hyperedge_degree)
    plot_cascade_centrality(cascade_df, save_path=str(output_path / 'cascade_centrality.png'), show=show_plots)
    results['cascade_centrality'] = cascade_df

    # 4. Accuracy vs Pareto Position
    print("\n[4/5] Accuracy vs Pareto Position...")
    pareto_df = create_accuracy_pareto_data(pareto_position, classification_accuracy)
    plot_accuracy_pareto(pareto_df, save_path=str(output_path / 'accuracy_pareto.png'), show=show_plots)
    results['accuracy_pareto'] = pareto_df

    # 5. Embedding Clusters
    print("\n[5/5] Embedding Clusters (t-SNE)...")
    embedding_df = create_embedding_clusters(Z, y_pred, class_names)
    plot_embedding_clusters(embedding_df, save_path=str(output_path / 'embedding_clusters.png'), show=show_plots)
    results['embedding_clusters'] = embedding_df

    print("\n" + "="*80)
    print(f"[COMPLETE] All plots saved to: {output_path}")
    print("="*80)

    return results


# ============================================================================
# 7. EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create mock data for testing
    print("Running with mock data for demonstration...")

    np.random.seed(42)
    N = 84  # Number of test nodes

    # Mock inputs
    y_true = np.random.randint(0, 4, size=N)
    y_pred = np.random.randint(0, 4, size=N)
    y_reg = np.random.uniform(0, 100, size=N)
    y_reg_pred = y_reg + np.random.normal(0, 10, size=N)
    node_types = np.random.choice(['aircraft', 'sector', 'airport'], size=N).tolist()
    Z = np.random.randn(84, 256)
    hyperedge_degree = np.random.randint(1, 20, size=84).astype(float)
    cascade_depth = np.random.randint(0, 10, size=84)
    pareto_position = np.random.uniform(0, 1, size=N)
    classification_accuracy = np.random.uniform(0.6, 0.95, size=N)

    # Generate all plots
    results = generate_all_plots(
        y_true=y_true,
        y_pred=y_pred,
        y_reg=y_reg,
        y_reg_pred=y_reg_pred,
        node_types=node_types,
        Z=Z,
        hyperedge_degree=hyperedge_degree,
        cascade_depth=cascade_depth,
        pareto_position=pareto_position,
        classification_accuracy=classification_accuracy,
        class_names=['Low', 'Medium', 'High', 'Critical'],
        output_dir='outputs/figures',
        show_plots=False  # Set to True to display interactively
    )

    # Print DataFrames
    print("\n" + "-"*80)
    print("DATAFRAMES GENERATED:")
    print("-"*80)
    for name, df in results.items():
        print(f"\n{name}:")
        print(df.head())
        print(f"Shape: {df.shape}")

    print("\n[COMPLETE] Check outputs/figures/ for saved plots.")
