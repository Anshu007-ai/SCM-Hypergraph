"""
Generate plot data from HT-HGNN model outputs.

This script processes trained model predictions and embeddings
to create DataFrames ready for visualization.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.manifold import TSNE


def compute_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Compute confusion matrix for 4-class classification.

    Args:
        y_true: numpy array of shape (N,) - true class labels
        y_pred: numpy array of shape (N,) - predicted class labels
        class_names: list of class names (default: ['Class 0', 'Class 1', 'Class 2', 'Class 3'])

    Returns:
        pandas.DataFrame: confusion matrix with class names as index/columns
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(4)]

    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=class_names, columns=class_names)


def compute_rmse_per_node_type(y_reg, y_reg_pred, node_types):
    """
    Compute RMSE per node type (aircraft, sector, airport).

    Args:
        y_reg: numpy array of shape (N,) - true delay values
        y_reg_pred: numpy array of shape (N,) - predicted delay values
        node_types: list of length N - node type for each sample

    Returns:
        pandas.DataFrame: RMSE for each node type
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


def create_cascade_centrality_data(cascade_depth, hyperedge_degree):
    """
    Create DataFrame for cascade depth vs centrality scatter plot.

    Args:
        cascade_depth: numpy array of shape (84,) - cascade depth for each node
        hyperedge_degree: numpy array of shape (84,) - hyperedge degree for each node

    Returns:
        pandas.DataFrame: columns = ['cascade_depth', 'hyperedge_degree']
    """
    return pd.DataFrame({
        'cascade_depth': cascade_depth,
        'hyperedge_degree': hyperedge_degree
    })


def create_accuracy_pareto_data(pareto_position, classification_accuracy):
    """
    Create DataFrame for accuracy vs Pareto position.

    Args:
        pareto_position: numpy array of shape (N,) - position on Pareto front
        classification_accuracy: numpy array of shape (N,) - per-node classification accuracy

    Returns:
        pandas.DataFrame: columns = ['pareto_position', 'accuracy']
    """
    return pd.DataFrame({
        'pareto_position': pareto_position,
        'accuracy': classification_accuracy
    })


def create_embedding_clusters(Z, y_pred, random_state=42, perplexity=30):
    """
    Run t-SNE on node embeddings and create 2D visualization data.

    Args:
        Z: numpy array of shape (84, 256) - node embeddings
        y_pred: numpy array of shape (84,) - predicted class labels
        random_state: int - random seed for reproducibility
        perplexity: int - t-SNE perplexity parameter (default: 30)

    Returns:
        pandas.DataFrame: columns = ['x', 'y', 'label']
    """
    # Adjust perplexity if needed (must be less than n_samples)
    n_samples = Z.shape[0]
    if perplexity >= n_samples:
        perplexity = max(5, n_samples // 3)

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    Z_2d = tsne.fit_transform(Z)

    return pd.DataFrame({
        'x': Z_2d[:, 0],
        'y': Z_2d[:, 1],
        'label': y_pred
    })


def generate_all_plot_data(y_true, y_pred, y_reg, y_reg_pred, node_types,
                           Z, hyperedge_degree, cascade_depth,
                           pareto_position, classification_accuracy,
                           class_names=None):
    """
    Generate all plot data in one call.

    Args:
        y_true: numpy array of shape (N,) - true class labels
        y_pred: numpy array of shape (N,) - predicted class labels
        y_reg: numpy array of shape (N,) - true delay values
        y_reg_pred: numpy array of shape (N,) - predicted delay values
        node_types: list of length N - node types
        Z: numpy array of shape (84, 256) - node embeddings
        hyperedge_degree: numpy array of shape (84,)
        cascade_depth: numpy array of shape (84,)
        pareto_position: numpy array of shape (N,)
        classification_accuracy: numpy array of shape (N,)
        class_names: list of class names (optional)

    Returns:
        dict: Dictionary containing all DataFrames
    """
    results = {
        'confusion_matrix': compute_confusion_matrix(y_true, y_pred, class_names),
        'rmse_per_node_type': compute_rmse_per_node_type(y_reg, y_reg_pred, node_types),
        'cascade_centrality': create_cascade_centrality_data(cascade_depth, hyperedge_degree),
        'accuracy_pareto': create_accuracy_pareto_data(pareto_position, classification_accuracy),
        'embedding_clusters': create_embedding_clusters(Z, y_pred)
    }

    return results


# Example usage
if __name__ == "__main__":
    # Example: Create dummy data for testing
    np.random.seed(42)
    N = 84  # Number of nodes

    # Mock inputs
    y_true = np.random.randint(0, 4, size=N)
    y_pred = np.random.randint(0, 4, size=N)
    y_reg = np.random.uniform(0, 100, size=N)
    y_reg_pred = y_reg + np.random.normal(0, 10, size=N)
    node_types = np.random.choice(['aircraft', 'sector', 'airport'], size=N).tolist()
    Z = np.random.randn(84, 256)
    hyperedge_degree = np.random.randint(1, 20, size=84)
    cascade_depth = np.random.randint(0, 10, size=84)
    pareto_position = np.random.uniform(0, 1, size=N)
    classification_accuracy = np.random.uniform(0.5, 1.0, size=N)

    # Define class names
    class_names = ['Low', 'Medium', 'High', 'Critical']

    # Generate all plot data
    plot_data = generate_all_plot_data(
        y_true, y_pred, y_reg, y_reg_pred, node_types,
        Z, hyperedge_degree, cascade_depth,
        pareto_position, classification_accuracy,
        class_names
    )

    # Display results
    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    print(plot_data['confusion_matrix'])
    print("\n")

    print("=" * 80)
    print("RMSE PER NODE TYPE")
    print("=" * 80)
    print(plot_data['rmse_per_node_type'])
    print("\n")

    print("=" * 80)
    print("CASCADE DEPTH VS CENTRALITY (first 5 rows)")
    print("=" * 80)
    print(plot_data['cascade_centrality'].head())
    print("\n")

    print("=" * 80)
    print("ACCURACY VS PARETO POSITION (first 5 rows)")
    print("=" * 80)
    print(plot_data['accuracy_pareto'].head())
    print("\n")

    print("=" * 80)
    print("EMBEDDING CLUSTERS (first 5 rows)")
    print("=" * 80)
    print(plot_data['embedding_clusters'].head())
    print("\n")

    # Save to CSV (optional)
    # plot_data['confusion_matrix'].to_csv('outputs/confusion_matrix.csv')
    # plot_data['rmse_per_node_type'].to_csv('outputs/rmse_per_node_type.csv', index=False)
    # plot_data['cascade_centrality'].to_csv('outputs/cascade_centrality.csv', index=False)
    # plot_data['accuracy_pareto'].to_csv('outputs/accuracy_pareto.csv', index=False)
    # plot_data['embedding_clusters'].to_csv('outputs/embedding_clusters.csv', index=False)
