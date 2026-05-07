"""
Visualization utilities for peptide absorbance analysis.

Provides plotting functions for EDA, model evaluation, feature importance,
and clustering visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Dict, List, Optional


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    ncols: int = 3,
    figsize: Optional[tuple] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot histograms for a list of features.

    Args:
        df: DataFrame containing the features.
        features: List of column names to plot.
        ncols: Number of columns in the subplot grid.
        figsize: Figure size tuple. Auto-calculated if None.
        dpi: Figure resolution.

    Returns:
        Matplotlib Figure object.
    """
    nrows = (len(features) + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = np.array(axes).flatten()

    for i, feat in enumerate(features):
        axes[i].hist(df[feat], bins=20, edgecolor='black', color='#1f77b4')
        axes[i].set_title(feat, fontsize=10)
        axes[i].set_xlabel('')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Feature Distributions', fontsize=14)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str],
    target: str = 'mean_abs',
    figsize: tuple = (12, 8),
    dpi: int = 150,
) -> plt.Figure:
    """Plot a correlation heatmap of features and the target variable.

    Args:
        df: DataFrame.
        features: Feature columns to include.
        target: Target column name.
        figsize: Figure size.
        dpi: Resolution.

    Returns:
        Matplotlib Figure object.
    """
    cols = features + [target] if target in df.columns else features
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Feature Correlation Matrix')
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Actual vs Predicted Absorbance',
    figsize: tuple = (6, 6),
    dpi: int = 150,
) -> plt.Figure:
    """Scatter plot of actual vs predicted values with ideal diagonal.

    Args:
        y_true: Ground truth values.
        y_pred: Model predictions.
        title: Plot title.
        figsize: Figure size.
        dpi: Resolution.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.scatterplot(x=y_true, y=y_pred, s=80, ax=ax)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', label='Ideal (y = x)')
    ax.set_xlabel('Actual Absorbance')
    ax.set_ylabel('Predicted Absorbance')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    n_top: int = 20,
    figsize: tuple = (10, 8),
    dpi: int = 150,
) -> plt.Figure:
    """Horizontal bar chart of feature importances.

    Args:
        importances: Array of importance values.
        feature_names: Corresponding feature names.
        n_top: Number of top features to display.
        figsize: Figure size.
        dpi: Resolution.

    Returns:
        Matplotlib Figure object.
    """
    indices = np.argsort(importances)[-n_top:]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.barh(range(len(indices)), importances[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {n_top} Feature Importances')
    fig.tight_layout()
    return fig


def plot_pca_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    absorbance: np.ndarray,
    figsize: tuple = (12, 5),
    dpi: int = 150,
) -> plt.Figure:
    """Plot PCA-reduced data colored by cluster label and by absorbance.

    Args:
        X: Feature matrix (will be PCA-transformed to 2D).
        labels: Cluster labels for each sample.
        absorbance: Absorbance values for coloring.
        figsize: Figure size.
        dpi: Resolution.

    Returns:
        Matplotlib Figure object.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', s=60)
    axes[0].set_title('Colored by Cluster')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=absorbance, cmap='viridis', s=60)
    axes[1].set_title('Colored by Absorbance')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    plt.colorbar(scatter2, ax=axes[1], label='Absorbance')

    fig.suptitle('PCA Visualization of Peptide Clusters')
    fig.tight_layout()
    return fig
