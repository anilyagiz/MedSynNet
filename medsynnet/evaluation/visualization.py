"""
Visualization utilities for synthetic data evaluation.

This module provides visualizations for comparing synthetic data quality,
including distribution plots, correlation heatmaps, and quality metric dashboards.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import io
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def set_plotting_style():
    """Set consistent plotting style for all visualizations."""
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    # Use high-quality rendering
    plt.rcParams['figure.dpi'] = 150
    # Modern color palette
    sns.set_palette("deep")


def compare_distributions(real_data: np.ndarray, 
                         synthetic_data: np.ndarray, 
                         feature_names: Optional[List[str]] = None,
                         n_cols: int = 3,
                         max_features: int = 12,
                         figsize: Tuple[int, int] = None) -> plt.Figure:
    """
    Create comparison plots of distributions for real vs synthetic data.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        feature_names: Names of features (optional)
        n_cols: Number of columns in the subplot grid
        max_features: Maximum number of features to plot
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    set_plotting_style()
    
    # Get feature names or generate them
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(real_data.shape[1])]
    
    # Limit number of features to plot
    n_features = min(real_data.shape[1], max_features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    # Set figure size if not provided
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 3)
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot distributions for each feature
    for i in range(n_features):
        ax = axes[i]
        
        # Get feature data
        real_feature = real_data[:, i]
        synthetic_feature = synthetic_data[:, i]
        
        # Plot KDE for both distributions
        sns.kdeplot(real_feature, ax=ax, label="Real", color="blue", fill=True, alpha=0.3)
        sns.kdeplot(synthetic_feature, ax=ax, label="Synthetic", color="red", fill=True, alpha=0.3)
        
        # Set labels and title
        ax.set_title(feature_names[i])
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    
    return fig


def plot_correlation_comparison(real_data: np.ndarray, 
                               synthetic_data: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (18, 8)) -> plt.Figure:
    """
    Create correlation heatmaps for real and synthetic data.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        feature_names: Names of features (optional)
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    set_plotting_style()
    
    # Get feature names or generate them
    if feature_names is None:
        feature_names = [f"F{i+1}" for i in range(real_data.shape[1])]
    
    # Calculate correlation matrices
    real_corr = np.corrcoef(real_data, rowvar=False)
    synthetic_corr = np.corrcoef(synthetic_data, rowvar=False)
    
    # Handle NaNs
    real_corr = np.nan_to_num(real_corr)
    synthetic_corr = np.nan_to_num(synthetic_corr)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot real data correlation
    sns.heatmap(real_corr, annot=False, ax=axes[0], cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=feature_names, yticklabels=feature_names)
    axes[0].set_title("Real Data Correlation")
    
    # Plot synthetic data correlation
    sns.heatmap(synthetic_corr, annot=False, ax=axes[1], cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=feature_names, yticklabels=feature_names)
    axes[1].set_title("Synthetic Data Correlation")
    
    plt.tight_layout()
    
    return fig


def plot_correlation_difference(real_data: np.ndarray, 
                              synthetic_data: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot the difference between real and synthetic correlation matrices.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        feature_names: Names of features (optional)
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    set_plotting_style()
    
    # Get feature names or generate them
    if feature_names is None:
        feature_names = [f"F{i+1}" for i in range(real_data.shape[1])]
    
    # Calculate correlation matrices
    real_corr = np.corrcoef(real_data, rowvar=False)
    synthetic_corr = np.corrcoef(synthetic_data, rowvar=False)
    
    # Handle NaNs
    real_corr = np.nan_to_num(real_corr)
    synthetic_corr = np.nan_to_num(synthetic_corr)
    
    # Calculate difference
    diff_corr = synthetic_corr - real_corr
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot difference
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(diff_corr, annot=False, ax=ax, cmap=cmap, vmin=-1, vmax=1,
                xticklabels=feature_names, yticklabels=feature_names, center=0)
    ax.set_title("Correlation Difference (Synthetic - Real)")
    
    plt.tight_layout()
    
    return fig


def plot_dimension_reduction(real_data: np.ndarray, 
                           synthetic_data: np.ndarray,
                           method: str = 'pca',
                           n_components: int = 2,
                           figsize: Tuple[int, int] = (10, 8),
                           random_state: int = 42) -> plt.Figure:
    """
    Plot dimension reduction visualization for real and synthetic data.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        method: Dimension reduction method ('pca' or 'tsne')
        n_components: Number of components to reduce to
        figsize: Figure size as (width, height)
        random_state: Random seed
        
    Returns:
        Matplotlib figure object
    """
    set_plotting_style()
    
    # Combine datasets for dimension reduction
    combined_data = np.vstack([real_data, synthetic_data])
    data_labels = np.concatenate([np.zeros(len(real_data)), np.ones(len(synthetic_data))])
    
    # Apply dimension reduction
    if method.lower() == 'pca':
        model = PCA(n_components=n_components, random_state=random_state)
        reduced_data = model.fit_transform(combined_data)
        reduction_name = 'PCA'
    elif method.lower() == 'tsne':
        model = TSNE(n_components=n_components, random_state=random_state)
        reduced_data = model.fit_transform(combined_data)
        reduction_name = 't-SNE'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Split back into real and synthetic
    real_reduced = reduced_data[:len(real_data)]
    synthetic_reduced = reduced_data[len(real_data):]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points
    ax.scatter(real_reduced[:, 0], real_reduced[:, 1], alpha=0.5, label='Real', color='blue')
    ax.scatter(synthetic_reduced[:, 0], synthetic_reduced[:, 1], alpha=0.5, label='Synthetic', color='red')
    
    # Add labels and legend
    ax.set_title(f"{reduction_name} Projection of Real and Synthetic Data")
    ax.set_xlabel(f"Component 1")
    ax.set_ylabel(f"Component 2")
    ax.legend()
    
    plt.tight_layout()
    
    return fig


def plot_metrics_radar(metrics: Dict[str, float], 
                     figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
    """
    Create a radar chart for data quality metrics.
    
    Args:
        metrics: Dictionary of metrics (values should be between 0 and 1)
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    set_plotting_style()
    
    # Ensure all metrics are between 0 and 1
    normalized_metrics = {}
    for key, value in metrics.items():
        if value < 0:
            normalized_metrics[key] = 0
        elif value > 1:
            normalized_metrics[key] = 1
        else:
            normalized_metrics[key] = value
    
    # Get metrics and labels
    labels = list(normalized_metrics.keys())
    values = list(normalized_metrics.values())
    
    # Number of variables
    N = len(labels)
    
    # What will be the angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Extend values to close the loop
    values += values[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Draw one axis per variable + add labels
    plt.xticks(angles[:-1], labels, fontsize=12)
    
    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], fontsize=12)
    plt.ylim(0, 1)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, alpha=0.4)
    
    plt.tight_layout()
    
    return fig


def plot_quality_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                                  metric_groups: Optional[Dict[str, List[str]]] = None,
                                  figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Create a bar chart comparing quality metrics across different generators.
    
    Args:
        metrics_dict: Dictionary of metrics dictionaries for each generator
        metric_groups: Dictionary mapping group names to lists of metric names
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    set_plotting_style()
    
    # Define metric groups if not provided
    if metric_groups is None:
        metric_groups = {
            'Statistical Similarity': ['correlation_similarity', 'wasserstein_distance_mean'],
            'Privacy': ['normalized_dcr', 'attack_auc_mean'],
            'Utility': ['relative_accuracy', 'relative_f1']
        }
    
    # Prepare data for plotting
    model_names = list(metrics_dict.keys())
    group_names = list(metric_groups.keys())
    
    # Create figure
    fig, axes = plt.subplots(1, len(group_names), figsize=figsize)
    
    # Plot each group
    for i, (group_name, metric_names) in enumerate(metric_groups.items()):
        ax = axes[i]
        
        # Get valid metric names that exist in all models
        valid_metrics = [m for m in metric_names if all(m in metrics_dict[model].get(group_name.lower(), {}) 
                                                       for model in model_names)]
        
        if not valid_metrics:
            ax.text(0.5, 0.5, f"No valid metrics for {group_name}", 
                   horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            continue
        
        # Prepare data for this group
        x = np.arange(len(valid_metrics))
        width = 0.8 / len(model_names)
        
        # Plot bars for each model
        for j, model in enumerate(model_names):
            metrics = metrics_dict[model].get(group_name.lower(), {})
            values = [metrics.get(m, 0) for m in valid_metrics]
            ax.bar(x + (j - len(model_names)/2 + 0.5) * width, values, width, label=model if i == 0 else "")
        
        # Configure plot
        ax.set_title(group_name)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_metrics, rotation=45, ha='right')
        if i == 0:
            ax.set_ylabel('Score')
    
    # Add legend to the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(model_names))
    
    plt.tight_layout()
    # Adjust for legend
    plt.subplots_adjust(bottom=0.2)
    
    return fig


def create_quality_dashboard(real_data: np.ndarray, 
                           synthetic_data: np.ndarray,
                           evaluation_results: Dict[str, Any],
                           feature_names: Optional[List[str]] = None) -> List[plt.Figure]:
    """
    Create a comprehensive dashboard of data quality visualizations.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        evaluation_results: Dictionary of evaluation metrics
        feature_names: Names of features (optional)
        
    Returns:
        List of matplotlib figure objects
    """
    # Initialize list of figures
    figures = []
    
    # 1. Distribution comparison
    dist_fig = compare_distributions(real_data, synthetic_data, feature_names)
    figures.append(dist_fig)
    
    # 2. Correlation comparison
    corr_fig = plot_correlation_comparison(real_data, synthetic_data, feature_names)
    figures.append(corr_fig)
    
    # 3. Correlation difference
    corr_diff_fig = plot_correlation_difference(real_data, synthetic_data, feature_names)
    figures.append(corr_diff_fig)
    
    # 4. Dimension reduction
    pca_fig = plot_dimension_reduction(real_data, synthetic_data, method='pca')
    figures.append(pca_fig)
    
    # 5. Quality metrics radar chart
    # Prepare metrics for radar chart (normalize to 0-1 range)
    radar_metrics = {}
    
    # Add statistical metrics
    if 'statistical' in evaluation_results:
        stat = evaluation_results['statistical']
        # Correlation similarity (already 0-1)
        if 'correlation_similarity' in stat:
            radar_metrics['Correlation'] = stat['correlation_similarity']
        
        # Wasserstein distance (convert to similarity)
        if 'wasserstein_distance_mean' in stat:
            radar_metrics['Distribution'] = 1.0 / (1.0 + stat['wasserstein_distance_mean'])
    
    # Add privacy metrics
    if 'privacy' in evaluation_results:
        priv = evaluation_results['privacy']
        if 'normalized_dcr' in priv:
            # Normalize to 0-1 (assuming reasonable range)
            radar_metrics['Privacy'] = min(priv['normalized_dcr'] / 2.0, 1.0)
    
    # Add membership inference metrics
    if 'membership_inference' in evaluation_results:
        mi = evaluation_results['membership_inference']
        if 'attack_auc_mean' in mi:
            # Convert to privacy score (1 - attack success)
            radar_metrics['MIA Resistance'] = 1.0 - mi['attack_auc_mean']
    
    # Add utility metrics
    if 'utility' in evaluation_results:
        util = evaluation_results['utility']
        if 'relative_accuracy' in util:
            radar_metrics['Accuracy'] = min(util['relative_accuracy'], 1.0)
        if 'relative_f1' in util:
            radar_metrics['F1 Score'] = min(util['relative_f1'], 1.0)
    
    # Add overall score
    if 'overall_quality_score' in evaluation_results:
        radar_metrics['Overall'] = min(evaluation_results['overall_quality_score'], 1.0)
    
    # Create radar chart
    if radar_metrics:
        radar_fig = plot_metrics_radar(radar_metrics)
        figures.append(radar_fig)
    
    return figures


def create_comparison_dashboard(real_data: np.ndarray,
                              synthetic_data_dict: Dict[str, np.ndarray],
                              comparison_results: Dict[str, Dict[str, Any]],
                              feature_names: Optional[List[str]] = None) -> List[plt.Figure]:
    """
    Create a dashboard comparing multiple synthetic data generators.
    
    Args:
        real_data: Real data samples
        synthetic_data_dict: Dictionary of synthetic datasets from different generators
        comparison_results: Dictionary of evaluation results for each generator
        feature_names: Names of features (optional)
        
    Returns:
        List of matplotlib figure objects
    """
    # Initialize list of figures
    figures = []
    
    # 1. PCA visualization with all generators
    combined_synthetic = np.vstack([data for data in synthetic_data_dict.values()])
    labels = np.concatenate([
        np.zeros(len(real_data)),  # Real data
        np.concatenate([np.full(len(data), i+1) for i, data in enumerate(synthetic_data_dict.values())])
    ])
    
    # Combine all data
    all_data = np.vstack([real_data, combined_synthetic])
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    reduced_data = pca.fit_transform(all_data)
    
    # Split back into real and synthetic
    real_reduced = reduced_data[:len(real_data)]
    synthetic_reduced = reduced_data[len(real_data):]
    
    # Create PCA plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot real data
    ax.scatter(real_reduced[:, 0], real_reduced[:, 1], alpha=0.7, label='Real', color='blue', s=50)
    
    # Plot synthetic data for each generator
    synthetic_start = 0
    colors = plt.cm.tab10(np.linspace(0, 1, len(synthetic_data_dict)))
    
    for i, (name, data) in enumerate(synthetic_data_dict.items()):
        synthetic_end = synthetic_start + len(data)
        data_reduced = synthetic_reduced[synthetic_start:synthetic_end]
        ax.scatter(data_reduced[:, 0], data_reduced[:, 1], alpha=0.5, label=name, color=colors[i], s=30)
        synthetic_start = synthetic_end
    
    # Add labels and legend
    ax.set_title(f"PCA Projection of Real vs. Multiple Synthetic Datasets")
    ax.set_xlabel(f"Principal Component 1")
    ax.set_ylabel(f"Principal Component 2")
    ax.legend()
    
    plt.tight_layout()
    figures.append(fig)
    
    # 2. Quality metrics comparison
    # Extract statistical metrics for comparison
    statistical_metrics = {}
    for name, results in comparison_results.items():
        if name != 'ranking' and 'statistical' in results:
            statistical_metrics[name] = results['statistical']
    
    if statistical_metrics:
        fig = plot_quality_metrics_comparison(
            {'Statistical': statistical_metrics},
            {'Statistical': ['correlation_similarity', 'wasserstein_distance_mean']}
        )
        figures.append(fig)
    
    # 3. Ranking comparison
    if 'ranking' in comparison_results:
        rankings = comparison_results['ranking']
        
        # Create ranking plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare ranking data
        model_names = list(rankings['overall'].keys())
        ranking_metrics = []
        ranking_values = []
        
        for metric, ranks in rankings.items():
            if metric in ['overall', 'statistical', 'utility']:
                ranking_metrics.append(metric.capitalize())
                ranking_values.append([ranks[model] for model in model_names])
        
        # Plot rankings
        x = np.arange(len(model_names))
        width = 0.8 / len(ranking_metrics)
        
        for i, (metric, values) in enumerate(zip(ranking_metrics, ranking_values)):
            ax.bar(x + (i - len(ranking_metrics)/2 + 0.5) * width, values, width, label=metric)
        
        # Lower rank is better
        ax.invert_yaxis()
        
        # Configure plot
        ax.set_title("Generator Rankings (Lower is Better)")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Rank')
        ax.legend()
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures


def save_dashboard_to_pdf(figures: List[plt.Figure], filename: str) -> None:
    """
    Save a list of figures to a PDF file.
    
    Args:
        figures: List of matplotlib figure objects
        filename: Output PDF filename
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(filename) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig) 