"""
Evaluation metrics for synthetic data quality assessment.

This module provides comprehensive metrics for evaluating the quality of
synthetic healthcare data, including statistical similarity, privacy,
and utility metrics.
"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import wasserstein_distance, energy_distance, ks_2samp


def statistical_similarity(real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistical similarity metrics between real and synthetic data.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        
    Returns:
        Dictionary of statistical similarity metrics
    """
    # Ensure same number of features
    assert real_data.shape[1] == synthetic_data.shape[1], "Data dimensions must match"
    
    # Get number of features
    num_features = real_data.shape[1]
    
    # Initialize metrics
    wasserstein_distances = []
    energy_distances = []
    ks_statistics = []
    ks_pvalues = []
    
    # Calculate metrics for each feature
    for i in range(num_features):
        real_feature = real_data[:, i]
        synthetic_feature = synthetic_data[:, i]
        
        # Wasserstein distance
        w_dist = wasserstein_distance(real_feature, synthetic_feature)
        wasserstein_distances.append(w_dist)
        
        # Energy distance
        e_dist = energy_distance(real_feature, synthetic_feature)
        energy_distances.append(e_dist)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = ks_2samp(real_feature, synthetic_feature)
        ks_statistics.append(ks_stat)
        ks_pvalues.append(ks_pval)
    
    # Calculate correlation similarity
    real_corr = np.corrcoef(real_data, rowvar=False)
    synthetic_corr = np.corrcoef(synthetic_data, rowvar=False)
    
    # Handle NaNs
    real_corr = np.nan_to_num(real_corr)
    synthetic_corr = np.nan_to_num(synthetic_corr)
    
    # Calculate Frobenius norm of difference
    diff_norm = np.linalg.norm(real_corr - synthetic_corr, ord='fro')
    
    # Normalize by the number of elements
    n = real_corr.shape[0]
    normalized_diff = diff_norm / (n * n)
    
    # Convert to similarity score (higher is better)
    corr_similarity = 1.0 / (1.0 + normalized_diff)
    
    # Return metrics
    return {
        'wasserstein_distance_mean': float(np.mean(wasserstein_distances)),
        'wasserstein_distance_max': float(np.max(wasserstein_distances)),
        'energy_distance_mean': float(np.mean(energy_distances)),
        'energy_distance_max': float(np.max(energy_distances)),
        'ks_statistic_mean': float(np.mean(ks_statistics)),
        'ks_pvalue_mean': float(np.mean(ks_pvalues)),
        'correlation_similarity': float(corr_similarity)
    }


def dimension_wise_similarity(real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate dimension-wise similarity metrics between real and synthetic data.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        
    Returns:
        Dictionary of dimension-wise similarity metrics
    """
    # Ensure same number of features
    assert real_data.shape[1] == synthetic_data.shape[1], "Data dimensions must match"
    
    # Get number of features
    num_features = real_data.shape[1]
    
    # Initialize metrics
    feature_metrics = {
        'wasserstein_distances': np.zeros(num_features),
        'mean_differences': np.zeros(num_features),
        'std_differences': np.zeros(num_features),
        'min_differences': np.zeros(num_features),
        'max_differences': np.zeros(num_features),
        'ks_statistics': np.zeros(num_features),
        'ks_pvalues': np.zeros(num_features)
    }
    
    # Calculate metrics for each feature
    for i in range(num_features):
        real_feature = real_data[:, i]
        synthetic_feature = synthetic_data[:, i]
        
        # Wasserstein distance
        feature_metrics['wasserstein_distances'][i] = wasserstein_distance(real_feature, synthetic_feature)
        
        # Statistical differences
        feature_metrics['mean_differences'][i] = np.abs(np.mean(real_feature) - np.mean(synthetic_feature))
        feature_metrics['std_differences'][i] = np.abs(np.std(real_feature) - np.std(synthetic_feature))
        feature_metrics['min_differences'][i] = np.abs(np.min(real_feature) - np.min(synthetic_feature))
        feature_metrics['max_differences'][i] = np.abs(np.max(real_feature) - np.max(synthetic_feature))
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = ks_2samp(real_feature, synthetic_feature)
        feature_metrics['ks_statistics'][i] = ks_stat
        feature_metrics['ks_pvalues'][i] = ks_pval
    
    return feature_metrics


def privacy_metrics(real_data: np.ndarray, synthetic_data: np.ndarray, k_values: List[int] = [3, 5, 10]) -> Dict[str, float]:
    """
    Calculate privacy metrics for synthetic data.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        k_values: Values of k for k-nearest neighbor distance
        
    Returns:
        Dictionary of privacy metrics
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Initialize metrics
    privacy_scores = {}
    
    # Calculate privacy metrics for each k value
    for k in k_values:
        # Find k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(real_data)
        
        # Calculate distances from synthetic to real data
        distances, _ = nn.kneighbors(synthetic_data)
        
        # Calculate average minimum distance
        min_distances = distances[:, 0]  # Closest neighbor
        avg_min_distance = float(np.mean(min_distances))
        
        # Calculate average k-distance
        avg_k_distance = float(np.mean(distances[:, k-1]))
        
        # Store metrics
        privacy_scores[f'min_distance_k{k}'] = avg_min_distance
        privacy_scores[f'avg_k_distance_k{k}'] = avg_k_distance
    
    # Calculate distance-to-closest-record (DCR)
    dcr = privacy_scores['min_distance_k3']
    privacy_scores['dcr'] = dcr
    
    # Add normalized DCR (higher is better for privacy)
    real_data_std = np.std(real_data, axis=0).mean()
    privacy_scores['normalized_dcr'] = dcr / real_data_std
    
    return privacy_scores


def utility_metrics(real_data: np.ndarray, 
                   synthetic_data: np.ndarray, 
                   labels_real: Optional[np.ndarray] = None,
                   labels_synthetic: Optional[np.ndarray] = None,
                   task: str = 'classification',
                   test_size: float = 0.3,
                   random_state: int = 42) -> Dict[str, float]:
    """
    Calculate utility metrics for synthetic data by training models.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        labels_real: Labels for real data (if available)
        labels_synthetic: Labels for synthetic data (if available)
        task: Type of utility task ('classification' or 'regression')
        test_size: Test split ratio
        random_state: Random seed
        
    Returns:
        Dictionary of utility metrics
    """
    # Check if supervised utility metrics can be calculated
    if labels_real is None or labels_synthetic is None:
        return {
            'supervised_utility': None,
            'note': 'Labels not provided, supervised utility metrics not calculated'
        }
    
    # Split real data into train and test sets
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        real_data, labels_real, test_size=test_size, random_state=random_state
    )
    
    # Initialize models and metrics based on task
    if task == 'classification':
        # Train model on real data
        real_model = RandomForestClassifier(random_state=random_state)
        real_model.fit(X_real_train, y_real_train)
        real_score = real_model.score(X_real_test, y_real_test)
        
        # Train model on synthetic data
        synthetic_model = RandomForestClassifier(random_state=random_state)
        synthetic_model.fit(synthetic_data, labels_synthetic)
        synthetic_score = synthetic_model.score(X_real_test, y_real_test)
        
        # Calculate predictions for ROC AUC
        try:
            real_probs = real_model.predict_proba(X_real_test)[:, 1]
            synthetic_probs = synthetic_model.predict_proba(X_real_test)[:, 1]
            
            real_auc = roc_auc_score(y_real_test, real_probs)
            synthetic_auc = roc_auc_score(y_real_test, synthetic_probs)
        except (IndexError, ValueError):
            # Multi-class or not probabilistic
            real_auc = None
            synthetic_auc = None
        
        # Calculate predictions for F1 score
        real_preds = real_model.predict(X_real_test)
        synthetic_preds = synthetic_model.predict(X_real_test)
        
        real_f1 = f1_score(y_real_test, real_preds, average='weighted')
        synthetic_f1 = f1_score(y_real_test, synthetic_preds, average='weighted')
        
        # Calculate utility metrics
        metrics = {
            'real_accuracy': float(real_score),
            'synthetic_accuracy': float(synthetic_score),
            'real_f1': float(real_f1),
            'synthetic_f1': float(synthetic_f1),
            'relative_accuracy': float(synthetic_score / real_score if real_score > 0 else 0),
            'relative_f1': float(synthetic_f1 / real_f1 if real_f1 > 0 else 0)
        }
        
        if real_auc is not None and synthetic_auc is not None:
            metrics['real_auc'] = float(real_auc)
            metrics['synthetic_auc'] = float(synthetic_auc)
            metrics['relative_auc'] = float(synthetic_auc / real_auc if real_auc > 0 else 0)
    
    elif task == 'regression':
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Train model on real data
        real_model = RandomForestRegressor(random_state=random_state)
        real_model.fit(X_real_train, y_real_train)
        real_preds = real_model.predict(X_real_test)
        real_mse = mean_squared_error(y_real_test, real_preds)
        real_r2 = r2_score(y_real_test, real_preds)
        
        # Train model on synthetic data
        synthetic_model = RandomForestRegressor(random_state=random_state)
        synthetic_model.fit(synthetic_data, labels_synthetic)
        synthetic_preds = synthetic_model.predict(X_real_test)
        synthetic_mse = mean_squared_error(y_real_test, synthetic_preds)
        synthetic_r2 = r2_score(y_real_test, synthetic_preds)
        
        # Calculate utility metrics
        metrics = {
            'real_mse': float(real_mse),
            'synthetic_mse': float(synthetic_mse),
            'real_r2': float(real_r2),
            'synthetic_r2': float(synthetic_r2),
            'relative_mse': float(real_mse / synthetic_mse if synthetic_mse > 0 else 0),
            'relative_r2': float(synthetic_r2 / real_r2 if real_r2 > 0 else 0)
        }
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return metrics


def membership_inference_attack(real_data: np.ndarray, 
                              synthetic_data: np.ndarray,
                              test_real_data: Optional[np.ndarray] = None,
                              n_splits: int = 5,
                              threshold_percentile: float = 95,
                              random_state: int = 42) -> Dict[str, float]:
    """
    Perform membership inference attack to assess privacy.
    
    Args:
        real_data: Real data used for training
        synthetic_data: Synthetic data
        test_real_data: Real data not used for training (if available)
        n_splits: Number of cross-validation splits
        threshold_percentile: Percentile for distance threshold
        random_state: Random seed
        
    Returns:
        Dictionary of membership inference attack metrics
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score
    
    # If test_real_data is not provided, use cross-validation
    if test_real_data is None:
        # Initialize metrics
        attack_results = []
        
        # Create KFold for cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        for train_idx, test_idx in kf.split(real_data):
            # Split real data
            train_real = real_data[train_idx]
            test_real = real_data[test_idx]
            
            # Create datasets for membership classification
            train_data = np.vstack([train_real, test_real])
            train_labels = np.concatenate([np.ones(len(train_real)), np.zeros(len(test_real))])
            
            # Compute distances to synthetic data
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(synthetic_data)
            
            # Get distances from real data to closest synthetic sample
            distances, _ = nn.kneighbors(train_data)
            distances = distances.flatten()
            
            # Calculate attack ROC AUC
            attack_auc = roc_auc_score(train_labels, -distances)  # Negative distances since smaller distance = more likely to be member
            attack_results.append(attack_auc)
        
        # Calculate final attack metrics
        attack_auc_mean = float(np.mean(attack_results))
        attack_auc_std = float(np.std(attack_results))
        
        # Risk level interpretation
        privacy_risk = "Low" if attack_auc_mean < 0.6 else ("Medium" if attack_auc_mean < 0.8 else "High")
        
        return {
            'attack_auc_mean': attack_auc_mean,
            'attack_auc_std': attack_auc_std,
            'privacy_risk': privacy_risk
        }
    
    else:
        # Use provided test data
        # Compute distances to synthetic data
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(synthetic_data)
        
        # Get distances for train (member) and test (non-member) data
        train_distances, _ = nn.kneighbors(real_data)
        test_distances, _ = nn.kneighbors(test_real_data)
        
        train_distances = train_distances.flatten()
        test_distances = test_distances.flatten()
        
        # Create datasets for membership classification
        distances = np.concatenate([train_distances, test_distances])
        labels = np.concatenate([np.ones(len(train_distances)), np.zeros(len(test_distances))])
        
        # Calculate attack ROC AUC
        attack_auc = roc_auc_score(labels, -distances)  # Negative distances since smaller distance = more likely to be member
        
        # Calculate attack accuracy using threshold
        threshold = np.percentile(train_distances, threshold_percentile)
        predictions = (distances <= threshold).astype(int)
        attack_accuracy = accuracy_score(labels, predictions)
        
        # Risk level interpretation
        privacy_risk = "Low" if attack_auc < 0.6 else ("Medium" if attack_auc < 0.8 else "High")
        
        return {
            'attack_auc': float(attack_auc),
            'attack_accuracy': float(attack_accuracy),
            'privacy_risk': privacy_risk
        }


def evaluate_synthetic_data(real_data: np.ndarray, 
                          synthetic_data: np.ndarray,
                          labels_real: Optional[np.ndarray] = None,
                          labels_synthetic: Optional[np.ndarray] = None,
                          task: str = 'classification',
                          privacy_tests: bool = True,
                          utility_tests: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation of synthetic data quality.
    
    Args:
        real_data: Real data samples
        synthetic_data: Synthetic data samples
        labels_real: Labels for real data (if available)
        labels_synthetic: Labels for synthetic data (if available)
        task: Type of utility task ('classification' or 'regression')
        privacy_tests: Whether to perform privacy tests
        utility_tests: Whether to perform utility tests
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Initialize results
    results = {}
    
    # Statistical similarity metrics
    results['statistical'] = statistical_similarity(real_data, synthetic_data)
    
    # Dimension-wise similarity
    dimension_metrics = dimension_wise_similarity(real_data, synthetic_data)
    
    # Feature-wise summary
    feature_summary = {
        'wasserstein_distance_mean_by_feature': float(np.mean(dimension_metrics['wasserstein_distances'])),
        'wasserstein_distance_worst_feature': float(np.max(dimension_metrics['wasserstein_distances'])),
        'wasserstein_distance_best_feature': float(np.min(dimension_metrics['wasserstein_distances'])),
        'feature_quality_variance': float(np.var(dimension_metrics['wasserstein_distances']))
    }
    
    results['features'] = feature_summary
    
    # Privacy metrics if enabled
    if privacy_tests:
        results['privacy'] = privacy_metrics(real_data, synthetic_data)
        
        # Membership inference attack
        results['membership_inference'] = membership_inference_attack(
            real_data, synthetic_data, random_state=42
        )
    
    # Utility metrics if enabled and labels are provided
    if utility_tests and labels_real is not None and labels_synthetic is not None:
        results['utility'] = utility_metrics(
            real_data, synthetic_data,
            labels_real, labels_synthetic,
            task=task
        )
    
    # Overall quality score (weighted average of metrics)
    # Statistical quality (40%)
    statistical_quality = 1.0 / (1.0 + results['statistical']['wasserstein_distance_mean'])
    
    # Feature quality (20%)
    feature_quality = 1.0 / (1.0 + feature_summary['wasserstein_distance_mean_by_feature'])
    
    # Privacy quality (20%)
    if privacy_tests:
        privacy_quality = results['privacy']['normalized_dcr']
        membership_inference_quality = 1.0 - results['membership_inference']['attack_auc_mean']
    else:
        privacy_quality = 0.5  # Default if not calculated
        membership_inference_quality = 0.5  # Default if not calculated
    
    # Utility quality (20%)
    if utility_tests and 'utility' in results and 'relative_accuracy' in results['utility']:
        utility_quality = results['utility']['relative_accuracy']
    else:
        utility_quality = 0.5  # Default if not calculated
    
    # Calculate overall score
    overall_score = (
        0.4 * statistical_quality +
        0.2 * feature_quality +
        0.1 * privacy_quality +
        0.1 * membership_inference_quality +
        0.2 * utility_quality
    )
    
    results['overall_quality_score'] = float(overall_score)
    
    return results


def compare_synthetic_generators(real_data: np.ndarray, 
                                synthetic_data_dict: Dict[str, np.ndarray],
                                labels_real: Optional[np.ndarray] = None,
                                labels_synthetic_dict: Optional[Dict[str, np.ndarray]] = None,
                                task: str = 'classification') -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple synthetic data generators.
    
    Args:
        real_data: Real data samples
        synthetic_data_dict: Dictionary of synthetic datasets from different generators
        labels_real: Labels for real data (if available)
        labels_synthetic_dict: Dictionary of labels for synthetic datasets (if available)
        task: Type of utility task ('classification' or 'regression')
        
    Returns:
        Dictionary of evaluation results for each generator
    """
    # Initialize results
    comparison_results = {}
    
    # Evaluate each synthetic dataset
    for name, synthetic_data in synthetic_data_dict.items():
        # Get labels for this synthetic dataset if available
        labels_synthetic = None
        if labels_synthetic_dict is not None and name in labels_synthetic_dict:
            labels_synthetic = labels_synthetic_dict[name]
        
        # Evaluate synthetic data
        results = evaluate_synthetic_data(
            real_data, synthetic_data,
            labels_real, labels_synthetic,
            task=task
        )
        
        # Store results
        comparison_results[name] = results
    
    # Add ranking information
    model_ranking = {}
    
    # Rank by overall quality score
    quality_scores = {name: results['overall_quality_score'] for name, results in comparison_results.items()}
    ranked_models = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
    
    model_ranking['overall'] = {model: rank+1 for rank, (model, _) in enumerate(ranked_models)}
    
    # Rank by statistical similarity
    statistical_scores = {name: 1.0/(1.0 + results['statistical']['wasserstein_distance_mean']) 
                        for name, results in comparison_results.items()}
    ranked_models = sorted(statistical_scores.items(), key=lambda x: x[1], reverse=True)
    
    model_ranking['statistical'] = {model: rank+1 for rank, (model, _) in enumerate(ranked_models)}
    
    # Rank by utility if available
    if all('utility' in results and 'relative_accuracy' in results['utility'] 
          for results in comparison_results.values()):
        utility_scores = {name: results['utility']['relative_accuracy'] 
                         for name, results in comparison_results.items()}
        ranked_models = sorted(utility_scores.items(), key=lambda x: x[1], reverse=True)
        
        model_ranking['utility'] = {model: rank+1 for rank, (model, _) in enumerate(ranked_models)}
    
    # Add ranking to results
    comparison_results['ranking'] = model_ranking
    
    return comparison_results 