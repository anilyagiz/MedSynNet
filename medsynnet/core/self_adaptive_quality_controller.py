"""
Self-Adaptive Data Quality Controller (SA-DQC) module.

This module implements the data quality assessment and control component of MedSynNet,
providing automated evaluation, anomaly detection, and feedback mechanisms to
improve the quality of synthetic healthcare data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from ..config import config
from ..utils.data_utils import TabularDataProcessor


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for evaluating similarity between real and synthetic data distributions.
    """
    
    def __init__(self, temperature: float = 0.1):
        """
        Initialize the contrastive loss.
        
        Args:
            temperature: Temperature parameter controlling the sharpness of softmax distribution
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, real_embeddings: torch.Tensor, synthetic_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrastive loss between real and synthetic data embeddings.
        
        Args:
            real_embeddings: Embeddings of real data
            synthetic_embeddings: Embeddings of synthetic data
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings to unit length
        real_embeddings = F.normalize(real_embeddings, dim=1)
        synthetic_embeddings = F.normalize(synthetic_embeddings, dim=1)
        
        # Concatenate embeddings for computing similarity matrix
        embeddings = torch.cat([real_embeddings, synthetic_embeddings], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # Create labels for positive pairs
        batch_size = real_embeddings.shape[0]
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Compute contrastive loss
        logits = F.log_softmax(similarity_matrix, dim=1)
        loss = F.nll_loss(logits, labels)
        
        return loss


class DataEncoder(nn.Module):
    """
    Encoder network for learning data representations for quality assessment.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, embedding_dim: int = 64):
        """
        Initialize the encoder network.
        
        Args:
            input_dim: Dimensionality of input data
            hidden_dims: Dimensions of hidden layers
            embedding_dim: Dimensionality of output embeddings
        """
        super(DataEncoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # Build encoder network
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU())
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input data
            
        Returns:
            Encoded representations
        """
        return self.encoder(x)


class QualityMetrics:
    """
    Class for computing various quality metrics for synthetic data.
    """
    
    @staticmethod
    def wasserstein_distance(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
        """
        Calculate Wasserstein distance between real and synthetic data distributions.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            
        Returns:
            Wasserstein distance
        """
        from scipy.stats import wasserstein_distance
        
        # Calculate distance for each feature
        num_features = real_data.shape[1]
        distances = []
        
        for i in range(num_features):
            real_feature = real_data[:, i]
            synthetic_feature = synthetic_data[:, i]
            dist = wasserstein_distance(real_feature, synthetic_feature)
            distances.append(dist)
        
        # Return average distance across all features
        return float(np.mean(distances))
    
    @staticmethod
    def privacy_score(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
        """
        Calculate a privacy score based on minimum distance between real and synthetic samples.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            
        Returns:
            Privacy score (higher is better)
        """
        # Calculate pairwise distances between real and synthetic samples
        # Use a subsample for efficiency if needed
        max_samples = 1000
        if real_data.shape[0] > max_samples:
            indices = np.random.choice(real_data.shape[0], max_samples, replace=False)
            real_subsample = real_data[indices]
        else:
            real_subsample = real_data
            
        if synthetic_data.shape[0] > max_samples:
            indices = np.random.choice(synthetic_data.shape[0], max_samples, replace=False)
            synthetic_subsample = synthetic_data[indices]
        else:
            synthetic_subsample = synthetic_data
        
        # Standardize data for fair distance comparison
        scaler = StandardScaler()
        real_standardized = scaler.fit_transform(real_subsample)
        synthetic_standardized = scaler.transform(synthetic_subsample)
        
        # Calculate minimum distances
        distances = pairwise_distances(real_standardized, synthetic_standardized, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        # Higher minimum distance means better privacy
        privacy_score = float(np.mean(min_distances))
        
        return privacy_score
    
    @staticmethod
    def log_likelihood(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
        """
        Approximate log-likelihood score using kernel density estimation.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            
        Returns:
            Log-likelihood score
        """
        from sklearn.neighbors import KernelDensity
        
        # Use PCA to reduce dimensionality if needed
        if real_data.shape[1] > 10:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=10)
            real_data_reduced = pca.fit_transform(real_data)
            synthetic_data_reduced = pca.transform(synthetic_data)
        else:
            real_data_reduced = real_data
            synthetic_data_reduced = synthetic_data
        
        # Standardize data
        scaler = StandardScaler()
        real_data_scaled = scaler.fit_transform(real_data_reduced)
        synthetic_data_scaled = scaler.transform(synthetic_data_reduced)
        
        # Fit KDE to real data
        kde = KernelDensity(kernel='gaussian').fit(real_data_scaled)
        
        # Score synthetic data
        log_likelihood = kde.score(synthetic_data_scaled)
        
        return float(log_likelihood)
    
    @staticmethod
    def correlation_similarity(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
        """
        Calculate similarity between correlation matrices of real and synthetic data.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            
        Returns:
            Correlation similarity score (higher is better)
        """
        # Calculate correlation matrices
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
        similarity = 1.0 / (1.0 + normalized_diff)
        
        return float(similarity)


class SelfAdaptiveQualityController:
    """
    Self-Adaptive Data Quality Controller for synthetic data.
    Evaluates synthetic data quality, detects anomalies, and provides feedback
    to generative models to improve data quality.
    """
    
    def __init__(self,
                input_dim: int,
                contrastive_batch_size: int = 32,
                contrastive_temperature: float = 0.1,
                outlier_contamination: float = 0.05,
                quality_threshold: float = 0.7,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the quality controller.
        
        Args:
            input_dim: Dimensionality of input/output data
            contrastive_batch_size: Batch size for contrastive learning
            contrastive_temperature: Temperature for contrastive loss
            outlier_contamination: Expected proportion of outliers
            quality_threshold: Threshold for accepting synthetic data
            device: Device to use for training
        """
        self.input_dim = input_dim
        self.contrastive_batch_size = contrastive_batch_size
        self.contrastive_temperature = contrastive_temperature
        self.outlier_contamination = outlier_contamination
        self.quality_threshold = quality_threshold
        self.device = device
        
        # Initialize encoder for contrastive learning
        self.encoder = DataEncoder(
            input_dim=input_dim,
            hidden_dims=[256, 128],
            embedding_dim=64
        ).to(device)
        
        # Initialize contrastive loss
        self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temperature)
        
        # Initialize outlier detector
        self.outlier_detector = None
        
        # Initialize metrics history
        self.history = {
            'quality_scores': [],
            'wasserstein_distances': [],
            'privacy_scores': [],
            'correlation_similarities': [],
            'log_likelihoods': [],
            'rejection_rates': []
        }
    
    def fit(self, real_data: torch.Tensor, epochs: int = 10) -> None:
        """
        Fit the quality controller on real data.
        
        Args:
            real_data: Real data samples
            epochs: Number of training epochs
        """
        self.encoder.train()
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(real_data)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.contrastive_batch_size,
            shuffle=True
        )
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                batch_data = batch[0].to(self.device)
                
                # Create two augmented versions of the data
                # For tabular data, we can use simple noise addition as augmentation
                noise1 = torch.randn_like(batch_data) * 0.01
                noise2 = torch.randn_like(batch_data) * 0.01
                
                augmented1 = batch_data + noise1
                augmented2 = batch_data + noise2
                
                # Get embeddings
                embeddings1 = self.encoder(augmented1)
                embeddings2 = self.encoder(augmented2)
                
                # Calculate contrastive loss
                loss = self.contrastive_loss(embeddings1, embeddings2)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Fit outlier detector on real data
        self._fit_outlier_detector(real_data.cpu().numpy())
    
    def evaluate(self, 
                real_data: torch.Tensor, 
                synthetic_data: torch.Tensor,
                return_detailed_metrics: bool = False) -> Union[float, Dict[str, float]]:
        """
        Evaluate quality of synthetic data.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            return_detailed_metrics: Whether to return detailed metrics
            
        Returns:
            Quality score or dictionary of quality metrics
        """
        self.encoder.eval()
        
        # Convert to numpy for metric calculations
        real_np = real_data.cpu().numpy()
        synthetic_np = synthetic_data.cpu().numpy()
        
        # Calculate quality metrics
        wasserstein_dist = QualityMetrics.wasserstein_distance(real_np, synthetic_np)
        privacy_score = QualityMetrics.privacy_score(real_np, synthetic_np)
        corr_similarity = QualityMetrics.correlation_similarity(real_np, synthetic_np)
        log_likelihood = QualityMetrics.log_likelihood(real_np, synthetic_np)
        
        # Calculate contrastive loss between real and synthetic data
        with torch.no_grad():
            # Sample batches
            if len(real_data) > self.contrastive_batch_size:
                real_batch_indices = np.random.choice(len(real_data), self.contrastive_batch_size, replace=False)
                real_batch = real_data[real_batch_indices].to(self.device)
            else:
                real_batch = real_data.to(self.device)
                
            if len(synthetic_data) > self.contrastive_batch_size:
                synthetic_batch_indices = np.random.choice(len(synthetic_data), self.contrastive_batch_size, replace=False)
                synthetic_batch = synthetic_data[synthetic_batch_indices].to(self.device)
            else:
                synthetic_batch = synthetic_data.to(self.device)
            
            # Get embeddings
            real_embeddings = self.encoder(real_batch)
            synthetic_embeddings = self.encoder(synthetic_batch)
            
            # Calculate contrastive loss
            contrastive_loss_val = self.contrastive_loss(real_embeddings, synthetic_embeddings).item()
        
        # Normalize metrics to [0, 1] range
        normalized_wasserstein = 1.0 / (1.0 + wasserstein_dist)  # Lower wasserstein is better
        normalized_likelihood = 1.0 / (1.0 + abs(log_likelihood))  # Higher likelihood is better
        normalized_contrastive = 1.0 / (1.0 + contrastive_loss_val)  # Lower contrastive loss is better
        
        # Calculate overall quality score (weighted average of metrics)
        quality_score = 0.25 * normalized_wasserstein + \
                       0.25 * privacy_score + \
                       0.25 * corr_similarity + \
                       0.15 * normalized_likelihood + \
                       0.10 * normalized_contrastive
        
        # Update history
        self.history['quality_scores'].append(quality_score)
        self.history['wasserstein_distances'].append(wasserstein_dist)
        self.history['privacy_scores'].append(privacy_score)
        self.history['correlation_similarities'].append(corr_similarity)
        self.history['log_likelihoods'].append(log_likelihood)
        
        if return_detailed_metrics:
            return {
                'quality_score': quality_score,
                'wasserstein_distance': wasserstein_dist,
                'privacy_score': privacy_score,
                'correlation_similarity': corr_similarity,
                'log_likelihood': log_likelihood,
                'contrastive_loss': contrastive_loss_val
            }
        else:
            return quality_score
    
    def detect_anomalies(self, synthetic_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalies in the synthetic data.
        
        Args:
            synthetic_data: Synthetic data samples
            
        Returns:
            Tuple of (filtered_data, rejection_mask)
        """
        if self.outlier_detector is None:
            raise ValueError("Outlier detector not fit. Call fit() first.")
        
        # Get anomaly scores (-1 for outliers, 1 for inliers)
        synthetic_np = synthetic_data.cpu().numpy()
        anomaly_scores = self.outlier_detector.predict(synthetic_np)
        
        # Create rejection mask (True for rejected samples)
        rejection_mask = torch.tensor(anomaly_scores == -1, dtype=torch.bool)
        
        # Filter data
        filtered_data = synthetic_data[~rejection_mask]
        
        # Update history
        rejection_rate = float(rejection_mask.sum()) / len(synthetic_data)
        self.history['rejection_rates'].append(rejection_rate)
        
        return filtered_data, rejection_mask
    
    def filter_samples(self, 
                      real_data: torch.Tensor, 
                      synthetic_data: torch.Tensor,
                      quality_threshold: Optional[float] = None) -> torch.Tensor:
        """
        Filter synthetic samples based on quality assessment.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            quality_threshold: Quality threshold for filtering (overrides instance threshold)
            
        Returns:
            Filtered synthetic data
        """
        # Evaluate quality of each sample
        sample_qualities = self._evaluate_individual_samples(real_data, synthetic_data)
        
        # Apply threshold
        threshold = quality_threshold if quality_threshold is not None else self.quality_threshold
        quality_mask = sample_qualities >= threshold
        
        # Filter data
        filtered_data = synthetic_data[quality_mask]
        
        # Detect anomalies in quality-filtered data
        if len(filtered_data) > 0:
            filtered_data, _ = self.detect_anomalies(filtered_data)
        
        return filtered_data
    
    def _evaluate_individual_samples(self, real_data: torch.Tensor, synthetic_data: torch.Tensor) -> torch.Tensor:
        """
        Evaluate quality of individual synthetic samples.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            
        Returns:
            Quality scores for each synthetic sample
        """
        self.encoder.eval()
        
        # Get embeddings of real data
        with torch.no_grad():
            real_embeddings = self.encoder(real_data.to(self.device))
            
            # Initialize quality scores
            quality_scores = torch.zeros(len(synthetic_data))
            
            # Process in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(synthetic_data), batch_size):
                batch = synthetic_data[i:i+batch_size].to(self.device)
                batch_embeddings = self.encoder(batch)
                
                # Calculate similarity to real data
                similarity_matrix = torch.mm(batch_embeddings, real_embeddings.t())
                
                # Average similarity across real samples
                batch_scores = similarity_matrix.mean(dim=1).cpu()
                
                # Min-max normalize scores to [0, 1]
                if batch_scores.max() > batch_scores.min():
                    batch_scores = (batch_scores - batch_scores.min()) / (batch_scores.max() - batch_scores.min())
                
                # Store scores
                quality_scores[i:i+batch_size] = batch_scores
        
        return quality_scores
    
    def _fit_outlier_detector(self, real_data: np.ndarray) -> None:
        """
        Fit outlier detector on real data.
        
        Args:
            real_data: Real data samples
        """
        # Initialize and fit Isolation Forest for outlier detection
        self.outlier_detector = IsolationForest(
            contamination=self.outlier_contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        
        self.outlier_detector.fit(real_data)
    
    def save(self, filepath: str) -> None:
        """
        Save the quality controller to disk.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'outlier_detector': self.outlier_detector,
            'history': self.history,
            'config': {
                'input_dim': self.input_dim,
                'contrastive_batch_size': self.contrastive_batch_size,
                'contrastive_temperature': self.contrastive_temperature,
                'outlier_contamination': self.outlier_contamination,
                'quality_threshold': self.quality_threshold
            }
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load a saved quality controller from disk.
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load configuration
        config = checkpoint['config']
        self.input_dim = config['input_dim']
        self.contrastive_batch_size = config['contrastive_batch_size']
        self.contrastive_temperature = config['contrastive_temperature']
        self.outlier_contamination = config['outlier_contamination']
        self.quality_threshold = config['quality_threshold']
        
        # Load encoder
        self.encoder = DataEncoder(
            input_dim=self.input_dim,
            hidden_dims=[256, 128],
            embedding_dim=64
        ).to(self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        # Load outlier detector
        self.outlier_detector = checkpoint['outlier_detector']
        
        # Load history
        self.history = checkpoint['history'] 