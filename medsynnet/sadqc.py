import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
import numpy as np
from typing import Dict, List, Tuple

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features_1, features_2):
        """Calculate contrastive loss."""
        batch_size = features_1.shape[0]
        features = torch.cat([features_1, features_2], dim=0)
        
        # Calculate similarity matrix
        similarity = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        similarity = similarity / self.temperature
        
        # Determine positive pairs
        labels = torch.arange(batch_size, device=features_1.device)
        labels = torch.cat([labels, labels])
        
        # Calculate InfoNCE loss
        loss = F.cross_entropy(similarity, labels)
        return loss

class QualityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
    
    def forward(self, x):
        return self.encoder(x)

class SADQC:
    def __init__(self, input_dim, hidden_dim=256, contamination=0.1):
        self.quality_encoder = QualityEncoder(input_dim, hidden_dim)
        self.contrastive_loss = ContrastiveLoss()
        self.isolation_forest = IsolationForest(contamination=contamination)
        self.quality_threshold = 0.5
    
    def compute_quality_score(self, real_data: torch.Tensor, synthetic_data: torch.Tensor) -> float:
        """Calculate quality score for synthetic data."""
        self.quality_encoder.eval()
        with torch.no_grad():
            real_features = self.quality_encoder(real_data)
            synthetic_features = self.quality_encoder(synthetic_data)
            
            # Calculate contrastive loss
            loss = self.contrastive_loss(real_features, synthetic_features)
            
            # Normalize quality score
            quality_score = 1.0 / (1.0 + loss.item())
            
        return quality_score
    
    def detect_anomalies(self, data: torch.Tensor) -> np.ndarray:
        """Perform anomaly detection."""
        features = self.quality_encoder(data).detach().cpu().numpy()
        predictions = self.isolation_forest.fit_predict(features)
        return predictions
    
    def train_quality_model(self, real_data: torch.Tensor, synthetic_data: torch.Tensor,
                           num_epochs: int = 10, learning_rate: float = 0.001):
        """Train quality model."""
        optimizer = torch.optim.Adam(self.quality_encoder.parameters(), lr=learning_rate)
        
        self.quality_encoder.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            real_features = self.quality_encoder(real_data)
            synthetic_features = self.quality_encoder(synthetic_data)
            
            loss = self.contrastive_loss(real_features, synthetic_features)
            loss.backward()
            optimizer.step()
    
    def evaluate_batch(self, synthetic_batch: torch.Tensor, real_reference: torch.Tensor) -> Dict[str, float]:
        """Evaluate synthetic data batch."""
        quality_score = self.compute_quality_score(real_reference, synthetic_batch)
        anomaly_predictions = self.detect_anomalies(synthetic_batch)
        
        anomaly_ratio = (anomaly_predictions == -1).mean()
        
        return {
            'quality_score': quality_score,
            'anomaly_ratio': anomaly_ratio,
            'is_accepted': quality_score > self.quality_threshold and anomaly_ratio < 0.3
        }
    
    def update_threshold(self, recent_scores: List[float]):
        """Dynamically update quality threshold."""
        if len(recent_scores) > 0:
            # Adjust threshold based on distribution of recent scores
            self.quality_threshold = np.percentile(recent_scores, 25)  # Use lower quartile 