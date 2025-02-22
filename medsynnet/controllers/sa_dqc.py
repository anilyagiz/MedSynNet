import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features):
        batch_size = features.shape[0]
        labels = torch.arange(batch_size).to(features.device)
        
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = similarity_matrix / self.temperature
        
        loss = -torch.log(
            torch.exp(similarity_matrix[mask]) / 
            torch.exp(similarity_matrix).sum(dim=1)
        ).mean()
        
        return loss

class QualityEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class SelfAdaptiveDataQualityController:
    def __init__(self, input_dim, contamination=0.1):
        self.input_dim = input_dim
        self.quality_encoder = QualityEncoder(input_dim)
        self.contrastive_loss = ContrastiveLoss()
        self.anomaly_detector = IsolationForest(contamination=contamination)
        self.quality_threshold = 0.5
        
    def compute_quality_score(self, real_data, synthetic_data):
        """Compute quality score using contrastive learning."""
        real_features = self.quality_encoder(torch.FloatTensor(real_data))
        synthetic_features = self.quality_encoder(torch.FloatTensor(synthetic_data))
        
        combined_features = torch.cat([real_features, synthetic_features])
        quality_loss = self.contrastive_loss(combined_features)
        
        # Normalize quality score between 0 and 1
        quality_score = 1.0 / (1.0 + quality_loss.item())
        return quality_score
        
    def detect_anomalies(self, synthetic_data):
        """Detect anomalous synthetic samples using Isolation Forest."""
        # Fit and predict anomalies (-1 for anomalies, 1 for normal samples)
        predictions = self.anomaly_detector.fit_predict(synthetic_data)
        anomaly_indices = np.where(predictions == -1)[0]
        return anomaly_indices
        
    def evaluate_batch(self, real_batch, synthetic_batch):
        """Evaluate quality of synthetic batch."""
        quality_score = self.compute_quality_score(real_batch, synthetic_batch)
        anomaly_indices = self.detect_anomalies(synthetic_batch)
        
        # Filter out low quality and anomalous samples
        mask = np.ones(len(synthetic_batch), dtype=bool)
        mask[anomaly_indices] = False
        
        if quality_score < self.quality_threshold:
            return None, quality_score
        
        return synthetic_batch[mask], quality_score
    
    def update_threshold(self, quality_scores, percentile=75):
        """Adaptively update quality threshold based on historical scores."""
        if len(quality_scores) > 0:
            self.quality_threshold = np.percentile(quality_scores, percentile)
    
    def train(self, real_data_loader: DataLoader, synthetic_data_loader: DataLoader, epochs=10):
        """Train the quality controller."""
        optimizer = torch.optim.Adam(self.quality_encoder.parameters())
        
        for epoch in range(epochs):
            epoch_loss = 0
            for (real_batch, _), (synthetic_batch, _) in zip(real_data_loader, synthetic_data_loader):
                optimizer.zero_grad()
                
                # Combine real and synthetic data
                combined_data = torch.cat([real_batch, synthetic_batch])
                features = self.quality_encoder(combined_data)
                
                # Compute contrastive loss
                loss = self.contrastive_loss(features)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(real_data_loader):.4f}")
    
    def get_feedback(self, synthetic_data, real_data_stats):
        """Get feedback for generator improvement."""
        feedback = {
            'quality_score': self.compute_quality_score(real_data_stats, synthetic_data),
            'anomaly_ratio': len(self.detect_anomalies(synthetic_data)) / len(synthetic_data),
            'distribution_diff': np.mean(np.abs(np.mean(real_data_stats, axis=0) - 
                                              np.mean(synthetic_data, axis=0)))
        }
        return feedback 