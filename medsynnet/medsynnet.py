import torch
from torch.utils.data import DataLoader
import numpy as np

from .models.hhg import HierarchicalHybridGenerator
from .models.fsl import FederatedSyntheticLearning
from .controllers.sa_dqc import SelfAdaptiveDataQualityController

class MedSynNet:
    def __init__(self, input_dim, latent_dim=64, num_rounds=3):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Initialize components
        self.generator = HierarchicalHybridGenerator(input_dim, latent_dim)
        self.federated_learning = FederatedSyntheticLearning(self.generator, num_rounds)
        self.quality_controller = SelfAdaptiveDataQualityController(input_dim)
        
        self.quality_history = []
        
    def train(self, client_datasets, batch_size=32, epochs=10):
        """Train the MedSynNet framework."""
        # Create data loaders for each client
        client_loaders = [
            DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for dataset in client_datasets
        ]
        
        # Initialize clients for federated learning
        clients = [
            self.federated_learning.create_client(loader)
            for loader in client_loaders
        ]
        
        # Start federated learning process
        print("Starting Federated Learning...")
        self.federated_learning.start_federated_learning(clients)
        
        # Train quality controller
        print("Training Quality Controller...")
        real_loader = client_loaders[0]  # Use first client's data for quality control
        synthetic_data = self.generate_synthetic_data(len(client_datasets[0]))
        synthetic_loader = DataLoader(synthetic_data, batch_size=batch_size, shuffle=True)
        
        self.quality_controller.train(real_loader, synthetic_loader, epochs)
        
    def generate_synthetic_data(self, num_samples):
        """Generate synthetic medical data."""
        synthetic_data = self.generator.generate_synthetic_data(num_samples)
        
        # Quality control
        quality_score = self.quality_controller.compute_quality_score(
            self.get_real_data_stats(),
            synthetic_data
        )
        self.quality_history.append(quality_score)
        
        # Update quality threshold
        self.quality_controller.update_threshold(self.quality_history)
        
        # Get feedback for improvement
        feedback = self.quality_controller.get_feedback(
            synthetic_data,
            self.get_real_data_stats()
        )
        
        print("Generation Quality Metrics:")
        print(f"Quality Score: {feedback['quality_score']:.4f}")
        print(f"Anomaly Ratio: {feedback['anomaly_ratio']:.4f}")
        print(f"Distribution Difference: {feedback['distribution_diff']:.4f}")
        
        return synthetic_data
    
    def get_real_data_stats(self):
        """Get statistics of real data for quality comparison."""
        # This should be implemented based on your specific data
        # For now, return dummy stats
        return np.random.randn(100, self.input_dim)
    
    def save_model(self, path):
        """Save the trained model."""
        torch.save({
            'generator_state': self.generator.state_dict(),
            'quality_controller_state': self.quality_controller.quality_encoder.state_dict(),
            'quality_history': self.quality_history,
        }, path)
    
    def load_model(self, path):
        """Load a trained model."""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.quality_controller.quality_encoder.load_state_dict(
            checkpoint['quality_controller_state']
        )
        self.quality_history = checkpoint['quality_history'] 