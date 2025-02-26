"""
Generative Adversarial Network (GAN) for tabular data.
"""
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any

from ...config import config


class TabularGenerator(nn.Module):
    """
    Generator network for tabular data GAN.
    """
    
    def __init__(self, 
                latent_dim: int,
                output_dim: int,
                hidden_dims: List[int] = None,
                dropout_rate: float = 0.1):
        """
        Initialize the Generator.
        
        Args:
            latent_dim: Dimensionality of latent space
            output_dim: Dimensionality of output data
            hidden_dims: Dimensions of hidden layers
            dropout_rate: Dropout rate
        """
        super(TabularGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        
        # Build generator network
        layers = []
        prev_dim = latent_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator.
        
        Args:
            z: Latent vector
            
        Returns:
            Generated data
        """
        return self.model(z)


class TabularDiscriminator(nn.Module):
    """
    Discriminator network for tabular data GAN.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dims: List[int] = None,
                dropout_rate: float = 0.1,
                spectral_norm: bool = True):
        """
        Initialize the Discriminator.
        
        Args:
            input_dim: Dimensionality of input data
            hidden_dims: Dimensions of hidden layers
            dropout_rate: Dropout rate
            spectral_norm: Whether to use spectral normalization
        """
        super(TabularDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        
        # Build discriminator network
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            linear = nn.Linear(prev_dim, dim)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Output layer
        final_layer = nn.Linear(prev_dim, 1)
        if spectral_norm:
            final_layer = nn.utils.spectral_norm(final_layer)
        layers.append(final_layer)
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.
        
        Args:
            x: Input data
            
        Returns:
            Discrimination scores
        """
        return self.model(x)


class TabularGAN:
    """
    GAN for tabular data generation.
    """
    
    def __init__(self,
                input_dim: int,
                latent_dim: int = 128,
                generator_hidden_dims: List[int] = None,
                discriminator_hidden_dims: List[int] = None,
                dropout_rate: float = 0.1,
                lr_g: float = 1e-4,
                lr_d: float = 4e-4,
                spectral_norm: bool = True,
                gradient_penalty_weight: float = 10.0,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the GAN.
        
        Args:
            input_dim: Dimensionality of input/output data
            latent_dim: Dimensionality of latent space
            generator_hidden_dims: Dimensions of generator hidden layers
            discriminator_hidden_dims: Dimensions of discriminator hidden layers
            dropout_rate: Dropout rate
            lr_g: Learning rate for generator
            lr_d: Learning rate for discriminator
            spectral_norm: Whether to use spectral normalization
            gradient_penalty_weight: Weight for gradient penalty in WGAN-GP
            device: Device to use for training
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.gradient_penalty_weight = gradient_penalty_weight
        
        # Create generator and discriminator
        self.generator = TabularGenerator(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=generator_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        self.discriminator = TabularDiscriminator(
            input_dim=input_dim,
            hidden_dims=discriminator_hidden_dims,
            dropout_rate=dropout_rate,
            spectral_norm=spectral_norm
        )
        
        # Move models to device
        self.generator.to(device)
        self.discriminator.to(device)
        
        # Initialize optimizers
        self.optimizer_g = Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        # Initialize history
        self.history = {
            'epoch': [],
            'generator_loss': [],
            'discriminator_loss': [],
            'gradient_penalty': []
        }
    
    def _gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """
        Calculate gradient penalty for WGAN-GP.
        
        Args:
            real_data: Real data samples
            fake_data: Generated data samples
            
        Returns:
            Gradient penalty term
        """
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand(real_data.size())
        
        # Interpolate between real and fake data
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)
        
        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Gradients have shape (batch_size, input_dim)
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def _train_discriminator(self, real_data: torch.Tensor) -> Dict[str, float]:
        """
        Train the discriminator for one step.
        
        Args:
            real_data: Real data samples
            
        Returns:
            Dictionary with training metrics
        """
        batch_size = real_data.size(0)
        
        # Zero gradients
        self.optimizer_d.zero_grad()
        
        # Generate fake data
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z).detach()
        
        # Calculate discriminator outputs
        real_outputs = self.discriminator(real_data)
        fake_outputs = self.discriminator(fake_data)
        
        # Wasserstein loss with gradient penalty
        d_loss_real = -torch.mean(real_outputs)
        d_loss_fake = torch.mean(fake_outputs)
        
        # Add gradient penalty
        gradient_penalty = self._gradient_penalty(real_data, fake_data)
        d_loss = d_loss_real + d_loss_fake + self.gradient_penalty_weight * gradient_penalty
        
        # Backward pass and optimization
        d_loss.backward()
        self.optimizer_d.step()
        
        return {
            'discriminator_loss': d_loss.item(),
            'gradient_penalty': gradient_penalty.item()
        }
    
    def _train_generator(self, batch_size: int) -> Dict[str, float]:
        """
        Train the generator for one step.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dictionary with training metrics
        """
        # Zero gradients
        self.optimizer_g.zero_grad()
        
        # Generate fake data
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z)
        
        # Calculate discriminator outputs
        fake_outputs = self.discriminator(fake_data)
        
        # Generator loss (maximize D(G(z)))
        g_loss = -torch.mean(fake_outputs)
        
        # Backward pass and optimization
        g_loss.backward()
        self.optimizer_g.step()
        
        return {
            'generator_loss': g_loss.item()
        }
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader, d_steps: int = 5) -> Dict[str, float]:
        """
        Train the GAN for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            d_steps: Number of discriminator steps per generator step
            
        Returns:
            Dictionary with training metrics
        """
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        total_gp = 0
        num_batches = 0
        
        for batch in dataloader:
            real_data = batch[0].to(self.device)
            batch_size = real_data.size(0)
            
            # Train discriminator for d_steps
            for _ in range(d_steps):
                d_metrics = self._train_discriminator(real_data)
                total_d_loss += d_metrics['discriminator_loss']
                total_gp += d_metrics['gradient_penalty']
            
            # Train generator once
            g_metrics = self._train_generator(batch_size)
            total_g_loss += g_metrics['generator_loss']
            
            num_batches += 1
        
        # Calculate average losses
        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / (num_batches * d_steps)
        avg_gp = total_gp / (num_batches * d_steps)
        
        return {
            'generator_loss': avg_g_loss,
            'discriminator_loss': avg_d_loss,
            'gradient_penalty': avg_gp
        }
    
    def train(self, 
             dataloader: torch.utils.data.DataLoader,
             epochs: int = 100,
             d_steps: int = 5,
             verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the GAN.
        
        Args:
            dataloader: DataLoader for training data
            epochs: Number of epochs to train
            d_steps: Number of discriminator steps per generator step
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(dataloader, d_steps)
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['generator_loss'].append(metrics['generator_loss'])
            self.history['discriminator_loss'].append(metrics['discriminator_loss'])
            self.history['gradient_penalty'].append(metrics['gradient_penalty'])
            
            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch}/{epochs}, "
                      f"G Loss: {metrics['generator_loss']:.4f}, "
                      f"D Loss: {metrics['discriminator_loss']:.4f}, "
                      f"GP: {metrics['gradient_penalty']:.4f}")
        
        return self.history
    
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the trained GAN.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            samples = self.generator(z)
        return samples
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'history': self.history,
            'model_config': {
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'gradient_penalty_weight': self.gradient_penalty_weight
            }
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.history = checkpoint['history'] 