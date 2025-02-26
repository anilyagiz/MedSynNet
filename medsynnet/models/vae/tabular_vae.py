"""
Variational Autoencoder (VAE) for tabular data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any

from ...config import config


class TabularVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for tabular healthcare data.
    """
    
    def __init__(self, 
                input_dim: int,
                latent_dim: int = 128,
                hidden_dims: List[int] = None,
                dropout_rate: float = 0.1,
                beta: float = 1.0):
        """
        Initialize the TabularVAE.
        
        Args:
            input_dim: Dimensionality of input data
            latent_dim: Dimensionality of latent space
            hidden_dims: Dimensions of hidden layers
            dropout_rate: Dropout rate
            beta: Weight of KL divergence term in loss function
        """
        super(TabularVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        
        # Encoder network
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(nn.LeakyReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space projections
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder network
        decoder_layers = []
        prev_dim = latent_dim
        
        for dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(nn.LeakyReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input data to latent distribution parameters.
        
        Args:
            x: Input data tensor
            
        Returns:
            Tuple of (mean, log_variance) of latent distribution
        """
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed data.
        
        Args:
            z: Latent vector
            
        Returns:
            Reconstructed data
        """
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input data tensor
            
        Returns:
            Tuple of (reconstructed_x, mean, log_variance)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var
    
    def loss_function(self, 
                     x_reconstructed: torch.Tensor, 
                     x: torch.Tensor, 
                     mu: torch.Tensor, 
                     log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate VAE loss function.
        
        Args:
            x_reconstructed: Reconstructed data
            x: Original input data
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        loss = recon_loss + self.beta * kl_div
        
        return {
            'loss': loss,
            'reconstruction_loss': recon_loss,
            'kl_divergence': kl_div
        }
    
    def sample(self, num_samples: int, device: str = 'cuda') -> torch.Tensor:
        """
        Generate samples from the VAE.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to use for generation
            
        Returns:
            Generated samples
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


class TabularVAETrainer:
    """
    Trainer for TabularVAE model.
    """
    
    def __init__(self, 
                model: TabularVAE,
                learning_rate: float = 1e-4,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the VAE trainer.
        
        Args:
            model: VAE model
            learning_rate: Learning rate for optimizer
            device: Device to use for training
        """
        self.model = model
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.model.to(device)
        
        self.history = {
            'epoch': [],
            'loss': [],
            'reconstruction_loss': [],
            'kl_divergence': []
        }
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        num_batches = len(dataloader)
        
        for batch in dataloader:
            x = batch[0].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            x_reconstructed, mu, log_var = self.model(x)
            
            # Calculate loss
            loss_dict = self.model.loss_function(x_reconstructed, x, mu, log_var)
            loss = loss_dict['loss']
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_recon_loss += loss_dict['reconstruction_loss'].item()
            epoch_kl_loss += loss_dict['kl_divergence'].item()
        
        # Calculate average losses
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        
        return {
            'loss': avg_loss,
            'reconstruction_loss': avg_recon_loss,
            'kl_divergence': avg_kl_loss
        }
    
    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: DataLoader for validation data
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                # Forward pass
                x_reconstructed, mu, log_var = self.model(x)
                
                # Calculate loss
                loss_dict = self.model.loss_function(x_reconstructed, x, mu, log_var)
                
                # Accumulate losses
                val_loss += loss_dict['loss'].item()
                val_recon_loss += loss_dict['reconstruction_loss'].item()
                val_kl_loss += loss_dict['kl_divergence'].item()
        
        # Calculate average losses
        avg_loss = val_loss / num_batches
        avg_recon_loss = val_recon_loss / num_batches
        avg_kl_loss = val_kl_loss / num_batches
        
        return {
            'loss': avg_loss,
            'reconstruction_loss': avg_recon_loss,
            'kl_divergence': avg_kl_loss
        }
    
    def train(self, 
             train_dataloader: torch.utils.data.DataLoader,
             val_dataloader: Optional[torch.utils.data.DataLoader] = None,
             epochs: int = 100,
             early_stopping_patience: int = 10,
             verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Train for one epoch
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validate if validation data is provided
            if val_dataloader:
                val_metrics = self.validate(val_dataloader)
                current_val_loss = val_metrics['loss']
                
                # Check for early stopping
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
                
                if verbose and epoch % 5 == 0:
                    print(f"Epoch {epoch}/{epochs}, "
                          f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"Recon Loss: {val_metrics['reconstruction_loss']:.4f}, "
                          f"KL Div: {val_metrics['kl_divergence']:.4f}")
            else:
                if verbose and epoch % 5 == 0:
                    print(f"Epoch {epoch}/{epochs}, "
                          f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Recon Loss: {train_metrics['reconstruction_loss']:.4f}, "
                          f"KL Div: {train_metrics['kl_divergence']:.4f}")
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['loss'].append(train_metrics['loss'])
            self.history['reconstruction_loss'].append(train_metrics['reconstruction_loss'])
            self.history['kl_divergence'].append(train_metrics['kl_divergence'])
        
        return self.history
    
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the trained VAE.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(num_samples, self.device)
        return samples
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_config': {
                'input_dim': self.model.input_dim,
                'latent_dim': self.model.latent_dim,
                'beta': self.model.beta
            }
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history'] 