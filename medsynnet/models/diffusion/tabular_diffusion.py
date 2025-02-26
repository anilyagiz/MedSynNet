"""
Diffusion Model for tabular data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any

from ...config import config


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for diffusion timesteps.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to create position embeddings.
        
        Args:
            time: Timestep tensor [batch_size]
            
        Returns:
            Position embeddings [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TabularDiffusionNet(nn.Module):
    """
    U-Net style network for diffusion model on tabular data.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dims: List[int] = None,
                time_embedding_dim: int = 128,
                dropout_rate: float = 0.1):
        """
        Initialize the diffusion network.
        
        Args:
            input_dim: Dimensionality of input data
            hidden_dims: Dimensions of hidden layers
            time_embedding_dim: Dimension of time embeddings
            dropout_rate: Dropout rate
        """
        super(TabularDiffusionNet, self).__init__()
        
        self.input_dim = input_dim
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        
        # Time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 2),
            nn.GELU(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim)
        )
        
        # Down path (encoder)
        self.down_blocks = nn.ModuleList()
        in_features = input_dim
        
        for dim in hidden_dims:
            self.down_blocks.append(nn.ModuleList([
                nn.Linear(in_features, dim),
                nn.GroupNorm(8, dim),  # Group norm for stability
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(time_embedding_dim, dim)  # Time conditioning
            ]))
            in_features = dim
        
        # Middle block
        self.mid_block1 = nn.Linear(hidden_dims[-1], hidden_dims[-1] * 2)
        self.mid_norm1 = nn.GroupNorm(8, hidden_dims[-1] * 2)
        self.mid_time1 = nn.Linear(time_embedding_dim, hidden_dims[-1] * 2)
        
        self.mid_block2 = nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1])
        self.mid_norm2 = nn.GroupNorm(8, hidden_dims[-1])
        self.mid_time2 = nn.Linear(time_embedding_dim, hidden_dims[-1])
        
        # Up path (decoder)
        self.up_blocks = nn.ModuleList()
        hidden_dims = list(reversed(hidden_dims))
        
        for i, dim in enumerate(hidden_dims):
            # Skip connection dimension
            skip_dim = hidden_dims[i-1] if i > 0 else hidden_dims[0]
            
            # Input dimension with skip connection
            in_features = dim + skip_dim if i > 0 else dim
            out_features = dim // 2 if i < len(hidden_dims) - 1 else input_dim
            
            self.up_blocks.append(nn.ModuleList([
                nn.Linear(in_features, out_features),
                nn.GroupNorm(8, out_features) if i < len(hidden_dims) - 1 else nn.Identity(),
                nn.SiLU() if i < len(hidden_dims) - 1 else nn.Identity(),
                nn.Dropout(dropout_rate) if i < len(hidden_dims) - 1 else nn.Identity(),
                nn.Linear(time_embedding_dim, out_features)  # Time conditioning
            ]))
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input noisy data
            time: Diffusion timesteps
            
        Returns:
            Predicted noise
        """
        # Time embeddings
        t = self.time_mlp(time)
        
        # Down path with skip connections
        skip_connections = []
        for block in self.down_blocks:
            linear, norm, act, dropout, time_proj = block
            h = linear(x)
            h = norm(h)
            h = h + time_proj(t).unsqueeze(1)  # Add time signal
            h = act(h)
            h = dropout(h)
            
            skip_connections.append(h)
            x = h
        
        # Middle block
        h = self.mid_block1(x)
        h = self.mid_norm1(h)
        h = h + self.mid_time1(t).unsqueeze(1)
        h = F.silu(h)
        
        h = self.mid_block2(h)
        h = self.mid_norm2(h)
        h = h + self.mid_time2(t).unsqueeze(1)
        h = F.silu(h)
        
        # Up path with skip connections
        skip_connections = list(reversed(skip_connections))
        
        for i, block in enumerate(self.up_blocks):
            linear, norm, act, dropout, time_proj = block
            
            # Add skip connection if not the first block
            if i > 0:
                h = torch.cat([h, skip_connections[i]], dim=-1)
            
            h = linear(h)
            h = norm(h)
            
            if i < len(self.up_blocks) - 1:  # Don't add time signal to the final layer
                h = h + time_proj(t).unsqueeze(1)
                h = act(h)
                h = dropout(h)
            
        return h


class TabularDiffusion:
    """
    Diffusion model for tabular data generation.
    """
    
    def __init__(self,
                input_dim: int,
                hidden_dims: List[int] = None,
                noise_steps: int = 1000,
                beta_start: float = 1e-4,
                beta_end: float = 0.02,
                time_embedding_dim: int = 128,
                dropout_rate: float = 0.1,
                learning_rate: float = 1e-4,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the diffusion model.
        
        Args:
            input_dim: Dimensionality of input data
            hidden_dims: Dimensions of hidden layers
            noise_steps: Number of noise steps
            beta_start: Start value for noise schedule
            beta_end: End value for noise schedule
            time_embedding_dim: Dimension of time embeddings
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
            device: Device to use for training
        """
        self.input_dim = input_dim
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # Define beta schedule
        self.beta = self._prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        # Create network
        self.model = TabularDiffusionNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            time_embedding_dim=time_embedding_dim,
            dropout_rate=dropout_rate
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize history
        self.history = {
            'epoch': [],
            'loss': []
        }
    
    def _prepare_noise_schedule(self) -> torch.Tensor:
        """
        Prepare noise schedule for diffusion.
        
        Returns:
            Beta schedule
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def _noise_data(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to data at timestep t.
        
        Args:
            x: Input data
            t: Timestep
            
        Returns:
            Tuple of (noisy_data, noise)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        epsilon = torch.randn_like(x)
        
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Train the model for one step.
        
        Args:
            x: Input data
            
        Returns:
            Dictionary with training metrics
        """
        batch_size = x.shape[0]
        
        # Sample timesteps
        t = torch.randint(0, self.noise_steps, (batch_size,), device=self.device).long()
        
        # Add noise to data
        noisy_x, noise = self._noise_data(x, t)
        
        # Predict noise
        noise_pred = self.model(noisy_x, t)
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item()
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
        
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            x = batch[0].to(self.device)
            metrics = self.train_step(x)
            total_loss += metrics['loss']
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        return {
            'loss': avg_loss
        }
    
    def train(self, 
             dataloader: torch.utils.data.DataLoader,
             epochs: int = 100,
             verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the diffusion model.
        
        Args:
            dataloader: DataLoader for training data
            epochs: Number of epochs to train
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(dataloader)
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['loss'].append(metrics['loss'])
            
            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {metrics['loss']:.4f}")
        
        return self.history
    
    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the diffusion model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        self.model.eval()
        
        # Start with random noise
        x = torch.randn((num_samples, self.input_dim)).to(self.device)
        
        # Gradually denoise the data
        for i in reversed(range(self.noise_steps)):
            t = torch.ones(num_samples, device=self.device).long() * i
            
            # Predict noise
            predicted_noise = self.model(x, t)
            
            # Remove noise (reverse diffusion step)
            alpha = self.alpha[t][:, None]
            alpha_hat = self.alpha_hat[t][:, None]
            beta = self.beta[t][:, None]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        return x
    
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
                'input_dim': self.input_dim,
                'noise_steps': self.noise_steps,
                'beta_start': self.beta_start,
                'beta_end': self.beta_end
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