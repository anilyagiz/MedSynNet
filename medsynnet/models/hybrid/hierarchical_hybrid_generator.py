"""
Hierarchical Hybrid Generator (HHG) for tabular data.

This module implements the core component of MedSynNet, a hierarchical generator
that combines VAE, Diffusion, and GAN models to generate high-quality synthetic tabular data.
"""
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union, Any

from ..vae.tabular_vae import TabularVAE, TabularVAETrainer
from ..diffusion.tabular_diffusion import TabularDiffusion
from ..gan.tabular_gan import TabularGAN
from ...config import config
from ...utils.data_utils import TabularDataProcessor


class HierarchicalHybridGenerator:
    """
    Hierarchical Hybrid Generator (HHG) that combines VAE, Diffusion, and GAN models
    for high-quality synthetic healthcare data generation.
    
    The HHG works in three stages:
    1. VAE: Captures essential features and compresses the data
    2. Diffusion: Refines the features with a denoising process
    3. GAN: Enhances realism through adversarial training
    """
    
    def __init__(self,
                input_dim: int,
                vae_config: Optional[Dict[str, Any]] = None,
                diffusion_config: Optional[Dict[str, Any]] = None,
                gan_config: Optional[Dict[str, Any]] = None,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Hierarchical Hybrid Generator.
        
        Args:
            input_dim: Dimensionality of input/output data
            vae_config: Configuration for VAE model
            diffusion_config: Configuration for Diffusion model
            gan_config: Configuration for GAN model
            device: Device to use for training and generation
        """
        self.input_dim = input_dim
        self.device = device
        
        # Get configurations from global config if not provided
        if vae_config is None:
            vae_config = config.get('hhg.vae', {})
        if diffusion_config is None:
            diffusion_config = config.get('hhg.diffusion', {})
        if gan_config is None:
            gan_config = config.get('hhg.gan', {})
        
        # Store configurations
        self.vae_config = vae_config
        self.diffusion_config = diffusion_config
        self.gan_config = gan_config
        
        # Initialize models based on configurations
        self._init_models()
        
        # Initialize training state
        self.is_trained = {
            'vae': False,
            'diffusion': False,
            'gan': False
        }
    
    def _init_models(self) -> None:
        """
        Initialize VAE, Diffusion, and GAN models.
        """
        # Initialize VAE if enabled
        if self.vae_config.get('enabled', True):
            self.vae = TabularVAE(
                input_dim=self.input_dim,
                latent_dim=self.vae_config.get('latent_dim', 128),
                hidden_dims=self.vae_config.get('hidden_dims', [256, 512, 256]),
                dropout_rate=self.vae_config.get('dropout_rate', 0.1),
                beta=self.vae_config.get('beta', 1.0)
            )
            
            self.vae_trainer = TabularVAETrainer(
                model=self.vae,
                learning_rate=self.vae_config.get('learning_rate', 1e-4),
                device=self.device
            )
        else:
            self.vae = None
            self.vae_trainer = None
        
        # Initialize Diffusion model if enabled
        if self.diffusion_config.get('enabled', True):
            self.diffusion = TabularDiffusion(
                input_dim=self.input_dim,
                hidden_dims=self.diffusion_config.get('hidden_dims', [256, 512, 256]),
                noise_steps=self.diffusion_config.get('noise_steps', 1000),
                beta_start=self.diffusion_config.get('beta_start', 1e-4),
                beta_end=self.diffusion_config.get('beta_end', 0.02),
                learning_rate=self.diffusion_config.get('learning_rate', 1e-4),
                device=self.device
            )
        else:
            self.diffusion = None
        
        # Initialize GAN if enabled
        if self.gan_config.get('enabled', True):
            self.gan = TabularGAN(
                input_dim=self.input_dim,
                latent_dim=self.gan_config.get('latent_dim', 128),
                generator_hidden_dims=self.gan_config.get('generator_hidden_dims', [256, 512, 256]),
                discriminator_hidden_dims=self.gan_config.get('discriminator_hidden_dims', [256, 512, 256]),
                lr_g=self.gan_config.get('learning_rate_g', 1e-4),
                lr_d=self.gan_config.get('learning_rate_d', 4e-4),
                gradient_penalty_weight=self.gan_config.get('gradient_penalty_weight', 10.0),
                device=self.device
            )
        else:
            self.gan = None
    
    def train_vae(self, 
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: Optional[torch.utils.data.DataLoader] = None,
                 epochs: Optional[int] = None,
                 verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the VAE model.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Number of epochs to train (overrides config)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        if self.vae is None or self.vae_trainer is None:
            raise ValueError("VAE is not enabled in the configuration")
        
        if epochs is None:
            epochs = self.vae_config.get('epochs', 100)
        
        if verbose:
            print(f"Training VAE for {epochs} epochs...")
        
        history = self.vae_trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=epochs,
            verbose=verbose
        )
        
        self.is_trained['vae'] = True
        return history
    
    def train_diffusion(self, 
                       train_dataloader: torch.utils.data.DataLoader,
                       epochs: Optional[int] = None,
                       verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the Diffusion model.
        
        Args:
            train_dataloader: DataLoader for training data
            epochs: Number of epochs to train (overrides config)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        if self.diffusion is None:
            raise ValueError("Diffusion model is not enabled in the configuration")
        
        if epochs is None:
            epochs = self.diffusion_config.get('epochs', 100)
        
        if verbose:
            print(f"Training Diffusion model for {epochs} epochs...")
        
        history = self.diffusion.train(
            dataloader=train_dataloader,
            epochs=epochs,
            verbose=verbose
        )
        
        self.is_trained['diffusion'] = True
        return history
    
    def train_gan(self, 
                 train_dataloader: torch.utils.data.DataLoader,
                 epochs: Optional[int] = None,
                 d_steps: int = 5,
                 verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the GAN model.
        
        Args:
            train_dataloader: DataLoader for training data
            epochs: Number of epochs to train (overrides config)
            d_steps: Number of discriminator steps per generator step
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        if self.gan is None:
            raise ValueError("GAN is not enabled in the configuration")
        
        if epochs is None:
            epochs = self.gan_config.get('epochs', 100)
        
        if verbose:
            print(f"Training GAN for {epochs} epochs...")
        
        history = self.gan.train(
            dataloader=train_dataloader,
            epochs=epochs,
            d_steps=d_steps,
            verbose=verbose
        )
        
        self.is_trained['gan'] = True
        return history
    
    def train(self,
             train_dataloader: torch.utils.data.DataLoader,
             val_dataloader: Optional[torch.utils.data.DataLoader] = None,
             vae_epochs: Optional[int] = None,
             diffusion_epochs: Optional[int] = None,
             gan_epochs: Optional[int] = None,
             sequential: bool = True,
             verbose: bool = True) -> Dict[str, Dict[str, List[float]]]:
        """
        Train all enabled models in the hierarchical generator.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            vae_epochs: Number of epochs to train VAE (overrides config)
            diffusion_epochs: Number of epochs to train Diffusion (overrides config)
            gan_epochs: Number of epochs to train GAN (overrides config)
            sequential: Whether to train models sequentially (recommended)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history for all models
        """
        history = {}
        
        if sequential:
            # Train models sequentially (VAE -> Diffusion -> GAN)
            if self.vae is not None and self.vae_trainer is not None:
                history['vae'] = self.train_vae(
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    epochs=vae_epochs,
                    verbose=verbose
                )
                
                # Generate data from VAE for next stage if needed
                if self.diffusion is not None or self.gan is not None:
                    if verbose:
                        print("Generating data from VAE for next stage...")
                    vae_samples = self.vae_trainer.generate_samples(len(train_dataloader.dataset))
                    vae_dataset = torch.utils.data.TensorDataset(vae_samples)
                    vae_dataloader = torch.utils.data.DataLoader(
                        vae_dataset,
                        batch_size=train_dataloader.batch_size,
                        shuffle=True
                    )
                    train_dataloader = vae_dataloader
            
            if self.diffusion is not None:
                history['diffusion'] = self.train_diffusion(
                    train_dataloader=train_dataloader,
                    epochs=diffusion_epochs,
                    verbose=verbose
                )
                
                # Generate data from Diffusion for GAN stage if needed
                if self.gan is not None:
                    if verbose:
                        print("Generating data from Diffusion for GAN stage...")
                    diffusion_samples = self.diffusion.sample(len(train_dataloader.dataset))
                    diffusion_dataset = torch.utils.data.TensorDataset(diffusion_samples)
                    diffusion_dataloader = torch.utils.data.DataLoader(
                        diffusion_dataset,
                        batch_size=train_dataloader.batch_size,
                        shuffle=True
                    )
                    train_dataloader = diffusion_dataloader
            
            if self.gan is not None:
                history['gan'] = self.train_gan(
                    train_dataloader=train_dataloader,
                    epochs=gan_epochs,
                    verbose=verbose
                )
        else:
            # Train models in parallel (not using hierarchical benefits)
            if self.vae is not None and self.vae_trainer is not None:
                history['vae'] = self.train_vae(
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    epochs=vae_epochs,
                    verbose=verbose
                )
                
            if self.diffusion is not None:
                history['diffusion'] = self.train_diffusion(
                    train_dataloader=train_dataloader,
                    epochs=diffusion_epochs,
                    verbose=verbose
                )
                
            if self.gan is not None:
                history['gan'] = self.train_gan(
                    train_dataloader=train_dataloader,
                    epochs=gan_epochs,
                    verbose=verbose
                )
        
        return history
    
    def generate(self, 
                num_samples: int, 
                use_vae: bool = True,
                use_diffusion: bool = True,
                use_gan: bool = True,
                return_intermediate: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate synthetic data using the hierarchical generator.
        
        Args:
            num_samples: Number of samples to generate
            use_vae: Whether to use VAE in generation
            use_diffusion: Whether to use Diffusion in generation
            use_gan: Whether to use GAN in generation
            return_intermediate: Whether to return intermediate samples from each stage
            
        Returns:
            Generated samples or dictionary with samples from each stage
        """
        samples = {}
        current_samples = None
        
        # Stage 1: VAE generation
        if use_vae and self.vae is not None and self.is_trained['vae']:
            if return_intermediate:
                samples['vae'] = self.vae_trainer.generate_samples(num_samples).cpu()
            current_samples = self.vae_trainer.generate_samples(num_samples)
        else:
            # If not using VAE, start with random noise
            current_samples = torch.randn(num_samples, self.input_dim).to(self.device)
        
        # Stage 2: Diffusion refinement
        if use_diffusion and self.diffusion is not None and self.is_trained['diffusion']:
            if current_samples is not None:
                # Use output from previous stage as conditioning
                self.diffusion.model.eval()
                with torch.no_grad():
                    for i in reversed(range(self.diffusion.noise_steps // 4)):  # Use fewer steps for efficiency
                        t = torch.ones(num_samples, device=self.device).long() * i
                        predicted_noise = self.diffusion.model(current_samples, t)
                        
                        # Gradually denoise
                        alpha = self.diffusion.alpha[t][:, None]
                        alpha_hat = self.diffusion.alpha_hat[t][:, None]
                        beta = self.diffusion.beta[t][:, None]
                        
                        if i > 0:
                            noise = torch.randn_like(current_samples) * 0.5  # Reduced noise for controlled refinement
                        else:
                            noise = torch.zeros_like(current_samples)
                        
                        current_samples = 1 / torch.sqrt(alpha) * (current_samples - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            else:
                # Generate completely from diffusion if no previous samples
                current_samples = self.diffusion.sample(num_samples)
                
            if return_intermediate:
                samples['diffusion'] = current_samples.cpu()
        
        # Stage 3: GAN enhancement
        if use_gan and self.gan is not None and self.is_trained['gan']:
            if current_samples is not None:
                # Use samples from previous stage to condition GAN
                self.gan.discriminator.eval()
                
                # Iterative refinement with GAN
                for _ in range(5):  # Few iterations to refine
                    with torch.no_grad():
                        # Get criticism from discriminator
                        d_score = self.gan.discriminator(current_samples)
                    
                    # Update samples to increase discriminator score
                    grad_steps = 3
                    for _ in range(grad_steps):
                        current_samples.requires_grad_(True)
                        d_score = self.gan.discriminator(current_samples)
                        
                        # Compute gradients of samples with respect to discriminator score
                        gradients = torch.autograd.grad(
                            outputs=d_score,
                            inputs=current_samples,
                            grad_outputs=torch.ones_like(d_score),
                            create_graph=False,
                            retain_graph=False
                        )[0]
                        
                        # Update samples to increase discriminator score
                        with torch.no_grad():
                            current_samples = current_samples + 0.01 * gradients
                            current_samples = current_samples.detach()
            else:
                # Generate completely from GAN if no previous samples
                current_samples = self.gan.generate_samples(num_samples)
                
            if return_intermediate:
                samples['gan'] = current_samples.cpu()
        
        # Return samples
        if return_intermediate:
            samples['final'] = current_samples.cpu()
            return samples
        else:
            return current_samples.cpu()
    
    def save(self, directory: str) -> None:
        """
        Save all trained models to a directory.
        
        Args:
            directory: Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        if self.vae is not None and self.is_trained['vae']:
            self.vae_trainer.save_model(os.path.join(directory, 'vae.pt'))
        
        if self.diffusion is not None and self.is_trained['diffusion']:
            self.diffusion.save_model(os.path.join(directory, 'diffusion.pt'))
        
        if self.gan is not None and self.is_trained['gan']:
            self.gan.save_model(os.path.join(directory, 'gan.pt'))
        
        # Save configuration
        torch.save({
            'input_dim': self.input_dim,
            'vae_config': self.vae_config,
            'diffusion_config': self.diffusion_config,
            'gan_config': self.gan_config,
            'is_trained': self.is_trained
        }, os.path.join(directory, 'hhg_config.pt'))
    
    def load(self, directory: str) -> None:
        """
        Load all models from a directory.
        
        Args:
            directory: Directory to load models from
        """
        # Load configuration
        config_path = os.path.join(directory, 'hhg_config.pt')
        if os.path.exists(config_path):
            config_dict = torch.load(config_path, map_location=self.device)
            self.input_dim = config_dict['input_dim']
            self.vae_config = config_dict['vae_config']
            self.diffusion_config = config_dict['diffusion_config']
            self.gan_config = config_dict['gan_config']
            self.is_trained = config_dict['is_trained']
            
            # Reinitialize models with loaded config
            self._init_models()
        
        # Load VAE if it exists
        vae_path = os.path.join(directory, 'vae.pt')
        if os.path.exists(vae_path) and self.vae is not None and self.vae_trainer is not None:
            self.vae_trainer.load_model(vae_path)
            self.is_trained['vae'] = True
        
        # Load Diffusion if it exists
        diffusion_path = os.path.join(directory, 'diffusion.pt')
        if os.path.exists(diffusion_path) and self.diffusion is not None:
            self.diffusion.load_model(diffusion_path)
            self.is_trained['diffusion'] = True
        
        # Load GAN if it exists
        gan_path = os.path.join(directory, 'gan.pt')
        if os.path.exists(gan_path) and self.gan is not None:
            self.gan.load_model(gan_path)
            self.is_trained['gan'] = True 