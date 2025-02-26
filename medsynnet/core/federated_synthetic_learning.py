"""
Federated Synthetic Learning (FSL) module.

This module implements the privacy-preserving federated learning component of MedSynNet,
allowing multiple institutions to collaboratively train synthetic data generators
without sharing raw patient data.
"""
import torch
import numpy as np
import os
import copy
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import OrderedDict

from ..models.hybrid.hierarchical_hybrid_generator import HierarchicalHybridGenerator
from ..config import config


class Client:
    """
    Client class for federated learning.
    Represents a healthcare institution participating in the federated learning process.
    """
    
    def __init__(self, 
                client_id: int,
                data: torch.Tensor,
                model: HierarchicalHybridGenerator,
                local_epochs: int = 5,
                batch_size: int = 64,
                dp_enabled: bool = False,
                dp_epsilon: float = 3.0,
                dp_delta: float = 1e-5,
                dp_max_grad_norm: float = 1.0):
        """
        Initialize a client for federated learning.
        
        Args:
            client_id: Unique identifier for the client
            data: Client's local data
            model: Local copy of the global generator model
            local_epochs: Number of local training epochs per round
            batch_size: Batch size for local training
            dp_enabled: Whether to use differential privacy
            dp_epsilon: Epsilon value for differential privacy
            dp_delta: Delta value for differential privacy
            dp_max_grad_norm: Maximum gradient norm for differential privacy
        """
        self.client_id = client_id
        self.data = data
        self.model = model
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        
        # Differential privacy settings
        self.dp_enabled = dp_enabled
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_max_grad_norm = dp_max_grad_norm
        
        # Create data loader
        self.dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.data),
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def train(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Train the local model for a number of epochs.
        
        Args:
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing model updates and metrics
        """
        if verbose:
            print(f"Training client {self.client_id}...")
        
        # Training history
        history = self.model.train(
            train_dataloader=self.dataloader,
            sequential=True,
            vae_epochs=self.local_epochs,
            diffusion_epochs=self.local_epochs,
            gan_epochs=self.local_epochs,
            verbose=verbose
        )
        
        # Extract model updates (differences from previous state)
        if self.dp_enabled:
            # Apply differential privacy to model updates
            model_updates = self._apply_differential_privacy()
        else:
            # Without DP, just return the trained model's parameters
            model_updates = self._get_model_parameters()
        
        return {
            'client_id': self.client_id,
            'model_updates': model_updates,
            'history': history,
            'num_samples': len(self.data)
        }
    
    def update_model(self, global_model_params: Dict[str, Any]) -> None:
        """
        Update the local model with global model parameters.
        
        Args:
            global_model_params: Global model parameters
        """
        # Update VAE parameters
        if 'vae' in global_model_params and self.model.vae is not None:
            self.model.vae.load_state_dict(global_model_params['vae'])
        
        # Update Diffusion parameters
        if 'diffusion' in global_model_params and self.model.diffusion is not None:
            self.model.diffusion.model.load_state_dict(global_model_params['diffusion'])
        
        # Update GAN parameters
        if 'gan_generator' in global_model_params and self.model.gan is not None:
            self.model.gan.generator.load_state_dict(global_model_params['gan_generator'])
            
        if 'gan_discriminator' in global_model_params and self.model.gan is not None:
            self.model.gan.discriminator.load_state_dict(global_model_params['gan_discriminator'])
    
    def _get_model_parameters(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        model_params = {}
        
        # Extract VAE parameters
        if self.model.vae is not None:
            model_params['vae'] = copy.deepcopy(self.model.vae.state_dict())
        
        # Extract Diffusion parameters
        if self.model.diffusion is not None:
            model_params['diffusion'] = copy.deepcopy(self.model.diffusion.model.state_dict())
        
        # Extract GAN parameters
        if self.model.gan is not None:
            model_params['gan_generator'] = copy.deepcopy(self.model.gan.generator.state_dict())
            model_params['gan_discriminator'] = copy.deepcopy(self.model.gan.discriminator.state_dict())
        
        return model_params
    
    def _apply_differential_privacy(self) -> Dict[str, Any]:
        """
        Apply differential privacy to model updates.
        
        Returns:
            Dictionary of model parameters with differential privacy applied
        """
        model_params = self._get_model_parameters()
        
        # Calculate sensitivity and clip gradients
        for key in model_params:
            for param_name, param in model_params[key].items():
                if isinstance(param, torch.Tensor):
                    # Clip gradients to bound sensitivity
                    norm = torch.norm(param)
                    if norm > self.dp_max_grad_norm:
                        param.mul_(self.dp_max_grad_norm / norm)
                    
                    # Add Gaussian noise for differential privacy
                    noise_scale = (2 * self.dp_max_grad_norm * np.sqrt(2 * np.log(1.25 / self.dp_delta))) / self.dp_epsilon
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(noise)
                    
                    model_params[key][param_name] = param
        
        return model_params


class FederatedSyntheticLearning:
    """
    Federated Synthetic Learning implementation.
    Coordinates the federated learning process across multiple healthcare institutions.
    """
    
    def __init__(self,
                input_dim: int,
                num_clients: int = 5,
                fraction_fit: float = 1.0,
                min_fit_clients: int = 2,
                min_available_clients: int = 2,
                local_epochs: int = 5,
                batch_size: int = 64,
                privacy_config: Optional[Dict[str, Any]] = None,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Federated Synthetic Learning coordinator.
        
        Args:
            input_dim: Dimensionality of input/output data
            num_clients: Number of clients (institutions)
            fraction_fit: Fraction of clients to select for training each round
            min_fit_clients: Minimum number of clients to select for training
            min_available_clients: Minimum number of available clients required
            local_epochs: Number of local training epochs per round
            batch_size: Batch size for local training
            privacy_config: Differential privacy configuration
            device: Device to use for training
        """
        self.input_dim = input_dim
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.device = device
        
        # Initialize privacy config
        if privacy_config is None:
            privacy_config = config.get('fsl.privacy', {})
        self.privacy_config = privacy_config
        
        # Initialize global model
        self.global_model = HierarchicalHybridGenerator(
            input_dim=input_dim,
            device=device
        )
        
        # Initialize clients
        self.clients = []
        
        # Initialize training history
        self.history = {
            'round': [],
            'num_clients': [],
            'client_losses': [],
            'global_metrics': []
        }
    
    def add_client(self, client_id: int, data: torch.Tensor) -> None:
        """
        Add a client to the federated learning system.
        
        Args:
            client_id: Unique identifier for the client
            data: Client's local data
        """
        # Create local model for the client
        client_model = HierarchicalHybridGenerator(
            input_dim=self.input_dim,
            device=self.device
        )
        
        # Initialize client model with global model parameters
        client_model_params = self._get_global_model_parameters()
        
        # Create client
        client = Client(
            client_id=client_id,
            data=data,
            model=client_model,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            dp_enabled=self.privacy_config.get('differential_privacy', False),
            dp_epsilon=self.privacy_config.get('epsilon', 3.0),
            dp_delta=self.privacy_config.get('delta', 1e-5),
            dp_max_grad_norm=self.privacy_config.get('max_grad_norm', 1.0)
        )
        
        # Update client model with global parameters
        client.update_model(client_model_params)
        
        # Add client to list
        self.clients.append(client)
    
    def initialize_with_synthetic_data(self, num_samples_per_client: int = 1000) -> None:
        """
        Initialize clients with synthetic data for development/testing.
        
        Args:
            num_samples_per_client: Number of synthetic samples per client
        """
        # Generate random data for each client
        for i in range(self.num_clients):
            data = torch.randn(num_samples_per_client, self.input_dim).to(self.device)
            self.add_client(i, data)
    
    def train_round(self, round_num: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Conduct one round of federated learning.
        
        Args:
            round_num: Current round number
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with round metrics
        """
        if len(self.clients) < self.min_available_clients:
            raise ValueError(f"Not enough clients. Found {len(self.clients)}, "
                           f"need at least {self.min_available_clients}")
        
        # Select clients for this round
        num_clients_to_fit = max(
            self.min_fit_clients,
            int(self.fraction_fit * len(self.clients))
        )
        
        # Randomly select clients
        selected_clients = np.random.choice(
            self.clients,
            size=num_clients_to_fit,
            replace=False
        ).tolist()
        
        if verbose:
            print(f"Round {round_num}: Selected {len(selected_clients)} clients for training")
        
        # Train selected clients
        client_results = []
        for client in selected_clients:
            result = client.train(verbose=verbose)
            client_results.append(result)
        
        # Aggregate model updates
        aggregated_params = self._aggregate_model_updates(client_results)
        
        # Update global model
        self._update_global_model(aggregated_params)
        
        # Evaluate global model
        global_metrics = self._evaluate_global_model()
        
        # Update history
        client_losses = [res['history'] for res in client_results]
        self.history['round'].append(round_num)
        self.history['num_clients'].append(len(selected_clients))
        self.history['client_losses'].append(client_losses)
        self.history['global_metrics'].append(global_metrics)
        
        # Distribute global model to all clients
        self._distribute_global_model()
        
        return {
            'round': round_num,
            'num_clients': len(selected_clients),
            'client_losses': client_losses,
            'global_metrics': global_metrics
        }
    
    def train(self, rounds: int = 10, verbose: bool = True) -> Dict[str, List[Any]]:
        """
        Train the federated learning system for a number of rounds.
        
        Args:
            rounds: Number of training rounds
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        if verbose:
            print(f"Starting federated training for {rounds} rounds with {len(self.clients)} clients")
        
        for r in range(1, rounds + 1):
            self.train_round(r, verbose=verbose)
            
            if verbose and r % 1 == 0:
                metrics = self.history['global_metrics'][-1]
                print(f"Round {r}/{rounds} completed. "
                      f"Global metrics: {metrics}")
        
        return self.history
    
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """
        Generate synthetic samples using the global model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        return self.global_model.generate(num_samples)
    
    def save(self, directory: str) -> None:
        """
        Save the global model and federation metadata.
        
        Args:
            directory: Directory to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save global model
        self.global_model.save(os.path.join(directory, 'global_model'))
        
        # Save federation metadata
        torch.save({
            'input_dim': self.input_dim,
            'num_clients': self.num_clients,
            'fraction_fit': self.fraction_fit,
            'min_fit_clients': self.min_fit_clients,
            'min_available_clients': self.min_available_clients,
            'local_epochs': self.local_epochs,
            'batch_size': self.batch_size,
            'privacy_config': self.privacy_config,
            'history': self.history
        }, os.path.join(directory, 'fsl_metadata.pt'))
    
    def load(self, directory: str) -> None:
        """
        Load the global model and federation metadata.
        
        Args:
            directory: Directory to load from
        """
        # Load global model
        self.global_model.load(os.path.join(directory, 'global_model'))
        
        # Load federation metadata
        metadata_path = os.path.join(directory, 'fsl_metadata.pt')
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
            self.input_dim = metadata['input_dim']
            self.num_clients = metadata['num_clients']
            self.fraction_fit = metadata['fraction_fit']
            self.min_fit_clients = metadata['min_fit_clients']
            self.min_available_clients = metadata['min_available_clients']
            self.local_epochs = metadata['local_epochs']
            self.batch_size = metadata['batch_size']
            self.privacy_config = metadata['privacy_config']
            self.history = metadata['history']
    
    def _get_global_model_parameters(self) -> Dict[str, Any]:
        """
        Get the current global model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        model_params = {}
        
        # Extract VAE parameters
        if self.global_model.vae is not None:
            model_params['vae'] = copy.deepcopy(self.global_model.vae.state_dict())
        
        # Extract Diffusion parameters
        if self.global_model.diffusion is not None:
            model_params['diffusion'] = copy.deepcopy(self.global_model.diffusion.model.state_dict())
        
        # Extract GAN parameters
        if self.global_model.gan is not None:
            model_params['gan_generator'] = copy.deepcopy(self.global_model.gan.generator.state_dict())
            model_params['gan_discriminator'] = copy.deepcopy(self.global_model.gan.discriminator.state_dict())
        
        return model_params
    
    def _update_global_model(self, aggregated_params: Dict[str, Any]) -> None:
        """
        Update the global model with aggregated parameters.
        
        Args:
            aggregated_params: Aggregated model parameters
        """
        # Update VAE parameters
        if 'vae' in aggregated_params and self.global_model.vae is not None:
            self.global_model.vae.load_state_dict(aggregated_params['vae'])
        
        # Update Diffusion parameters
        if 'diffusion' in aggregated_params and self.global_model.diffusion is not None:
            self.global_model.diffusion.model.load_state_dict(aggregated_params['diffusion'])
        
        # Update GAN parameters
        if 'gan_generator' in aggregated_params and self.global_model.gan is not None:
            self.global_model.gan.generator.load_state_dict(aggregated_params['gan_generator'])
            
        if 'gan_discriminator' in aggregated_params and self.global_model.gan is not None:
            self.global_model.gan.discriminator.load_state_dict(aggregated_params['gan_discriminator'])
    
    def _distribute_global_model(self) -> None:
        """
        Distribute the global model to all clients.
        """
        global_params = self._get_global_model_parameters()
        
        for client in self.clients:
            client.update_model(global_params)
    
    def _aggregate_model_updates(self, client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate model updates from clients using federated averaging.
        
        Args:
            client_results: List of client training results
            
        Returns:
            Aggregated model parameters
        """
        # Get total number of samples across all clients
        total_samples = sum(result['num_samples'] for result in client_results)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Get keys for each model component
        model_keys = set()
        for result in client_results:
            model_keys.update(result['model_updates'].keys())
        
        # For each model component (VAE, Diffusion, GAN)
        for key in model_keys:
            # Skip if this model component is not in all clients
            if not all(key in result['model_updates'] for result in client_results):
                continue
                
            # Get all state dicts for this component
            all_state_dicts = [result['model_updates'][key] for result in client_results]
            
            # Initialize aggregated state dict with zeros
            aggregated_state_dict = copy.deepcopy(all_state_dicts[0])
            for param_name in aggregated_state_dict:
                if isinstance(aggregated_state_dict[param_name], torch.Tensor):
                    aggregated_state_dict[param_name] = torch.zeros_like(aggregated_state_dict[param_name])
            
            # Weighted average of parameters
            for i, result in enumerate(client_results):
                state_dict = result['model_updates'][key]
                weight = result['num_samples'] / total_samples
                
                for param_name, param in state_dict.items():
                    if isinstance(param, torch.Tensor):
                        aggregated_state_dict[param_name] += param * weight
            
            aggregated_params[key] = aggregated_state_dict
        
        return aggregated_params
    
    def _evaluate_global_model(self) -> Dict[str, float]:
        """
        Evaluate the global model with basic metrics.
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Generate some samples to check model integrity
        try:
            samples = self.global_model.generate(100)
            return {
                'model_integrity': 1.0,
                'sample_mean': float(samples.mean()),
                'sample_std': float(samples.std())
            }
        except Exception as e:
            return {
                'model_integrity': 0.0,
                'error': str(e)
            } 