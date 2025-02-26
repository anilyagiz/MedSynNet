"""
MedSynNet: Main integration module.

This module implements the main MedSynNet class that integrates all three core components:
1. Hierarchical Hybrid Generator (HHG)
2. Federated Synthetic Learning (FSL)
3. Self-Adaptive Data Quality Controller (SA-DQC)

Together, these components create a comprehensive framework for generating
high-quality synthetic healthcare data in a privacy-preserving manner.
"""
import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from ..config import config
from ..models.hybrid.hierarchical_hybrid_generator import HierarchicalHybridGenerator
from .federated_synthetic_learning import FederatedSyntheticLearning
from .self_adaptive_quality_controller import SelfAdaptiveQualityController
from ..utils.data_utils import TabularDataProcessor, load_dataset, save_dataset


class MedSynNet:
    """
    MedSynNet: A Modular Synthetic Healthcare Data Generation Framework.
    
    Integrates HHG, FSL, and SA-DQC components to create a comprehensive
    system for generating high-quality, privacy-preserving synthetic data.
    """
    
    def __init__(self, 
                data_processor: Optional[TabularDataProcessor] = None,
                use_federated: bool = True,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the MedSynNet framework.
        
        Args:
            data_processor: Tabular data processor for data preprocessing
            use_federated: Whether to use federated learning
            device: Device to use for computation
        """
        self.data_processor = data_processor
        self.use_federated = use_federated
        self.device = device
        
        # Initialize components to None
        self.generator = None
        self.federated = None
        self.quality_controller = None
        
        # Flag to track if the system is trained
        self.is_trained = False
    
    def load_data(self, 
                filepath: str,
                target_column: Optional[str] = None,
                categorical_columns: Optional[List[str]] = None,
                numerical_columns: Optional[List[str]] = None,
                normalization: str = 'standard',
                missing_value_strategy: str = 'mean',
                **kwargs) -> pd.DataFrame:
        """
        Load and preprocess dataset.
        
        Args:
            filepath: Path to the dataset file
            target_column: Name of target column (if any)
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            normalization: Type of normalization ('standard', 'minmax', or None)
            missing_value_strategy: Strategy for missing values
            **kwargs: Additional arguments for loading data
            
        Returns:
            Loaded DataFrame
        """
        # Load data
        data = load_dataset(filepath, **kwargs)
        
        # Initialize data processor if not already
        if self.data_processor is None:
            self.data_processor = TabularDataProcessor(
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns,
                target_column=target_column,
                normalization=normalization,
                missing_value_strategy=missing_value_strategy
            )
            
            # Fit the processor on the data
            self.data_processor.fit(data)
        
        return data
    
    def setup(self, 
             input_dim: Optional[int] = None,
             num_clients: int = 5,
             sa_dqc_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up the MedSynNet framework components.
        
        Args:
            input_dim: Dimensionality of input/output data
            num_clients: Number of clients for federated learning
            sa_dqc_config: Configuration for SA-DQC
        """
        # Determine input dimension
        if input_dim is None:
            if self.data_processor is not None:
                input_dim = self.data_processor.total_dimensions
            else:
                raise ValueError("Input dimension must be provided if data processor is not initialized")
        
        # Initialize generator
        if self.use_federated:
            # Initialize federated learning
            self.federated = FederatedSyntheticLearning(
                input_dim=input_dim,
                num_clients=num_clients,
                device=self.device
            )
            
            # Get the global model as our generator
            self.generator = self.federated.global_model
        else:
            # Initialize standalone generator
            self.generator = HierarchicalHybridGenerator(
                input_dim=input_dim,
                device=self.device
            )
        
        # Initialize quality controller
        if sa_dqc_config is None:
            sa_dqc_config = config.get('sa_dqc', {})
            
        self.quality_controller = SelfAdaptiveQualityController(
            input_dim=input_dim,
            contrastive_batch_size=sa_dqc_config.get('contrastive_batch_size', 32),
            contrastive_temperature=sa_dqc_config.get('contrastive_temperature', 0.1),
            outlier_contamination=sa_dqc_config.get('outlier_contamination', 0.05),
            quality_threshold=sa_dqc_config.get('quality_threshold', 0.7),
            device=self.device
        )
    
    def distribute_data(self, 
                       data: Union[pd.DataFrame, torch.Tensor],
                       num_clients: Optional[int] = None,
                       method: str = 'iid',
                       split_ratio: Optional[List[float]] = None) -> None:
        """
        Distribute data across clients for federated learning.
        
        Args:
            data: Input data to distribute
            num_clients: Number of clients
            method: Distribution method ('iid', 'non-iid', 'dirichlet')
            split_ratio: Ratio of data to distribute to each client
        """
        if not self.use_federated:
            raise ValueError("Federated learning is not enabled")
        
        if self.federated is None:
            raise ValueError("Federated learning component not initialized. Call setup() first.")
        
        # Preprocess data if needed
        if isinstance(data, pd.DataFrame):
            if self.data_processor is None:
                raise ValueError("Data processor is not initialized. Call load_data() first.")
            
            processed_data = torch.tensor(
                self.data_processor.transform(data),
                dtype=torch.float32
            ).to(self.device)
        else:
            processed_data = data.to(self.device)
        
        # Use number of clients from federated component if not specified
        if num_clients is None:
            num_clients = self.federated.num_clients
        
        # Calculate default split ratio if not provided
        if split_ratio is None:
            split_ratio = [1.0 / num_clients] * num_clients
        
        # Distribute data based on method
        if method == 'iid':
            # IID distribution: randomly shuffle and split
            indices = torch.randperm(len(processed_data))
            
            start_idx = 0
            for i in range(num_clients):
                # Calculate number of samples for this client
                num_samples = int(len(processed_data) * split_ratio[i])
                
                # Handle last client
                if i == num_clients - 1:
                    end_idx = len(processed_data)
                else:
                    end_idx = start_idx + num_samples
                
                # Get client data
                client_indices = indices[start_idx:end_idx]
                client_data = processed_data[client_indices]
                
                # Add client to federated system
                self.federated.add_client(i, client_data)
                
                start_idx = end_idx
                
        elif method == 'non-iid':
            # Non-IID distribution: sort by first feature and split
            sorted_indices = torch.argsort(processed_data[:, 0])
            
            start_idx = 0
            for i in range(num_clients):
                # Calculate number of samples for this client
                num_samples = int(len(processed_data) * split_ratio[i])
                
                # Handle last client
                if i == num_clients - 1:
                    end_idx = len(processed_data)
                else:
                    end_idx = start_idx + num_samples
                
                # Get client data
                client_indices = sorted_indices[start_idx:end_idx]
                client_data = processed_data[client_indices]
                
                # Add client to federated system
                self.federated.add_client(i, client_data)
                
                start_idx = end_idx
                
        elif method == 'dirichlet':
            # Dirichlet distribution for heterogeneous data split
            from numpy.random import dirichlet
            
            # Create random partitions with Dirichlet distribution
            alpha = 0.5  # Concentration parameter (lower = more heterogeneous)
            proportions = dirichlet([alpha] * num_clients, size=processed_data.shape[1])
            
            # Assign each sample to a client based on proportions
            client_data = [[] for _ in range(num_clients)]
            
            for i in range(len(processed_data)):
                # Choose client for this sample based on first feature's proportions
                p = proportions[0]
                client_idx = np.random.choice(num_clients, p=p)
                client_data[client_idx].append(i)
            
            # Add clients to federated system
            for i in range(num_clients):
                if len(client_data[i]) > 0:
                    client_indices = torch.tensor(client_data[i], dtype=torch.long)
                    self.federated.add_client(i, processed_data[client_indices])
                else:
                    # Add client with random subset if no data assigned
                    random_indices = torch.randperm(len(processed_data))[:10]
                    self.federated.add_client(i, processed_data[random_indices])
        else:
            raise ValueError(f"Unknown distribution method: {method}")
    
    def train(self,
             data: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
             train_federated: Optional[bool] = None,
             federated_rounds: int = 10,
             generator_config: Optional[Dict[str, Any]] = None,
             train_quality_controller: bool = True,
             quality_controller_epochs: int = 10,
             quality_feedback: bool = True,
             verbose: bool = True) -> Dict[str, Any]:
        """
        Train the MedSynNet framework.
        
        Args:
            data: Input data for training
            train_federated: Whether to train in federated mode
            federated_rounds: Number of federated learning rounds
            generator_config: Configuration for generator training
            train_quality_controller: Whether to train the quality controller
            quality_controller_epochs: Number of epochs to train quality controller
            quality_feedback: Whether to use quality feedback during training
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        # Validate state
        if self.generator is None:
            raise ValueError("Generator not initialized. Call setup() first.")
        
        # Process data if provided
        if data is not None:
            if isinstance(data, pd.DataFrame):
                if self.data_processor is None:
                    raise ValueError("Data processor not initialized. Call load_data() first.")
                
                processed_data = torch.tensor(
                    self.data_processor.transform(data),
                    dtype=torch.float32
                ).to(self.device)
            else:
                processed_data = data.to(self.device)
        else:
            processed_data = None
        
        # Determine training mode
        if train_federated is None:
            train_federated = self.use_federated
        
        # Training history
        history = {}
        
        # Train in federated mode
        if train_federated and self.federated is not None:
            if verbose:
                print("Training in federated mode...")
            
            # Train federated learning system
            federated_history = self.federated.train(
                rounds=federated_rounds,
                verbose=verbose
            )
            
            history['federated'] = federated_history
        
        # Train in standalone mode
        else:
            if processed_data is None:
                raise ValueError("Data must be provided for standalone training")
                
            if verbose:
                print("Training in standalone mode...")
            
            # Create data loader
            train_dataset = torch.utils.data.TensorDataset(processed_data)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=True
            )
            
            # Use default generator config if not provided
            if generator_config is None:
                generator_config = {}
            
            # Train generator
            generator_history = self.generator.train(
                train_dataloader=train_dataloader,
                sequential=True,
                verbose=verbose,
                **generator_config
            )
            
            history['generator'] = generator_history
        
        # Train quality controller if enabled
        if train_quality_controller and self.quality_controller is not None and processed_data is not None:
            if verbose:
                print("Training quality controller...")
            
            self.quality_controller.fit(
                real_data=processed_data,
                epochs=quality_controller_epochs
            )
            
            # Generate initial synthetic data for quality assessment
            if verbose:
                print("Generating synthetic data for quality assessment...")
            
            synthetic_data = self.generate(
                num_samples=len(processed_data),
                apply_quality_filter=False
            )
            
            # Evaluate quality
            quality_metrics = self.quality_controller.evaluate(
                real_data=processed_data,
                synthetic_data=synthetic_data,
                return_detailed_metrics=True
            )
            
            history['quality'] = quality_metrics
            
            if verbose:
                print(f"Initial quality assessment: {quality_metrics}")
            
            # Apply quality feedback if enabled
            if quality_feedback:
                if verbose:
                    print("Applying quality feedback...")
                
                # Filter samples based on quality
                filtered_data = self.quality_controller.filter_samples(
                    real_data=processed_data,
                    synthetic_data=synthetic_data
                )
                
                if len(filtered_data) > 0:
                    # Create feedback dataloader
                    feedback_dataset = torch.utils.data.TensorDataset(filtered_data)
                    feedback_dataloader = torch.utils.data.DataLoader(
                        feedback_dataset,
                        batch_size=64,
                        shuffle=True
                    )
                    
                    # Fine-tune generator with filtered data
                    if train_federated and self.federated is not None:
                        # For federated learning, fine-tune global model
                        self.generator.train(
                            train_dataloader=feedback_dataloader,
                            sequential=True,
                            verbose=verbose,
                            vae_epochs=5,
                            diffusion_epochs=5,
                            gan_epochs=5
                        )
                    else:
                        # For standalone model, fine-tune directly
                        self.generator.train(
                            train_dataloader=feedback_dataloader,
                            sequential=True,
                            verbose=verbose,
                            vae_epochs=5,
                            diffusion_epochs=5,
                            gan_epochs=5
                        )
                    
                    # Re-evaluate quality
                    synthetic_data = self.generate(
                        num_samples=len(processed_data),
                        apply_quality_filter=False
                    )
                    
                    quality_metrics = self.quality_controller.evaluate(
                        real_data=processed_data,
                        synthetic_data=synthetic_data,
                        return_detailed_metrics=True
                    )
                    
                    history['quality_after_feedback'] = quality_metrics
                    
                    if verbose:
                        print(f"Quality after feedback: {quality_metrics}")
        
        # Mark as trained
        self.is_trained = True
        
        return history
    
    def generate(self, 
                num_samples: int, 
                apply_quality_filter: bool = True,
                as_dataframe: bool = False) -> Union[torch.Tensor, pd.DataFrame]:
        """
        Generate synthetic data.
        
        Args:
            num_samples: Number of samples to generate
            apply_quality_filter: Whether to apply quality filtering
            as_dataframe: Whether to return as pandas DataFrame
            
        Returns:
            Generated synthetic data
        """
        # Validate state
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Generate data
        if self.use_federated and self.federated is not None:
            synthetic_data = self.federated.generate_samples(num_samples)
        else:
            synthetic_data = self.generator.generate(num_samples)
        
        # Apply quality filtering if enabled
        if apply_quality_filter and self.quality_controller is not None:
            # We need real data for quality comparison
            # Here we're just using the synthetic data itself as a reference
            # In practice, you would use a reference real dataset
            synthetic_data, _ = self.quality_controller.detect_anomalies(synthetic_data)
            
            # Generate more samples if needed
            if len(synthetic_data) < num_samples:
                additional_samples = num_samples - len(synthetic_data)
                
                # Generate additional samples
                if self.use_federated and self.federated is not None:
                    additional_data = self.federated.generate_samples(additional_samples * 2)
                else:
                    additional_data = self.generator.generate(additional_samples * 2)
                
                # Filter additional samples
                additional_data, _ = self.quality_controller.detect_anomalies(additional_data)
                
                # Take only what we need
                if len(additional_data) > 0:
                    additional_data = additional_data[:min(additional_samples, len(additional_data))]
                    synthetic_data = torch.cat([synthetic_data, additional_data], dim=0)
        
        # Convert to DataFrame if requested
        if as_dataframe and self.data_processor is not None:
            return self.data_processor.inverse_transform(synthetic_data.cpu().numpy())
        else:
            return synthetic_data
    
    def evaluate(self, 
                real_data: Union[pd.DataFrame, torch.Tensor],
                synthetic_data: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
                num_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate synthetic data quality.
        
        Args:
            real_data: Real data for comparison
            synthetic_data: Synthetic data to evaluate (if None, will generate)
            num_samples: Number of samples to generate if synthetic_data is None
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Validate state
        if self.quality_controller is None:
            raise ValueError("Quality controller not initialized. Call setup() first.")
        
        # Process real data if needed
        if isinstance(real_data, pd.DataFrame):
            if self.data_processor is None:
                raise ValueError("Data processor not initialized. Call load_data() first.")
            
            real_processed = torch.tensor(
                self.data_processor.transform(real_data),
                dtype=torch.float32
            ).to(self.device)
        else:
            real_processed = real_data.to(self.device)
        
        # Generate synthetic data if not provided
        if synthetic_data is None:
            synthetic_processed = self.generate(num_samples, apply_quality_filter=False)
        else:
            # Process synthetic data if needed
            if isinstance(synthetic_data, pd.DataFrame):
                if self.data_processor is None:
                    raise ValueError("Data processor not initialized. Call load_data() first.")
                
                synthetic_processed = torch.tensor(
                    self.data_processor.transform(synthetic_data),
                    dtype=torch.float32
                ).to(self.device)
            else:
                synthetic_processed = synthetic_data.to(self.device)
        
        # Evaluate quality
        metrics = self.quality_controller.evaluate(
            real_data=real_processed,
            synthetic_data=synthetic_processed,
            return_detailed_metrics=True
        )
        
        return metrics
    
    def save(self, directory: str) -> None:
        """
        Save the MedSynNet framework.
        
        Args:
            directory: Directory to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save generator
        if self.use_federated and self.federated is not None:
            self.federated.save(os.path.join(directory, 'federated'))
        elif self.generator is not None:
            self.generator.save(os.path.join(directory, 'generator'))
        
        # Save quality controller
        if self.quality_controller is not None:
            self.quality_controller.save(os.path.join(directory, 'quality_controller.pt'))
        
        # Save data processor if available
        if self.data_processor is not None:
            import pickle
            with open(os.path.join(directory, 'data_processor.pkl'), 'wb') as f:
                pickle.dump(self.data_processor, f)
        
        # Save configuration
        torch.save({
            'is_trained': self.is_trained,
            'use_federated': self.use_federated,
            'device': self.device
        }, os.path.join(directory, 'config.pt'))
    
    def load(self, directory: str) -> None:
        """
        Load the MedSynNet framework.
        
        Args:
            directory: Directory to load from
        """
        # Load configuration
        config_path = os.path.join(directory, 'config.pt')
        if os.path.exists(config_path):
            config_dict = torch.load(config_path, map_location=self.device)
            self.is_trained = config_dict['is_trained']
            self.use_federated = config_dict['use_federated']
            self.device = config_dict['device']
        
        # Load data processor if available
        data_processor_path = os.path.join(directory, 'data_processor.pkl')
        if os.path.exists(data_processor_path):
            import pickle
            with open(data_processor_path, 'rb') as f:
                self.data_processor = pickle.load(f)
        
        # Determine input dimension
        if self.data_processor is not None:
            input_dim = self.data_processor.total_dimensions
        else:
            # Try to infer from saved models
            input_dim = None
        
        # Set up components
        self.setup(input_dim=input_dim)
        
        # Load generator
        if self.use_federated and self.federated is not None:
            federated_path = os.path.join(directory, 'federated')
            if os.path.exists(federated_path):
                self.federated.load(federated_path)
                self.generator = self.federated.global_model
        elif self.generator is not None:
            generator_path = os.path.join(directory, 'generator')
            if os.path.exists(generator_path):
                self.generator.load(generator_path)
        
        # Load quality controller
        quality_controller_path = os.path.join(directory, 'quality_controller.pt')
        if os.path.exists(quality_controller_path) and self.quality_controller is not None:
            self.quality_controller.load(quality_controller_path) 