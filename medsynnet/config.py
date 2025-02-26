"""
Configuration settings for MedSynNet framework.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Base directory
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR.parent / 'data'

# Default configuration
DEFAULT_CONFIG = {
    # General settings
    "random_seed": 42,
    "device": "cuda",  # 'cuda' or 'cpu'
    "verbose": True,
    
    # Data settings
    "data_dir": str(DATA_DIR),
    "results_dir": str(BASE_DIR.parent / 'results'),
    "train_test_split": 0.8,
    "validation_split": 0.1,
    
    # Hierarchical Hybrid Generator (HHG) settings
    "hhg": {
        # VAE settings
        "vae": {
            "enabled": True,
            "latent_dim": 128,
            "hidden_dims": [256, 512, 256],
            "learning_rate": 1e-4,
            "batch_size": 64,
            "epochs": 100,
            "beta": 1.0,  # KL divergence weight
            "dropout_rate": 0.1,
        },
        
        # Diffusion model settings
        "diffusion": {
            "enabled": True,
            "noise_steps": 1000,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "hidden_dims": [256, 512, 256],
            "learning_rate": 1e-4,
            "batch_size": 64,
            "epochs": 100,
        },
        
        # GAN settings
        "gan": {
            "enabled": True,
            "latent_dim": 128,
            "generator_hidden_dims": [256, 512, 256],
            "discriminator_hidden_dims": [256, 512, 256],
            "learning_rate_g": 1e-4,
            "learning_rate_d": 4e-4,
            "batch_size": 64,
            "epochs": 100,
            "gradient_penalty_weight": 10.0,
        },
    },
    
    # Federated Synthetic Learning (FSL) settings
    "fsl": {
        "enabled": True,
        "num_clients": 5,
        "fraction_fit": 1.0,
        "min_fit_clients": 2,
        "min_available_clients": 2,
        "rounds": 10,
        "local_epochs": 5,
        "privacy": {
            "differential_privacy": True,
            "epsilon": 3.0,
            "delta": 1e-5,
            "max_grad_norm": 1.0,
        },
    },
    
    # Self-Adaptive Data Quality Controller (SA-DQC) settings
    "sa_dqc": {
        "enabled": True,
        "quality_threshold": 0.7,
        "contrastive_batch_size": 32,
        "contrastive_temperature": 0.1,
        "outlier_detection": {
            "contamination": 0.05,
            "n_estimators": 100,
        },
        "feedback_frequency": 10,  # epochs
    },
    
    # Evaluation settings
    "evaluation": {
        "metrics": ["log_likelihood", "wasserstein_distance", "privacy_score", "clinical_validity"],
        "n_samples": 1000,
        "batch_size": 64,
    },
}


class Config:
    """Configuration class to manage all settings."""
    
    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with optional custom settings.
        
        Args:
            custom_config: Dictionary of custom configuration parameters to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if custom_config:
            self._update_config(self.config, custom_config)
    
    def _update_config(self, default_config: Dict[str, Any], custom_config: Dict[str, Any]) -> None:
        """
        Recursively update configuration dictionary.
        
        Args:
            default_config: Default configuration dictionary
            custom_config: Custom configuration dictionary
        """
        for key, value in custom_config.items():
            if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                self._update_config(default_config[key], value)
            else:
                default_config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        if len(keys) == 1:
            self.config[key] = value
            return
        
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access to configuration."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-like setting of configuration."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()


# Global configuration instance
config = Config() 