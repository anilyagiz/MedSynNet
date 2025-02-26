#!/usr/bin/env python
"""
MedSynNet Diabetes Example

This example demonstrates how to use the MedSynNet framework to generate
synthetic diabetes data based on the Pima Indians Diabetes Dataset.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Add the parent directory to the path to import medsynnet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medsynnet.utils.data_utils import TabularDataProcessor, load_dataset, save_dataset
from medsynnet.models.vae.tabular_vae import TabularVAE, TabularVAETrainer
from medsynnet.models.gan.tabular_gan import TabularGAN
from medsynnet.models.diffusion.tabular_diffusion import TabularDiffusion
from medsynnet.models.hybrid.hierarchical_hybrid_generator import HierarchicalHybridGenerator
from medsynnet.core.self_adaptive_quality_controller import SelfAdaptiveQualityController
from medsynnet.evaluation.metrics import evaluate_synthetic_data
from medsynnet.evaluation.visualization import (
    compare_distributions, 
    plot_correlation_comparison,
    create_quality_dashboard,
    save_dashboard_to_pdf
)

# Create directories for outputs
RESULTS_DIR = Path("results/diabetes_example")
MODEL_DIR = RESULTS_DIR / "models"
SYNTH_DATA_DIR = RESULTS_DIR / "synthetic_data"
PLOTS_DIR = RESULTS_DIR / "plots"

for dir_path in [RESULTS_DIR, MODEL_DIR, SYNTH_DATA_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def prepare_data():
    """Load and preprocess the diabetes dataset."""
    print("\n=== Preparing Data ===")
    
    # Load the data
    data_path = Path("data/diabetes.csv")
    data = load_dataset(data_path)
    print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
    
    # Define categorical and numerical columns
    categorical_columns = ["Outcome"]
    numerical_columns = [col for col in data.columns if col not in categorical_columns]
    
    # Initialize the data processor
    data_processor = TabularDataProcessor(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        normalization='standard',
        missing_value_strategy='mean'
    )
    
    # Fit the processor and transform the data
    processed_data = data_processor.fit_transform(data)
    print(f"Processed data shape: {processed_data.shape}")
    
    # Split the data into train, validation, and test sets
    train_data, test_data, val_data = data_processor.train_test_val_split(
        data, train_size=0.7, test_size=0.15, val_size=0.15
    )
    print(f"Train set: {train_data.shape}, Validation set: {val_data.shape}, Test set: {test_data.shape}")
    
    # Create data loaders
    train_loader = data_processor.create_dataloader(train_data, batch_size=64, shuffle=True)
    val_loader = data_processor.create_dataloader(val_data, batch_size=64, shuffle=False)
    test_loader = data_processor.create_dataloader(test_data, batch_size=64, shuffle=False)
    
    return data, processed_data, data_processor, train_loader, val_loader, test_loader, train_data, val_data, test_data

def train_vae(train_loader, val_loader, input_dim):
    """Train a VAE model on the diabetes dataset."""
    print("\n=== Training VAE ===")
    
    # Initialize VAE model
    vae_model = TabularVAE(
        input_dim=input_dim,
        latent_dim=8,
        hidden_dims=[32, 64, 32],
        dropout_rate=0.1,
        beta=1.0
    )
    
    # Initialize VAE trainer
    vae_trainer = TabularVAETrainer(
        model=vae_model,
        learning_rate=1e-3,
        device=device
    )
    
    # Train the VAE model
    history = vae_trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=50,
        early_stopping_patience=10,
        verbose=True
    )
    
    # Save the model
    vae_trainer.save_model(MODEL_DIR / "diabetes_vae.pt")
    print(f"VAE model saved to {MODEL_DIR / 'diabetes_vae.pt'}")
    
    return vae_trainer

def train_gan(train_loader, input_dim):
    """Train a GAN model on the diabetes dataset."""
    print("\n=== Training GAN ===")
    
    # Initialize GAN model
    gan_model = TabularGAN(
        input_dim=input_dim,
        latent_dim=8,
        generator_hidden_dims=[32, 64, 32],
        discriminator_hidden_dims=[32, 64, 32],
        dropout_rate=0.1,
        lr_g=1e-4,
        lr_d=4e-4,
        spectral_norm=True,
        gradient_penalty_weight=10.0,
        device=device
    )
    
    # Train the GAN model
    history = gan_model.train(
        dataloader=train_loader,
        epochs=100,
        d_steps=5,
        verbose=True
    )
    
    # Save the model
    gan_model.save_model(MODEL_DIR / "diabetes_gan.pt")
    print(f"GAN model saved to {MODEL_DIR / 'diabetes_gan.pt'}")
    
    return gan_model

def train_diffusion(train_loader, input_dim):
    """Train a Diffusion model on the diabetes dataset."""
    print("\n=== Training Diffusion Model ===")
    
    # Initialize Diffusion model
    diffusion_model = TabularDiffusion(
        input_dim=input_dim,
        hidden_dims=[32, 64, 32],
        noise_steps=100,
        beta_start=1e-4,
        beta_end=0.02,
        time_embedding_dim=32,
        dropout_rate=0.1,
        learning_rate=1e-3,
        device=device
    )
    
    # Train the Diffusion model
    history = diffusion_model.train(
        dataloader=train_loader,
        epochs=50,
        verbose=True
    )
    
    # Save the model
    diffusion_model.save_model(MODEL_DIR / "diabetes_diffusion.pt")
    print(f"Diffusion model saved to {MODEL_DIR / 'diabetes_diffusion.pt'}")
    
    return diffusion_model

def train_hybrid_model(train_loader, val_loader, input_dim):
    """Train a Hierarchical Hybrid Generator on the diabetes dataset."""
    print("\n=== Training Hierarchical Hybrid Generator ===")
    
    # Define configurations for each component
    vae_config = {
        "enabled": True,
        "latent_dim": 8,
        "hidden_dims": [32, 64, 32],
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 50,
        "beta": 1.0,
        "dropout_rate": 0.1
    }
    
    diffusion_config = {
        "enabled": True,
        "noise_steps": 100,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "hidden_dims": [32, 64, 32],
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 50
    }
    
    gan_config = {
        "enabled": True,
        "latent_dim": 8,
        "generator_hidden_dims": [32, 64, 32],
        "discriminator_hidden_dims": [32, 64, 32],
        "learning_rate_g": 1e-4,
        "learning_rate_d": 4e-4,
        "batch_size": 64,
        "epochs": 100,
        "gradient_penalty_weight": 10.0
    }
    
    # Initialize the Hierarchical Hybrid Generator
    hybrid_model = HierarchicalHybridGenerator(
        input_dim=input_dim,
        vae_config=vae_config,
        diffusion_config=diffusion_config,
        gan_config=gan_config,
        device=device
    )
    
    # Train the Hierarchical Hybrid Generator
    history = hybrid_model.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        vae_epochs=50,
        diffusion_epochs=50,
        gan_epochs=100,
        sequential=True,
        verbose=True
    )
    
    # Save the model
    hybrid_model.save(MODEL_DIR / "diabetes_hybrid")
    print(f"Hybrid model saved to {MODEL_DIR / 'diabetes_hybrid'}")
    
    return hybrid_model

def generate_synthetic_data(models, data_processor, num_samples=1000):
    """Generate synthetic data using each model."""
    print("\n=== Generating Synthetic Data ===")
    
    # Dictionary to store synthetic data from each model
    synthetic_data_dict = {}
    
    # Generate samples from VAE
    if 'vae' in models:
        vae_trainer = models['vae']
        vae_samples = vae_trainer.generate_samples(num_samples).cpu().numpy()
        vae_df = data_processor.inverse_transform(vae_samples)
        synthetic_data_dict['VAE'] = vae_df
        save_dataset(vae_df, SYNTH_DATA_DIR / "diabetes_vae_synthetic.csv")
        print(f"Generated {num_samples} samples using VAE")
    
    # Generate samples from GAN
    if 'gan' in models:
        gan_model = models['gan']
        gan_samples = gan_model.generate_samples(num_samples).cpu().numpy()
        gan_df = data_processor.inverse_transform(gan_samples)
        synthetic_data_dict['GAN'] = gan_df
        save_dataset(gan_df, SYNTH_DATA_DIR / "diabetes_gan_synthetic.csv")
        print(f"Generated {num_samples} samples using GAN")
    
    # Generate samples from Diffusion model
    if 'diffusion' in models:
        diffusion_model = models['diffusion']
        diffusion_samples = diffusion_model.sample(num_samples).cpu().numpy()
        diffusion_df = data_processor.inverse_transform(diffusion_samples)
        synthetic_data_dict['Diffusion'] = diffusion_df
        save_dataset(diffusion_df, SYNTH_DATA_DIR / "diabetes_diffusion_synthetic.csv")
        print(f"Generated {num_samples} samples using Diffusion model")
    
    # Generate samples from Hybrid model
    if 'hybrid' in models:
        hybrid_model = models['hybrid']
        hybrid_samples = hybrid_model.generate(num_samples).cpu().numpy()
        hybrid_df = data_processor.inverse_transform(hybrid_samples)
        synthetic_data_dict['Hybrid'] = hybrid_df
        save_dataset(hybrid_df, SYNTH_DATA_DIR / "diabetes_hybrid_synthetic.csv")
        print(f"Generated {num_samples} samples using Hierarchical Hybrid Generator")
    
    return synthetic_data_dict

def evaluate_models(original_data, synthetic_data_dict, data_processor):
    """Evaluate the quality of synthetic data generated by each model."""
    print("\n=== Evaluating Synthetic Data Quality ===")
    
    # Get the raw numpy arrays for evaluation
    real_data_np = data_processor.transform(original_data)
    
    # Evaluate each model's synthetic data
    evaluation_results = {}
    for model_name, synth_df in synthetic_data_dict.items():
        print(f"Evaluating {model_name} synthetic data...")
        synth_data_np = data_processor.transform(synth_df)
        
        # Evaluate synthetic data
        results = evaluate_synthetic_data(
            real_data=real_data_np,
            synthetic_data=synth_data_np,
            task='classification',
            privacy_tests=True,
            utility_tests=True
        )
        
        evaluation_results[model_name] = results
        
        # Create and save visualizations
        feature_names = list(original_data.columns)
        
        # Distribution comparison
        fig_dist = compare_distributions(
            real_data_np, synth_data_np, 
            feature_names=feature_names
        )
        fig_dist.savefig(PLOTS_DIR / f"{model_name}_distribution_comparison.png", dpi=300, bbox_inches='tight')
        
        # Correlation comparison
        fig_corr = plot_correlation_comparison(
            real_data_np, synth_data_np,
            feature_names=feature_names
        )
        fig_corr.savefig(PLOTS_DIR / f"{model_name}_correlation_comparison.png", dpi=300, bbox_inches='tight')
        
        # Create quality dashboard
        dashboard_figures = create_quality_dashboard(
            real_data_np, synth_data_np,
            evaluation_results[model_name],
            feature_names=feature_names
        )
        
        # Save dashboard as PDF
        save_dashboard_to_pdf(
            dashboard_figures,
            str(PLOTS_DIR / f"{model_name}_quality_dashboard.pdf")
        )
        
        print(f"Evaluation results for {model_name}:")
        print(f"  Statistical similarity: {results['statistical_similarity']['wasserstein_distance']:.4f} (Wasserstein distance)")
        print(f"  Privacy score: {results['privacy_metrics']['privacy_score']:.4f}")
        if 'utility_metrics' in results:
            print(f"  Utility score: {results['utility_metrics']['utility_score']:.4f}")
        print()
    
    return evaluation_results

def main():
    """Main function to run the diabetes example."""
    print("MedSynNet Diabetes Example")
    print("==========================")
    
    # Prepare data
    data, processed_data, data_processor, train_loader, val_loader, test_loader, train_data, val_data, test_data = prepare_data()
    input_dim = processed_data.shape[1]
    
    # Train models
    models = {}
    
    # Uncomment the models you want to train
    models['vae'] = train_vae(train_loader, val_loader, input_dim)
    models['gan'] = train_gan(train_loader, input_dim)
    models['diffusion'] = train_diffusion(train_loader, input_dim)
    models['hybrid'] = train_hybrid_model(train_loader, val_loader, input_dim)
    
    # Generate synthetic data
    synthetic_data_dict = generate_synthetic_data(models, data_processor, num_samples=1000)
    
    # Evaluate the synthetic data
    evaluation_results = evaluate_models(data, synthetic_data_dict, data_processor)
    
    print("\n=== Example Complete ===")
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main() 