"""
Command-line interface for the MedSynNet framework.

This module provides a command-line interface to run the MedSynNet framework
for generating and evaluating synthetic healthcare data.
"""
import os
import sys
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from medsynnet.core.medsynnet import MedSynNet
from medsynnet.utils.data_processor import DataProcessor
from medsynnet.models.hybrid.hierarchical_hybrid_generator import HierarchicalHybridGenerator
from medsynnet.core.federated_synthetic_learning import FederatedSyntheticLearning
from medsynnet.core.self_adaptive_quality_controller import SelfAdaptiveQualityController
from medsynnet.evaluation.metrics import evaluate_synthetic_data, compare_synthetic_generators
from medsynnet.evaluation.visualization import (
    create_quality_dashboard, 
    create_comparison_dashboard,
    save_dashboard_to_pdf
)
from medsynnet.utils.config import load_config, save_config


def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="MedSynNet: A framework for synthetic healthcare data generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train a synthetic data generator model"
    )
    train_parser.add_argument(
        "--data", type=str, required=True, help="Path to input data CSV file"
    )
    train_parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    train_parser.add_argument(
        "--output", type=str, default="./output", help="Output directory"
    )
    train_parser.add_argument(
        "--model-type", type=str, choices=["vae", "gan", "diffusion", "hybrid"], 
        default="hybrid", help="Type of model to train"
    )
    train_parser.add_argument(
        "--federated", action="store_true", help="Use federated learning"
    )
    train_parser.add_argument(
        "--n-clients", type=int, default=3, help="Number of clients for federated learning"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    train_parser.add_argument(
        "--eval", action="store_true", help="Evaluate model after training"
    )
    train_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    train_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    
    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate synthetic data using a trained model"
    )
    generate_parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
    generate_parser.add_argument(
        "--n-samples", type=int, default=1000, help="Number of samples to generate"
    )
    generate_parser.add_argument(
        "--output", type=str, default="./synthetic_data.csv", help="Output file path"
    )
    generate_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    generate_parser.add_argument(
        "--use-quality-control", action="store_true", help="Apply quality control to generated data"
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate synthetic data quality"
    )
    evaluate_parser.add_argument(
        "--real-data", type=str, required=True, help="Path to real data CSV file"
    )
    evaluate_parser.add_argument(
        "--synthetic-data", type=str, required=True, help="Path to synthetic data CSV file"
    )
    evaluate_parser.add_argument(
        "--output", type=str, default="./evaluation_results", help="Output directory"
    )
    evaluate_parser.add_argument(
        "--labels-column", type=str, default=None, help="Column name for labels (for utility metrics)"
    )
    evaluate_parser.add_argument(
        "--task", type=str, choices=["classification", "regression"], 
        default="classification", help="Task type for utility metrics"
    )
    evaluate_parser.add_argument(
        "--dashboard", action="store_true", help="Generate visual dashboard"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare multiple synthetic data generators"
    )
    compare_parser.add_argument(
        "--real-data", type=str, required=True, help="Path to real data CSV file"
    )
    compare_parser.add_argument(
        "--synthetic-data", type=str, nargs="+", required=True, 
        help="Paths to synthetic data CSV files (space-separated)"
    )
    compare_parser.add_argument(
        "--names", type=str, nargs="+", help="Names of generators (space-separated)"
    )
    compare_parser.add_argument(
        "--output", type=str, default="./comparison_results", help="Output directory"
    )
    compare_parser.add_argument(
        "--labels-column", type=str, default=None, help="Column name for labels (for utility metrics)"
    )
    compare_parser.add_argument(
        "--task", type=str, choices=["classification", "regression"], 
        default="classification", help="Task type for utility metrics"
    )
    compare_parser.add_argument(
        "--dashboard", action="store_true", help="Generate visual dashboard"
    )
    
    # Initialize command
    init_parser = subparsers.add_parser(
        "init", help="Initialize a configuration file"
    )
    init_parser.add_argument(
        "--output", type=str, default="./config.yaml", help="Output configuration file path"
    )
    init_parser.add_argument(
        "--model-type", type=str, choices=["vae", "gan", "diffusion", "hybrid"], 
        default="hybrid", help="Type of model to configure"
    )
    
    return parser


def train_command(args: argparse.Namespace) -> None:
    """Handle the train command."""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Use default configuration
        from medsynnet.utils.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    
    # Log configuration
    config_path = os.path.join(args.output, "config.yaml")
    save_config(config, config_path)
    
    print(f"Loading data from {args.data}...")
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Load and preprocess data
    data = data_processor.load_data(args.data)
    processed_data = data_processor.preprocess(data)
    
    # Get features and labels if available
    if data_processor.label_column is not None:
        features = processed_data.drop(columns=[data_processor.label_column])
        labels = processed_data[data_processor.label_column]
    else:
        features = processed_data
        labels = None
    
    # Convert to numpy arrays
    X = features.to_numpy()
    y = labels.to_numpy() if labels is not None else None
    
    # Store column names for later
    feature_names = features.columns.tolist()
    
    # Save feature names and data processor state
    with open(os.path.join(args.output, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    data_processor.save(os.path.join(args.output, "data_processor.pkl"))
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize appropriate model
    if args.model_type == "hybrid":
        # Use Hierarchical Hybrid Generator
        model = HierarchicalHybridGenerator(
            input_dim=X.shape[1],
            vae_config=config["models"]["vae"],
            diffusion_config=config["models"]["diffusion"],
            gan_config=config["models"]["gan"],
            device=device
        )
        print("Initialized Hierarchical Hybrid Generator")
    
    elif args.model_type == "vae":
        # Use VAE model
        from medsynnet.models.vae.tabular_vae import TabularVAE
        model = TabularVAE(
            input_dim=X.shape[1],
            **config["models"]["vae"]
        )
        print("Initialized Variational Autoencoder")
    
    elif args.model_type == "gan":
        # Use GAN model
        from medsynnet.models.gan.tabular_gan import TabularGAN
        model = TabularGAN(
            input_dim=X.shape[1],
            **config["models"]["gan"]
        )
        print("Initialized Generative Adversarial Network")
    
    elif args.model_type == "diffusion":
        # Use Diffusion model
        from medsynnet.models.diffusion.tabular_diffusion import TabularDiffusion
        model = TabularDiffusion(
            input_dim=X.shape[1],
            **config["models"]["diffusion"]
        )
        print("Initialized Diffusion Model")
    
    # Set up federated learning if requested
    if args.federated:
        print(f"Setting up federated learning with {args.n_clients} clients...")
        
        # Create federated learning environment
        fsl = FederatedSyntheticLearning(
            model=model,
            num_clients=args.n_clients,
            client_selection_fraction=config["federated"]["client_selection_fraction"],
            differential_privacy_params=config["federated"]["differential_privacy_params"],
            device=device
        )
        
        # Split data for federated setting
        fsl.initialize_clients_with_data(X, y)
        
        # Train federated model
        print(f"Starting federated training for {args.epochs} rounds...")
        training_history = fsl.train(
            rounds=args.epochs,
            local_epochs=config["federated"]["local_epochs"],
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        
        # Save federated learning model
        model_path = os.path.join(args.output, "federated_model.pt")
        fsl.save(model_path)
        print(f"Federated model saved to {model_path}")
        
        # Get the trained global model
        model = fsl.global_model
    
    else:
        # Standard centralized training
        print(f"Starting training for {args.epochs} epochs...")
        
        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device) if y is not None else None
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(
            X_tensor, y_tensor if y_tensor is not None else X_tensor
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True
        )
        
        # Train the model
        if args.model_type == "hybrid":
            training_history = model.train(
                data_loader=data_loader,
                epochs=args.epochs,
                sequential=True,
                verbose=args.verbose
            )
        else:
            training_history = model.train(
                data_loader=data_loader,
                epochs=args.epochs,
                verbose=args.verbose
            )
        
        # Save the trained model
        model_path = os.path.join(args.output, f"{args.model_type}_model.pt")
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    # Evaluate the model if requested
    if args.eval:
        print("Evaluating model...")
        
        # Generate synthetic data
        n_samples = min(10000, X.shape[0])
        synthetic_data = model.generate(n_samples=n_samples)
        
        if isinstance(synthetic_data, torch.Tensor):
            synthetic_data = synthetic_data.cpu().numpy()
        
        # Convert synthetic data to DataFrame
        synthetic_df = pd.DataFrame(synthetic_data, columns=feature_names)
        
        # Save synthetic data
        synthetic_path = os.path.join(args.output, "synthetic_data.csv")
        synthetic_df.to_csv(synthetic_path, index=False)
        print(f"Synthetic data saved to {synthetic_path}")
        
        # Evaluate synthetic data
        results = evaluate_synthetic_data(
            real_data=X,
            synthetic_data=synthetic_data,
            labels_real=y,
            labels_synthetic=None,  # No labels for synthetic data yet
            privacy_tests=True,
            utility_tests=False  # No labels for utility tests
        )
        
        # Save evaluation results
        results_path = os.path.join(args.output, "evaluation_results.json")
        with open(results_path, "w") as f:
            # Convert numpy values to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json_results = {k: {k2: convert_numpy(v2) for k2, v2 in v.items()} 
                          if isinstance(v, dict) else convert_numpy(v) 
                          for k, v in results.items()}
            
            json.dump(json_results, f, indent=2)
        
        print(f"Evaluation results saved to {results_path}")
        
        # Generate quality dashboard
        dashboard_figures = create_quality_dashboard(
            real_data=X,
            synthetic_data=synthetic_data,
            evaluation_results=results,
            feature_names=feature_names
        )
        
        # Save dashboard to PDF
        dashboard_path = os.path.join(args.output, "quality_dashboard.pdf")
        save_dashboard_to_pdf(dashboard_figures, dashboard_path)
        print(f"Quality dashboard saved to {dashboard_path}")
        
        # Print summary
        print("\nSynthetic Data Quality Summary:")
        print(f"- Overall Quality Score: {results['overall_quality_score']:.4f}")
        print(f"- Statistical Similarity (Correlation): {results['statistical']['correlation_similarity']:.4f}")
        print(f"- Distribution Similarity (Wasserstein): {1.0/(1.0 + results['statistical']['wasserstein_distance_mean']):.4f}")
        if 'privacy' in results:
            print(f"- Privacy Score (DCR): {results['privacy']['normalized_dcr']:.4f}")
        if 'membership_inference' in results:
            print(f"- Privacy Risk: {results['membership_inference']['privacy_risk']}")
    
    print("\nTraining completed successfully.")


def generate_command(args: argparse.Namespace) -> None:
    """Handle the generate command."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading model from {args.model}...")
    
    # Determine model type from file extension
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"Error: Model file {args.model} not found.")
        sys.exit(1)
    
    # Initialize appropriate model based on filename
    if "hybrid" in model_path.stem or "medsynnet" in model_path.stem:
        model = HierarchicalHybridGenerator.load(args.model)
        print("Loaded Hierarchical Hybrid Generator")
    elif "vae" in model_path.stem:
        from medsynnet.models.vae.tabular_vae import TabularVAE
        model = TabularVAE.load(args.model)
        print("Loaded Variational Autoencoder")
    elif "gan" in model_path.stem:
        from medsynnet.models.gan.tabular_gan import TabularGAN
        model = TabularGAN.load(args.model)
        print("Loaded Generative Adversarial Network")
    elif "diffusion" in model_path.stem:
        from medsynnet.models.diffusion.tabular_diffusion import TabularDiffusion
        model = TabularDiffusion.load(args.model)
        print("Loaded Diffusion Model")
    elif "federated" in model_path.stem:
        fsl = FederatedSyntheticLearning.load(args.model)
        model = fsl.global_model
        print("Loaded Federated Model")
    else:
        print(f"Error: Could not determine model type from filename {model_path.name}.")
        print("Please use a model file with 'vae', 'gan', 'diffusion', 'hybrid', or 'federated' in the name.")
        sys.exit(1)
    
    # Look for feature names and data processor in the same directory as the model
    model_dir = model_path.parent
    feature_names = None
    data_processor = None
    
    feature_names_path = model_dir / "feature_names.json"
    data_processor_path = model_dir / "data_processor.pkl"
    
    if feature_names_path.exists():
        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)
        print(f"Loaded feature names: {len(feature_names)} features")
    
    if data_processor_path.exists():
        data_processor = DataProcessor.load(str(data_processor_path))
        print("Loaded data processor for inverse transformations")
    
    # Generate synthetic data
    print(f"Generating {args.n_samples} synthetic samples...")
    
    if args.use_quality_control and hasattr(model, 'generate_with_quality_control'):
        # Initialize quality controller if not already present
        if not hasattr(model, 'quality_controller') or model.quality_controller is None:
            quality_controller = SelfAdaptiveQualityController(input_dim=model.input_dim)
            model.quality_controller = quality_controller
            print("Initialized Self-Adaptive Quality Controller")
        
        # Generate data with quality control
        synthetic_data = model.generate_with_quality_control(n_samples=args.n_samples)
    else:
        # Standard generation
        synthetic_data = model.generate(n_samples=args.n_samples)
    
    # Convert to numpy if it's a tensor
    if isinstance(synthetic_data, torch.Tensor):
        synthetic_data = synthetic_data.cpu().numpy()
    
    # Convert to DataFrame
    if feature_names is not None:
        # Use saved feature names
        synthetic_df = pd.DataFrame(synthetic_data, columns=feature_names)
    else:
        # Generate generic feature names
        synthetic_df = pd.DataFrame(
            synthetic_data, 
            columns=[f"feature_{i+1}" for i in range(synthetic_data.shape[1])]
        )
    
    # Apply inverse transformations if data processor is available
    if data_processor is not None:
        synthetic_df = data_processor.inverse_transform(synthetic_df)
        print("Applied inverse transformations to synthetic data")
    
    # Save synthetic data
    output_path = Path(args.output)
    output_dir = output_path.parent
    os.makedirs(output_dir, exist_ok=True)
    
    synthetic_df.to_csv(args.output, index=False)
    print(f"Synthetic data saved to {args.output}")
    
    print("\nGeneration completed successfully.")


def evaluate_command(args: argparse.Namespace) -> None:
    """Handle the evaluate command."""
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading real data from {args.real_data}...")
    real_df = pd.read_csv(args.real_data)
    
    print(f"Loading synthetic data from {args.synthetic_data}...")
    synthetic_df = pd.read_csv(args.synthetic_data)
    
    # Extract labels if provided
    if args.labels_column is not None:
        if args.labels_column not in real_df.columns:
            print(f"Error: Labels column '{args.labels_column}' not found in real data.")
            sys.exit(1)
        
        if args.labels_column not in synthetic_df.columns:
            print(f"Error: Labels column '{args.labels_column}' not found in synthetic data.")
            sys.exit(1)
        
        real_labels = real_df[args.labels_column].to_numpy()
        synthetic_labels = synthetic_df[args.labels_column].to_numpy()
        
        # Remove labels from features
        real_df = real_df.drop(columns=[args.labels_column])
        synthetic_df = synthetic_df.drop(columns=[args.labels_column])
        
        print(f"Extracted labels from column '{args.labels_column}'")
    else:
        real_labels = None
        synthetic_labels = None
    
    # Get feature names
    feature_names = real_df.columns.tolist()
    
    # Convert to numpy
    real_data = real_df.to_numpy()
    synthetic_data = synthetic_df.to_numpy()
    
    print(f"Real data: {real_data.shape[0]} samples, {real_data.shape[1]} features")
    print(f"Synthetic data: {synthetic_data.shape[0]} samples, {synthetic_data.shape[1]} features")
    
    # Evaluate synthetic data
    print("Evaluating synthetic data quality...")
    results = evaluate_synthetic_data(
        real_data=real_data,
        synthetic_data=synthetic_data,
        labels_real=real_labels,
        labels_synthetic=synthetic_labels,
        task=args.task,
        privacy_tests=True,
        utility_tests=real_labels is not None
    )
    
    # Save evaluation results
    results_path = os.path.join(args.output, "evaluation_results.json")
    with open(results_path, "w") as f:
        # Convert numpy values to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_results = {k: {k2: convert_numpy(v2) for k2, v2 in v.items()} 
                        if isinstance(v, dict) else convert_numpy(v) 
                        for k, v in results.items()}
        
        json.dump(json_results, f, indent=2)
    
    print(f"Evaluation results saved to {results_path}")
    
    # Generate quality dashboard if requested
    if args.dashboard:
        print("Generating quality dashboard...")
        dashboard_figures = create_quality_dashboard(
            real_data=real_data,
            synthetic_data=synthetic_data,
            evaluation_results=results,
            feature_names=feature_names
        )
        
        # Save dashboard to PDF
        dashboard_path = os.path.join(args.output, "quality_dashboard.pdf")
        save_dashboard_to_pdf(dashboard_figures, dashboard_path)
        print(f"Quality dashboard saved to {dashboard_path}")
    
    # Print summary
    print("\nSynthetic Data Quality Summary:")
    print(f"- Overall Quality Score: {results['overall_quality_score']:.4f}")
    print(f"- Statistical Similarity (Correlation): {results['statistical']['correlation_similarity']:.4f}")
    print(f"- Distribution Similarity (Wasserstein): {1.0/(1.0 + results['statistical']['wasserstein_distance_mean']):.4f}")
    
    if 'privacy' in results:
        print(f"- Privacy Score (DCR): {results['privacy']['normalized_dcr']:.4f}")
    
    if 'membership_inference' in results:
        print(f"- Privacy Risk: {results['membership_inference']['privacy_risk']}")
    
    if 'utility' in results and 'relative_accuracy' in results['utility']:
        print(f"- Utility (Relative Accuracy): {results['utility']['relative_accuracy']:.4f}")
        print(f"- Utility (Relative F1): {results['utility']['relative_f1']:.4f}")
    
    print("\nEvaluation completed successfully.")


def compare_command(args: argparse.Namespace) -> None:
    """Handle the compare command."""
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading real data from {args.real_data}...")
    real_df = pd.read_csv(args.real_data)
    
    # Create names for synthetic datasets if not provided
    if args.names is None:
        # Use filenames as names
        args.names = [Path(path).stem for path in args.synthetic_data]
    
    if len(args.names) != len(args.synthetic_data):
        print("Error: Number of names must match number of synthetic datasets.")
        sys.exit(1)
    
    # Load synthetic datasets
    synthetic_dfs = {}
    for name, path in zip(args.names, args.synthetic_data):
        print(f"Loading synthetic data '{name}' from {path}...")
        synthetic_dfs[name] = pd.read_csv(path)
    
    # Extract labels if provided
    if args.labels_column is not None:
        if args.labels_column not in real_df.columns:
            print(f"Error: Labels column '{args.labels_column}' not found in real data.")
            sys.exit(1)
        
        real_labels = real_df[args.labels_column].to_numpy()
        
        # Check synthetic datasets for labels
        synthetic_labels = {}
        for name, df in synthetic_dfs.items():
            if args.labels_column in df.columns:
                synthetic_labels[name] = df[args.labels_column].to_numpy()
                # Remove labels from features
                synthetic_dfs[name] = df.drop(columns=[args.labels_column])
            else:
                print(f"Warning: Labels column '{args.labels_column}' not found in synthetic data '{name}'.")
        
        # Remove labels from real features
        real_df = real_df.drop(columns=[args.labels_column])
        
        print(f"Extracted labels from column '{args.labels_column}'")
    else:
        real_labels = None
        synthetic_labels = None
    
    # Get feature names
    feature_names = real_df.columns.tolist()
    
    # Convert to numpy
    real_data = real_df.to_numpy()
    synthetic_data_dict = {name: df.to_numpy() for name, df in synthetic_dfs.items()}
    
    print(f"Real data: {real_data.shape[0]} samples, {real_data.shape[1]} features")
    for name, data in synthetic_data_dict.items():
        print(f"Synthetic data '{name}': {data.shape[0]} samples, {data.shape[1]} features")
    
    # Compare synthetic data generators
    print("Comparing synthetic data generators...")
    comparison_results = compare_synthetic_generators(
        real_data=real_data,
        synthetic_data_dict=synthetic_data_dict,
        labels_real=real_labels,
        labels_synthetic_dict=synthetic_labels,
        task=args.task
    )
    
    # Save comparison results
    results_path = os.path.join(args.output, "comparison_results.json")
    with open(results_path, "w") as f:
        # Convert numpy values to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_results = {k: {k2: convert_numpy(v2) for k2, v2 in v.items()} if isinstance(v, dict) else v
                       for k, v in comparison_results.items()}
        
        json.dump(json_results, f, indent=2)
    
    print(f"Comparison results saved to {results_path}")
    
    # Generate comparison dashboard if requested
    if args.dashboard:
        print("Generating comparison dashboard...")
        dashboard_figures = create_comparison_dashboard(
            real_data=real_data,
            synthetic_data_dict=synthetic_data_dict,
            comparison_results=comparison_results,
            feature_names=feature_names
        )
        
        # Save dashboard to PDF
        dashboard_path = os.path.join(args.output, "comparison_dashboard.pdf")
        save_dashboard_to_pdf(dashboard_figures, dashboard_path)
        print(f"Comparison dashboard saved to {dashboard_path}")
    
    # Print summary
    print("\nSynthetic Data Generators Comparison Summary:")
    
    # Overall ranking
    if 'ranking' in comparison_results and 'overall' in comparison_results['ranking']:
        print("\nOverall Ranking (lower is better):")
        overall_ranking = comparison_results['ranking']['overall']
        for name, rank in sorted(overall_ranking.items(), key=lambda x: x[1]):
            print(f"{rank}. {name}")
    
    # Quality score comparison
    print("\nQuality Scores:")
    for name in synthetic_data_dict.keys():
        if name in comparison_results and 'overall_quality_score' in comparison_results[name]:
            score = comparison_results[name]['overall_quality_score']
            print(f"- {name}: {score:.4f}")
    
    print("\nComparison completed successfully.")


def init_command(args: argparse.Namespace) -> None:
    """Handle the init command."""
    # Load default configuration
    from medsynnet.utils.config import DEFAULT_CONFIG
    
    # Get specific model configuration
    if args.model_type != "hybrid":
        # Extract only the specified model configuration
        config = {
            "models": {
                args.model_type: DEFAULT_CONFIG["models"][args.model_type]
            }
        }
    else:
        # For hybrid, keep all model configurations
        config = DEFAULT_CONFIG
    
    # Save configuration to output file
    save_config(config, args.output)
    print(f"Configuration file initialized at {args.output}")
    
    # Print help message
    print("\nYou can now edit this configuration file and use it with the train command:")
    print(f"python -m medsynnet.cli train --data your_data.csv --config {args.output}")
    
    if args.model_type == "hybrid":
        print("\nThe configuration includes settings for all model types:")
        print("- Variational Autoencoder (VAE)")
        print("- Generative Adversarial Network (GAN)")
        print("- Diffusion Model")
        print("- Federated Learning")
    else:
        print(f"\nThe configuration includes settings for {args.model_type.upper()} model.")


def main():
    """Main entry point for the CLI."""
    # Set up argument parser
    parser = setup_parser()
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return
    
    # Handle commands
    command_handlers = {
        "train": train_command,
        "generate": generate_command,
        "evaluate": evaluate_command,
        "compare": compare_command,
        "init": init_command
    }
    
    if args.command in command_handlers:
        command_handlers[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 