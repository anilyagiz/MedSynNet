# MedSynNet: Scaling Synthetic Healthcare Data Generation

MedSynNet is a scalable framework for generating high-quality synthetic healthcare data while maintaining privacy. It integrates multiple deep generative models with federated learning and adaptive quality control.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

MedSynNet addresses the challenge of generating synthetic healthcare data that maintains both statistical fidelity and privacy. The framework consists of three main components:

1. **Hierarchical Hybrid Generator (HHG)**: A multi-stage generative model that combines VAE, GAN, and Diffusion models to leverage their complementary strengths.

2. **Federated Synthetic Learning (FSL)**: Enables collaborative training across healthcare institutions without sharing raw patient data.

3. **Self-Adaptive Data Quality Controller (SA-DQC)**: Continuously monitors and improves the quality of generated data based on multiple evaluation metrics.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           MedSynNet                                 │
├────────────────┬─────────────────────────┬──────────────────────────┤
│                │                         │                          │
│ Hierarchical   │    Federated           │   Self-Adaptive          │
│ Hybrid         │    Synthetic           │   Data Quality           │
│ Generator      │    Learning            │   Controller             │
│                │                         │                          │
├────────────────┼─────────────────────────┼──────────────────────────┤
│                │                         │                          │
│ ┌──────┐       │  ┌────────┐             │ ┌──────────────┐         │
│ │ VAE  │       │  │ Client │             │ │ Quality      │         │
│ └──┬───┘       │  │   1    │             │ │ Metrics      │         │
│    ▼           │  └────┬───┘             │ └───────┬──────┘         │
│ ┌──────┐       │       │                 │         │                │
│ │ Diff │       │       ▼                 │         ▼                │
│ └──┬───┘       │  ┌────────┐             │ ┌──────────────┐         │
│    ▼           │  │ Global │◄────────────┼─┤ Anomaly      │         │
│ ┌──────┐       │  │ Model  │             │ │ Detection    │         │
│ │ GAN  │       │  └────┬───┘             │ └───────┬──────┘         │
│ └──────┘       │       │                 │         │                │
│                │       ▼                 │         ▼                │
│                │  ┌────────┐             │ ┌──────────────┐         │
│                │  │ Client │             │ │ Feedback     │         │
│                │  │   N    │             │ │ Loop         │         │
│                │  └────────┘             │ └──────────────┘         │
└────────────────┴─────────────────────────┴──────────────────────────┘
```

## Installation

```bash
git clone https://github.com/username/medsynnet.git
cd medsynnet
pip install -e .
```

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

## Quick Start

### Initialize Configuration

```bash
python -m medsynnet.cli init --output config.yaml
```

### Train a Model

```bash
python -m medsynnet.cli train --data your_data.csv --model-type hybrid --output ./output
```

### Generate Synthetic Data

```bash
python -m medsynnet.cli generate --model ./output/hybrid_model.pt --n-samples 1000 --output synthetic_data.csv
```

### Evaluate Synthetic Data

```bash
python -m medsynnet.cli evaluate --real-data real_data.csv --synthetic-data synthetic_data.csv --output evaluation
```

### Compare Multiple Generators

```bash
python -m medsynnet.cli compare --real-data real_data.csv --synthetic-data model1_data.csv model2_data.csv --names "VAE" "HHG" --output comparison
```

## Command-Line Interface

MedSynNet provides a comprehensive command-line interface for all operations:

```
usage: medsynnet.cli [-h] {train,generate,evaluate,compare,init} ...

MedSynNet: A framework for synthetic healthcare data generation

positional arguments:
  {train,generate,evaluate,compare,init}
                        Command to execute
    train               Train a synthetic data generator model
    generate            Generate synthetic data using a trained model
    evaluate            Evaluate synthetic data quality
    compare             Compare multiple synthetic data generators
    init                Initialize a configuration file

optional arguments:
  -h, --help            show this help message and exit
```

## Federated Learning

To train a model using federated learning:

```bash
python -m medsynnet.cli train --data your_data.csv --federated --n-clients 5 --output federated_output
```

## Hierarchical Hybrid Generator

The HHG combines multiple generative models in sequence:

1. **Variational Autoencoder (VAE)**: Learns a low-dimensional representation of the data.
2. **Diffusion Model**: Refines the generated samples with iterative denoising.
3. **Generative Adversarial Network (GAN)**: Enhances realism through adversarial training.

This architecture leverages the unique strengths of each model:
- VAE provides stable training and good representation learning
- Diffusion models excel at capturing complex distributions
- GANs improve sample quality and fine details

## Evaluation Metrics

MedSynNet provides comprehensive evaluation of synthetic data:

### Statistical Similarity
- Wasserstein distance
- Energy distance
- Correlation matrix similarity
- Kolmogorov-Smirnov tests

### Privacy Assessment
- Distance to closest record (DCR)
- Membership inference attack resistance
- Attribute inference risk

### Utility Metrics
- Machine learning utility
- Predictive performance preservation
- Downstream task effectiveness

## Quality Dashboard

The evaluation dashboard provides visualizations for:

- Feature distributions comparison
- Correlation heatmaps
- Dimension reduction plots (PCA, t-SNE)
- Privacy and utility metrics
- Overall quality radar chart

## Self-Adaptive Quality Control

The SA-DQC component:

- Monitors generated data quality in real-time
- Detects and filters anomalous samples
- Provides feedback to improve generator performance
- Adapts quality thresholds based on domain requirements

## API Documentation

For detailed API documentation, see the [API Reference](docs/api_reference.md).

## Examples

### Training a VAE Model

```python
from medsynnet.models.vae.tabular_vae import TabularVAE
from medsynnet.utils.data_processor import DataProcessor

# Load and preprocess data
data_processor = DataProcessor()
data = data_processor.load_data("your_data.csv")
processed_data = data_processor.preprocess(data)

# Initialize and train VAE
vae = TabularVAE(input_dim=processed_data.shape[1], hidden_dims=[128, 64], latent_dim=32)
vae.train(processed_data, epochs=50)

# Generate synthetic data
synthetic_data = vae.generate(n_samples=1000)
```

### Using the Complete MedSynNet Framework

```python
from medsynnet.core.medsynnet import MedSynNet
import pandas as pd

# Load data
data = pd.read_csv("your_data.csv")

# Initialize MedSynNet
medsynnet = MedSynNet(use_federated=False)

# Fit on real data
medsynnet.fit(data)

# Generate synthetic data
synthetic_data = medsynnet.generate(n_samples=1000)

# Evaluate quality
evaluation_results = medsynnet.evaluate(synthetic_data, real_data=data)
```
