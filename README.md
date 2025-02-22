# MedSynNet: A Modular Framework for Tabular Healthcare Data Generation

This project is a synthetic tabular healthcare data generation framework developed based on the IEEE paper ["Scaling Synthetic Healthcare Data Generation for AI-Driven Biomedical Informatics"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10807405).

## Features

- **Hierarchical Hybrid Generator (HHG)**
  - VAE-based feature encoding and decoding
  - Time-conditional diffusion model for data refinement
  - GAN for realism enhancement
  - Weighted hybrid combination strategy

- **Federated Synthetic Learning (FSL)**
  - Privacy-preserving distributed training
  - Decentralized model updates
  - Federated model aggregation
  - Multi-institution support

- **Self-Adaptive Data Quality Controller (SA-DQC)**
  - Contrastive learning-based quality assessment
  - Anomaly detection and filtering
  - Adaptive quality threshold updates
  - Feedback mechanism

## Usage Example

```python
from medsynnet.medsynnet import MedSynNet
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Normalize data
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split data (for federated learning simulation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
client1_X, client2_X, y1, y2 = train_test_split(X_train, y_train, test_size=0.5)

# Initialize MedSynNet
medsynnet = MedSynNet(input_dim=X.shape[1], latent_dim=32)

# Train the framework
medsynnet.train(client_datasets)

# Generate synthetic data
synthetic_data = medsynnet.generate_synthetic_data(1000)
```

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows:
.\venv\Scripts\activate
# For Linux/Mac:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- TensorFlow >= 2.8.0
- Flower >= 1.0.0 (Federated Learning)
- scikit-learn >= 0.24.2
- NumPy >= 1.19.2
- Pandas >= 1.2.4

## Supported Data Types

The framework currently supports the following tabular data types:
- Numerical features (continuous and discrete)
- Categorical features (with one-hot encoding)
- Binary features

## License

MIT

## References

1. [Scaling Synthetic Healthcare Data Generation for AI-Driven Biomedical Informatics](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10807405)
2. [GenerativeFL GitHub Repository](https://github.com/zhiyuan-ning/GenerativeFL) 