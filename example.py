import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch

from medsynnet.medsynnet import MedSynNet

# Load and preprocess data
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Normalize data
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split data into multiple clients (simulating federated setup)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create three client datasets
client1_X, client2_X, y1, y2 = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
client3_X, _, y3, _ = train_test_split(client2_X, y2, test_size=0.5, random_state=42)

# Convert to PyTorch datasets
def create_dataset(X, y):
    return TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y).reshape(-1, 1)
    )

client_datasets = [
    create_dataset(client1_X, y1),
    create_dataset(client2_X, y2),
    create_dataset(client3_X, y3)
]

# Initialize MedSynNet
input_dim = X.shape[1]
medsynnet = MedSynNet(input_dim=input_dim, latent_dim=32, num_rounds=5)

# Train the framework
print("Training MedSynNet...")
medsynnet.train(client_datasets, batch_size=32, epochs=20)

# Generate synthetic data
print("\nGenerating synthetic data...")
num_synthetic_samples = 1000
synthetic_data = medsynnet.generate_synthetic_data(num_synthetic_samples)

# Save the trained model
print("\nSaving model...")
medsynnet.save_model('medsynnet_model.pth')

print("Done!") 