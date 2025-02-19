import torch
import pytorch_lightning as pl
from medsynnet import HHG, FSL, SADQC

def main():
    # Model parameters
    input_dim = 784  # Example: For MNIST-like data
    latent_dim = 128
    hidden_dim = 256
    learning_rate = 0.0002
    
    # Create models
    hhg = HHG(input_dim=input_dim, latent_dim=latent_dim, learning_rate=learning_rate)
    fsl = FSL(hhg)
    sadqc = SADQC(input_dim=input_dim, hidden_dim=hidden_dim)
    
    # Data loaders (example)
    # Create your own data loader in real application
    train_loader = torch.utils.data.DataLoader(
        # Dataset here
        batch_size=32,
        shuffle=True
    )
    
    # Create trainer for training
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices=1
    )
    
    # Local training
    trainer.fit(hhg, train_loader)
    
    # Start federated learning (example)
    # Multiple clients in real application
    fsl.start_client(server_address="[::]:8080")
    
    # Quality control
    for batch in train_loader:
        # Generate synthetic data
        synthetic_batch = hhg(batch)
        
        # Quality assessment
        quality_metrics = sadqc.evaluate_batch(synthetic_batch, batch)
        
        if quality_metrics['is_accepted']:
            print("Batch accepted:", quality_metrics)
        else:
            print("Batch rejected:", quality_metrics)
        
        # Update quality model
        sadqc.train_quality_model(batch, synthetic_batch)

if __name__ == "__main__":
    main() 