import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel
import tensorflow as tf

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_var = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)

class DiffusionModel(nn.Module):
    def __init__(self, input_dim):
        super(DiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x, timesteps):
        t_emb = self.time_embed(timesteps.float().unsqueeze(-1))
        x = x + t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class HierarchicalHybridGenerator:
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # VAE components
        self.vae_encoder = VAEEncoder(input_dim, latent_dim)
        self.vae_decoder = VAEDecoder(latent_dim, input_dim)
        
        # Diffusion model
        self.diffusion = DiffusionModel(input_dim)
        
        # GAN components
        self.generator = Generator(latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def vae_forward(self, x):
        mu, log_var = self.vae_encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.vae_decoder(z)
        return recon, mu, log_var

    def generate_synthetic_data(self, batch_size):
        # Stage 1: VAE encoding
        z = torch.randn(batch_size, self.latent_dim)
        vae_output = self.vae_decoder(z)
        
        # Stage 2: Diffusion refinement
        timesteps = torch.randint(0, 1000, (batch_size,))
        diffusion_output = self.diffusion(vae_output, timesteps)
        
        # Stage 3: GAN enhancement
        gan_output = self.generator(z)
        
        # Combine outputs with weighted average
        alpha = 0.4  # VAE ağırlığı
        beta = 0.3   # Diffusion ağırlığı
        gamma = 0.3  # GAN ağırlığı
        
        final_output = (alpha * vae_output + 
                       beta * diffusion_output + 
                       gamma * gan_output)
        
        return final_output 