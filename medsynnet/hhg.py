import torch
import torch.nn as nn
import pytorch_lightning as pl

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        return self.mu(x), self.log_var(x)

class DiffusionModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.diffusion = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    
    def forward(self, x, t):
        # t: noise level
        return self.diffusion(x + t * torch.randn_like(x))

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.discriminator(x)

class HHG(pl.LightningModule):
    def __init__(self, input_dim=784, latent_dim=128, learning_rate=0.0002):
        super().__init__()
        self.save_hyperparameters()
        
        # Alt modüller
        self.encoder = VAEEncoder(input_dim, latent_dim)
        self.diffusion = DiffusionModel(latent_dim)
        self.generator = Generator(latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim)
        
        self.learning_rate = learning_rate
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # VAE aşaması
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        
        # Difüzyon aşaması
        t = torch.rand(z.shape[0], 1, device=z.device)
        z_refined = self.diffusion(z, t)
        
        # GAN aşaması
        return self.generator(z_refined)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs = batch
        
        # VAE loss
        if optimizer_idx == 0:
            mu, log_var = self.encoder(real_imgs)
            z = self.reparameterize(mu, log_var)
            z_refined = self.diffusion(z, torch.rand(z.shape[0], 1, device=z.device))
            recon = self.generator(z_refined)
            
            recon_loss = nn.MSELoss()(recon, real_imgs)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            vae_loss = recon_loss + 0.1 * kl_loss
            self.log('vae_loss', vae_loss)
            return vae_loss
        
        # Generator loss
        if optimizer_idx == 1:
            z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim, device=self.device)
            fake_imgs = self.generator(z)
            g_loss = -torch.mean(torch.log(self.discriminator(fake_imgs) + 1e-8))
            self.log('g_loss', g_loss)
            return g_loss
        
        # Discriminator loss
        if optimizer_idx == 2:
            z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim, device=self.device)
            fake_imgs = self.generator(z)
            
            real_loss = -torch.mean(torch.log(self.discriminator(real_imgs) + 1e-8))
            fake_loss = -torch.mean(torch.log(1 - self.discriminator(fake_imgs.detach()) + 1e-8))
            
            d_loss = real_loss + fake_loss
            self.log('d_loss', d_loss)
            return d_loss
    
    def configure_optimizers(self):
        vae_opt = torch.optim.Adam(list(self.encoder.parameters()) + 
                                 list(self.diffusion.parameters()), lr=self.learning_rate)
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        
        return [vae_opt, gen_opt, dis_opt], [] 