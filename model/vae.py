import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from model.nn.mlp import *


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        ### 0. configuration
        self.x_dim = 7
        self.z_dim = 2
        self.mse = nn.MSELoss()
        self.beta = 0.0

        ### 1. compute posterior: p(z|x)
        self.encoder = MLP(in_dim=self.x_dim, out_dim=self.z_dim, hidden=[32, 32])

        ### 2. compute likelihood: p(x|z)
        self.decoder = MLP(in_dim=self.z_dim, out_dim=self.x_dim, hidden=[32, 32])

    def save_model(self, model_name):
        torch.save(self.state_dict(), model_name)

    def load_model(self, file_name):
        self.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))

    def loss(self, z, x_pred, x_true):
        z_target = Normal(torch.zeros_like(z.loc), torch.ones_like(z.scale))
        loss_kl = kl_divergence(z, z_target).mean()
        loss_recon = self.mse(x_pred, x_true).mean()
        loss_latent = loss_recon + self.beta * loss_kl
        return loss_latent, loss_recon, loss_kl

    def encode(self, x):
        z = self.encoder(x)
        return Normal(z, torch.ones_like(z))

    def decode(self, z):
        return self.decoder(z)
