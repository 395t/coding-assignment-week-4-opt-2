import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self._shape = shape
    
    def forward(self, x):
        return x.reshape(len(x), *self._shape)

Conv321 = functools.partial(nn.Conv2d, kernel_size=3, stride=2, padding=1)
Tconv421 = functools.partial(nn.ConvTranspose2d, kernel_size=4, stride=2, padding=1)

class VAE(nn.Module):
    """VAE architecture adopted from http://ruishu.io/2018/03/14/vae."""

    def __init__(self, z_dim: int) -> None:
        super().__init__()
        act = nn.SiLU

        # Gaussian encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(32),  act(),
            Conv321(32, 32),  nn.BatchNorm2d(32),  act(),
            Conv321(32, 64),  nn.BatchNorm2d(64),  act(),
            Conv321(64, 128), nn.BatchNorm2d(128), act(),
            Reshape([128 * 4 * 4]),
            nn.Linear(128 * 4 * 4, 1024), nn.BatchNorm1d(1024), act(),
            nn.Linear(1024, z_dim * 2)
        )

        # Gaussian decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 4 * 4 * 512), nn.BatchNorm1d(4 * 4 * 512), act(),
            Reshape([512, 4, 4]),
            Tconv421(512, 128), nn.BatchNorm2d(128), act(),
            Tconv421(128, 64),  nn.BatchNorm2d(64),  act(),
            Tconv421(64,  32),  nn.BatchNorm2d(32),  act(),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        # Compute q(z|x)
        mu, log_sig = self.encoder(x).chunk(2, dim=1)
        sig = log_sig.exp() + 1e-5

        # Reparametrization trick for z
        epsilon = torch.randn_like(mu)
        z = mu + sig * epsilon

        # Compute p(x|z)
        x_hat = self.decoder(z)

        return z, mu, sig, x_hat