from __future__ import print_function

import optuna

import argparse
import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from betterconf import Config, field
from betterconf.config import AbstractProvider
from betterconf.caster import to_int, to_bool, to_list, to_float

from terrain_dataset import TerrainDataset

import json

HYPERPARAMS_FILE = r"C:\Users\79140\PycharmProjects\TerrainVAE\hyperparams.json"


class JSONProvider(AbstractProvider):

    def __init__(self):
        self.HYPERPARAMS_FILE = HYPERPARAMS_FILE
        with open(self.HYPERPARAMS_FILE, "r") as f:
            self._settings = json.load(f)

    def get(self, name):
        return self._settings.get(name)


provider = JSONProvider()


class HyperparamConfig(Config):
    batch_size = field("batch_size", provider=provider, caster=to_int)
    no_accel = field("no_accel", provider=provider, caster=to_bool)
    epochs = field("epochs", provider=provider, caster=to_int)
    seed = field("seed", provider=provider, caster=to_int)
    log_interval = field("log_interval", provider=provider, caster=to_int)
    img_size = field("img_size", provider=provider, caster=to_int)
    data_path = field("data_path", provider=provider)
    latent_dim = field("latent_dim", provider=provider, caster=to_int)
    learning_rate = field("learning_rate", provider=provider, caster=to_float)
    nf_min = field("nf_min", provider=provider, caster=to_int)
    pre_latent = field("pre_latent", provider=provider, caster=to_int)


hpcfg = HyperparamConfig()

use_accel = not hpcfg.no_accel and torch.accelerator.is_available()

torch.manual_seed(hpcfg.seed)

if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_accel else {}

dataset = TerrainDataset(root_dir=hpcfg.data_path,
                         img_size=hpcfg.img_size)

train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=hpcfg.batch_size,
                                           shuffle=True,
                                           drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=hpcfg.batch_size,
                                          shuffle=False,
                                          drop_last=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Vanilla Encoder layers

        self.enc1 = nn.Linear(hpcfg.img_size ** 2, 400)
        self.enc2 = nn.Linear(400, 400)
        self.enc31 = nn.Linear(400, hpcfg.latent_dim)
        self.enc32 = nn.Linear(400, hpcfg.latent_dim)

        # Vanilla Decoder layers

        self.dec1 = nn.Linear(hpcfg.latent_dim, 400)
        self.dec2 = nn.Linear(400, 400)
        self.dec3 = nn.Linear(400, hpcfg.img_size ** 2)

        # Convolutional Encoder layers

        self.conv1 = nn.Conv2d(
            1, hpcfg.nf_min, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(
            hpcfg.nf_min, hpcfg.nf_min * 2, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(
            hpcfg.nf_min * 2, hpcfg.nf_min * 4, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(
            hpcfg.nf_min * 4, hpcfg.nf_min * 8, kernel_size=3, stride=1, padding=2)

        self.batch1 = nn.BatchNorm2d(hpcfg.nf_min)
        self.batch2 = nn.BatchNorm2d(hpcfg.nf_min * 2)
        self.batch3 = nn.BatchNorm2d(hpcfg.nf_min * 4)
        self.batch4 = nn.BatchNorm2d(hpcfg.nf_min * 8)

        # Determine flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, hpcfg.img_size, hpcfg.img_size)
            x = F.leaky_relu(self.batch1(self.conv1(dummy)))
            x = F.leaky_relu(self.batch2(self.conv2(x)))
            x = F.leaky_relu(self.batch3(self.conv3(x)))
            x = F.leaky_relu(self.batch4(self.conv4(x)))
            _, _, self.conv_out_h, self.conv_out_w = x.shape
            self.flatten_dim = x.numel()

        self.e_fc1 = nn.Linear(in_features=self.flatten_dim,
                               out_features=hpcfg.pre_latent)

        self.e_fc21 = nn.Linear(
            in_features=hpcfg.pre_latent, out_features=hpcfg.latent_dim)
        self.e_fc22 = nn.Linear(
            in_features=hpcfg.pre_latent, out_features=hpcfg.latent_dim)

        # Convolutional Decoder layers

        self.d_fc1 = nn.Linear(hpcfg.latent_dim, hpcfg.pre_latent)
        self.d_fc2 = nn.Linear(hpcfg.pre_latent, self.flatten_dim)

        self.deconv1 = nn.ConvTranspose2d(
            hpcfg.nf_min * 8, hpcfg.nf_min * 4, kernel_size=3, stride=1, padding=2)
        self.deconv2 = nn.ConvTranspose2d(
            hpcfg.nf_min * 4, hpcfg.nf_min * 2, kernel_size=5, stride=1, padding=2)
        self.deconv3 = nn.ConvTranspose2d(
            hpcfg.nf_min * 2, hpcfg.nf_min, kernel_size=5, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(
            hpcfg.nf_min, 1, kernel_size=5, stride=1, padding=2)

        self.dbatch1 = nn.BatchNorm2d(hpcfg.nf_min * 4)
        self.dbatch2 = nn.BatchNorm2d(hpcfg.nf_min * 2)
        self.dbatch3 = nn.BatchNorm2d(hpcfg.nf_min)

    def encode(self, x):
        # First convolution + batch norm + activation
        xh = self.conv1(x)
        xh = F.leaky_relu(self.batch1(xh))

        # Second convolution + batch norm + activation
        xh = self.conv2(xh)
        xh = F.leaky_relu(self.batch2(xh))

        # Third convolution + batch norm + activation
        xh = self.conv3(xh)
        xh = F.leaky_relu(self.batch3(xh))

        # Fourth convolution + batch norm + activation
        xh = self.conv4(xh)
        xh = F.leaky_relu(self.batch4(xh))

        # Flatten convolutional feature maps into a single vector
        xh = torch.flatten(xh, start_dim=1)

        # Fully connected layer before latent projection
        xh = F.leaky_relu(self.e_fc1(xh))

        # Return latent mean and log-variance
        mu = self.e_fc21(xh)
        logvar = torch.clamp(self.e_fc22(xh), -10, 10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Fully connected projection from latent to flattened conv feature map
        xh = F.leaky_relu(self.d_fc1(z))
        xh = F.leaky_relu(self.d_fc2(xh))

        # Reshape into 4D tensor for ConvTranspose2d
        xh = xh.view(-1, hpcfg.nf_min * 8, self.conv_out_h, self.conv_out_w)

        # Sequential transposed convolutional upsampling
        xh = F.leaky_relu(self.dbatch1(self.deconv1(xh)))
        xh = F.leaky_relu(self.dbatch2(self.deconv2(xh)))
        xh = F.leaky_relu(self.dbatch3(self.deconv3(xh)))

        # Final conv to reconstruct 1-channel image
        xh = torch.sigmoid(self.deconv4(xh)).clamp(1e-7, 1 - 1e-7)

        return xh

    def forward(self, x):
        mu, logvar = self.encode(x)
        print("mu mean/std:", mu.mean().item(), mu.std().item())
        print("logvar mean/std:", logvar.mean().item(), logvar.std().item())
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=hpcfg.learning_rate)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


def objective(trial):
    # Suggest hyperparameters
    hpcfg.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    hpcfg.latent_dim = trial.suggest_int("latent_dim", 4, 64)
    hpcfg.nf_min = trial.suggest_categorical("nf_min", [4, 6, 8, 10, 12, 14])
    hpcfg.pre_latent = trial.suggest_int("pre_latent", 32, 128)

    # Recreate model and optimizer with suggested params
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=hpcfg.learning_rate)

    # one short epoch to estimate performance
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(train_loader):
        if batch_idx > 10:  # limit training batches for speed
            break
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, _, _ = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / ((batch_idx + 1) * hpcfg.batch_size)
    return avg_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, timeout=None)

    print("Best trial:")
    print(study.best_trial.params)
