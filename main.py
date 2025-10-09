from __future__ import print_function
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

from classify_terrain_tensor import count_features_by_class, classify_terrain_tensor, terrain_counts_tensor

import json


HYPERPARAMS_FILE = r"C:\Users\79140\PycharmProjects\TerrainVAE\hyperparams.json"

checkpoint_path = "checkpoints/vae_cnn.pt"


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
    kernel1 = field("kernel1", provider=provider, caster=to_int)
    kernel2 = field("kernel2", provider=provider, caster=to_int)
    kernel3 = field("kernel3", provider=provider, caster=to_int)
    kernel4 = field("kernel4", provider=provider, caster=to_int)


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

        # Convolutional Encoder
        self.conv1 = nn.Conv2d(1, hpcfg.nf_min, kernel_size=hpcfg.kernel1, stride=1, padding=hpcfg.kernel1 // 2)
        self.conv2 = nn.Conv2d(hpcfg.nf_min, hpcfg.nf_min * 2, kernel_size=hpcfg.kernel2, stride=1, padding=hpcfg.kernel2 // 2)
        self.conv3 = nn.Conv2d(hpcfg.nf_min * 2, hpcfg.nf_min * 4, kernel_size=hpcfg.kernel3, stride=1, padding=hpcfg.kernel3 // 2)
        self.conv4 = nn.Conv2d(hpcfg.nf_min * 4, hpcfg.nf_min * 8, kernel_size=hpcfg.kernel4, stride=1, padding=hpcfg.kernel4 // 2)

        self.batch1 = nn.BatchNorm2d(hpcfg.nf_min)
        self.batch2 = nn.BatchNorm2d(hpcfg.nf_min * 2)
        self.batch3 = nn.BatchNorm2d(hpcfg.nf_min * 4)
        self.batch4 = nn.BatchNorm2d(hpcfg.nf_min * 8)

        # Count mapping
        self.count_fc = nn.Linear(10, hpcfg.latent_dim)

        # Flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, hpcfg.img_size, hpcfg.img_size)
            x = F.leaky_relu(self.batch1(self.conv1(dummy)))
            x = F.leaky_relu(self.batch2(self.conv2(x)))
            x = F.leaky_relu(self.batch3(self.conv3(x)))
            x = F.leaky_relu(self.batch4(self.conv4(x)))
            _, _, self.conv_out_h, self.conv_out_w = x.shape
            self.flatten_dim = x.numel()

        # Fully connected encoder
        self.e_fc1 = nn.Linear(self.flatten_dim, hpcfg.pre_latent)
        self.e_fc21 = nn.Linear(hpcfg.pre_latent, hpcfg.latent_dim)
        self.e_fc22 = nn.Linear(hpcfg.pre_latent, hpcfg.latent_dim)

        # Fully connected decoder
        self.d_fc1 = nn.Linear(hpcfg.latent_dim + 10, hpcfg.pre_latent)  # input dim increased
        self.d_fc2 = nn.Linear(hpcfg.pre_latent, self.flatten_dim)

        # Convolutional decoder
        self.deconv1 = nn.ConvTranspose2d(hpcfg.nf_min * 8, hpcfg.nf_min * 4, kernel_size=hpcfg.kernel4, stride=1, padding=hpcfg.kernel4 // 2)
        self.deconv2 = nn.ConvTranspose2d(hpcfg.nf_min * 4, hpcfg.nf_min * 2, kernel_size=hpcfg.kernel3, stride=1, padding=hpcfg.kernel3 // 2)
        self.deconv3 = nn.ConvTranspose2d(hpcfg.nf_min * 2, hpcfg.nf_min, kernel_size=hpcfg.kernel2, stride=1, padding=hpcfg.kernel2 // 2)
        self.deconv4 = nn.Conv2d(hpcfg.nf_min, 1, kernel_size=hpcfg.kernel1, stride=1, padding=hpcfg.kernel1 // 2)

        self.dbatch1 = nn.BatchNorm2d(hpcfg.nf_min * 4)
        self.dbatch2 = nn.BatchNorm2d(hpcfg.nf_min * 2)
        self.dbatch3 = nn.BatchNorm2d(hpcfg.nf_min)

    def encode(self, x):
        # Convolutional feature extraction
        xh = F.leaky_relu(self.batch1(self.conv1(x)))
        
        xh = F.leaky_relu(self.batch2(self.conv2(xh)))
        
        xh = F.leaky_relu(self.batch3(self.conv3(xh)))
        
        xh = F.leaky_relu(self.batch4(self.conv4(xh)))

        xh = torch.flatten(xh, start_dim=1)
        xh = F.leaky_relu(self.e_fc1(xh))

        mu = self.e_fc21(xh)
        logvar = self.e_fc22(xh)

        # Compute terrain counts and concatenate
        batch_size = x.shape[0]
        counts_batch = torch.stack([
            terrain_counts_tensor(x[i, 0], num_classes=10, device=x.device)
            for i in range(batch_size)
        ])
        mu = torch.cat([mu, counts_batch], dim=1)
        logvar = torch.cat([logvar, counts_batch], dim=1)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        xh = F.leaky_relu(self.d_fc1(z))
        xh = F.leaky_relu(self.d_fc2(xh))
        
        xh = xh.view(-1, hpcfg.nf_min * 8, self.conv_out_h, self.conv_out_w)
        
        xh = F.leaky_relu(self.dbatch1(self.deconv1(xh)))
        
        xh = F.leaky_relu(self.dbatch2(self.deconv2(xh)))
        
        xh = F.leaky_relu(self.dbatch3(self.deconv3(xh)))
        
        xh = torch.sigmoid(self.deconv4(xh))

        return xh

    def forward(self, x):
        mu, logvar = self.encode(x)
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


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, _, KLD = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % hpcfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t KLD: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data),
                       KLD.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, _, KLD = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(hpcfg.batch_size, 1, hpcfg.img_size, hpcfg.img_size)[:n]])
                os.makedirs('results', exist_ok=True)
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, hpcfg.epochs + 1):
        train(epoch)
        with torch.no_grad():
            sample = torch.randn(64, hpcfg.latent_dim+10).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, hpcfg.img_size, hpcfg.img_size),
                       'results/sample_' + str(epoch) + '.png')

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": None,
        }, f"checkpoints/vae_cnn.pt")

        print("Weights saved!")
