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


hpcfg = HyperparamConfig()


use_accel = not hpcfg.no_accel and torch.accelerator.is_available()

torch.manual_seed(hpcfg.seed)


if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_accel else {}

transform = transforms.Compose([
            transforms.Resize((hpcfg.img_size, hpcfg.img_size)) if isinstance(hpcfg.img_size, int)
                                                                else transforms.Resize(hpcfg.img_size),
            transforms.ToTensor()])

dataset = TerrainDataset(root_dir=hpcfg.data_path,
                         transform=transform)

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

        self.enc1 = nn.Linear(hpcfg.img_size**2, 400)
        self.enc2 = nn.Linear(400, 400)
        self.enc31 = nn.Linear(400, hpcfg.latent_dim)
        self.enc32 = nn.Linear(400, hpcfg.latent_dim)
        
        self.dec1 = nn.Linear(hpcfg.latent_dim, 400)
        self.dec2 = nn.Linear(400, 400)
        self.dec3 = nn.Linear(400, hpcfg.img_size**2)

    def encode(self, x):
        xh = F.relu(self.enc1(x))
        xh = F.relu(self.enc2(xh))
        return self.enc31(xh), self.enc32(xh)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        xh = F.relu(self.dec1(z))
        xh = F.relu(self.dec2(xh))
        return torch.sigmoid(self.dec3(xh))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, hpcfg.img_size**2))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=hpcfg.learning_rate)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, hpcfg.img_size**2), reduction='sum')

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
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, hpcfg.latent_dim).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, hpcfg.img_size, hpcfg.img_size),
                       'results/sample_' + str(epoch) + '.png')