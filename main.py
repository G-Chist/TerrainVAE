from __future__ import print_function
import argparse
import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont

from betterconf import Config, field
from betterconf.config import AbstractProvider
from betterconf.caster import to_int, to_bool, to_list, to_float

from terrain_dataset import TerrainDataset

from classify_terrain_tensor import count_features_by_class, classify_terrain_tensor, terrain_counts_tensor

from queue import Queue

import json


HYPERPARAMS_FILE = r"C:\Users\79140\PycharmProjects\TerrainVAE\hyperparams.json"

checkpoint_path = "checkpoints/vae_cnn_classic.pt"


class JSONProvider(AbstractProvider):

    def __init__(self):
        self.HYPERPARAMS_FILE = HYPERPARAMS_FILE
        try:
            with open(self.HYPERPARAMS_FILE, "r") as f:
                self._settings = json.load(f)
        except FileNotFoundError:
            self.HYPERPARAMS_FILE = r"/home/matvei/PycharmProjects/TerrainVAE/hyperparams.json"
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

try:
    use_accel = not hpcfg.no_accel and torch.accelerator.is_available()
except AttributeError:
    use_accel = not hpcfg.no_accel and torch.cuda.is_available()

torch.manual_seed(hpcfg.seed)

if use_accel:
    try:
        device = torch.accelerator.current_accelerator()
    except:
        device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_accel else {}

try:
    dataset = TerrainDataset(root_dir=hpcfg.data_path,
                             img_size=hpcfg.img_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=hpcfg.batch_size,
                                               shuffle=True,
                                               drop_last=True)
except FileNotFoundError:
    print("Dataset not found!")

try:
    n_fixed = min(8, len(dataset))  # fixed examples
    fixed_data = torch.stack([dataset[i] for i in range(n_fixed)]).unsqueeze(1).to(device)  # [n,1,H,W]
except NameError:
    print("Variable dataset is not defined!")

# test_loader = torch.utils.data.DataLoader(dataset,
#                                          batch_size=hpcfg.batch_size,
#                                          shuffle=False,
#                                          drop_last=True)


# Compute traversability map using BFS with a threshold
def compute_traversable(grid, thresh=0.1, max_iters=1000):
    """
    Vectorized flood fill on GPU.
    grid: 2D tensor [H,W] on any device, values in [0,1].
    Returns tensor of same shape: 0.0 reachable, 1.0 unreachable.
    """
    H, W = grid.shape
    device = grid.device

    # Initialize reachable mask
    min_val = grid.min()
    reachable = (grid == min_val)

    # Structuring element (4-neighbor)
    kernel = torch.tensor([[0,1,0],
                           [1,0,1],
                           [0,1,0]], device=device).float().unsqueeze(0).unsqueeze(0)

    grid = grid.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    reachable = reachable.unsqueeze(0).unsqueeze(0).float()

    for _ in range(max_iters):
        # dilate reachable region
        neigh = F.conv2d(reachable, kernel, padding=1)
        # candidate pixels: adjacent to reachable ones
        mask = (neigh > 0) & (reachable == 0)

        # check height constraint
        diff = torch.abs(grid - grid[reachable.bool()].mean())  # local diff approx
        new_reachable = (mask & (diff <= thresh))
        if not new_reachable.any():
            break
        reachable[new_reachable] = 1.0

    return 1.0 - reachable.squeeze(0).squeeze(0)


def batch_traversable(batch, thresh=0.1, max_iters=200):
    """
    batch: [B,1,H,W]
    returns: [B,1,H,W]
    """
    B, _, H, W = batch.shape
    device = batch.device

    min_vals = batch.view(B, -1).min(dim=1)[0].view(B,1,1,1)
    reachable = (batch == min_vals).float()

    kernel = torch.tensor([[0,1,0],
                           [1,0,1],
                           [0,1,0]], device=device).float().view(1,1,3,3)

    grid = batch.clone()

    # Flatten batch into channels for grouped convolution
    x = reachable.view(1, B, H, W)

    for _ in range(max_iters):
        neigh = F.conv2d(x, kernel.repeat(B,1,1,1), padding=1, groups=B)
        mask = (neigh > 0) & (x == 0)
        diff = torch.abs(grid - grid * reachable).masked_fill(~mask.view(B,1,H,W), float('inf'))
        new = (mask.view(B,1,H,W) & (diff <= thresh)).float()
        if new.sum() == 0:
            break
        reachable = torch.clamp(reachable + new, 0, 1)
        x = reachable.view(1, B, H, W)

    return 1.0 - reachable


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, hpcfg.nf_min, 4, stride=2, padding=1),
            nn.BatchNorm2d(hpcfg.nf_min),
            nn.LeakyReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hpcfg.nf_min, hpcfg.nf_min * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hpcfg.nf_min * 2),
            nn.LeakyReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(hpcfg.nf_min * 2, hpcfg.nf_min * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hpcfg.nf_min * 4),
            nn.LeakyReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(hpcfg.nf_min * 4, hpcfg.nf_min * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(hpcfg.nf_min * 8),
            nn.LeakyReLU()
        )

        # Determine flattened dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, hpcfg.img_size, hpcfg.img_size)
            x = self.enc4(self.enc3(self.enc2(self.enc1(dummy))))
            _, _, H, W = x.shape
            self.H, self.W = H, W
            self.flat_dim = hpcfg.nf_min * 8 * H * W

        # Conditional latent mappings
        cond_dim = 10
        self.fc_mu = nn.Linear(self.flat_dim + cond_dim, hpcfg.latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim + cond_dim, hpcfg.latent_dim)
        self.fc_decode = nn.Linear(hpcfg.latent_dim + cond_dim, self.flat_dim)

        # Decoder (replaces ConvTranspose2d with Upsample + Conv2d)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hpcfg.nf_min * 8, hpcfg.nf_min * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(hpcfg.nf_min * 4),
            nn.LeakyReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hpcfg.nf_min * 4, hpcfg.nf_min * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(hpcfg.nf_min * 2),
            nn.LeakyReLU()
        )
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hpcfg.nf_min * 2, hpcfg.nf_min, 3, stride=1, padding=1),
            nn.BatchNorm2d(hpcfg.nf_min),
            nn.LeakyReLU()
        )
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hpcfg.nf_min, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Final resize for exact shape
        self.final_resize = nn.Upsample(size=(hpcfg.img_size, hpcfg.img_size), mode='bilinear', align_corners=False)

    def encode(self, x):
        xh = self.enc4(self.enc3(self.enc2(self.enc1(x))))
        xh = torch.flatten(xh, start_dim=1)

        batch_size = x.shape[0]
        cond = torch.stack([
            terrain_counts_tensor(x[i, 0], num_classes=10, device=x.device)
            for i in range(batch_size)
        ])

        xc = torch.cat([xh, cond], dim=1)
        mu = self.fc_mu(xc)
        logvar = self.fc_logvar(xc)
        return mu, logvar, cond

    def encode_injected(self, x, injected_cond):
        with torch.no_grad():
            # Encode with injected condition vector
            xh = self.enc4(self.enc3(self.enc2(self.enc1(x))))
            xh = torch.flatten(xh, start_dim=1)
            zc = torch.cat([xh, injected_cond], dim=1)
            mu = self.fc_mu(zc)
            logvar = self.fc_logvar(zc)
        return mu, logvar, injected_cond

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        zc = torch.cat([z, cond], dim=1)
        xh = F.leaky_relu(self.fc_decode(zc))
        xh = xh.view(-1, hpcfg.nf_min * 8, self.H, self.W)

        xh = self.dec1(xh)
        xh = self.dec2(xh)
        xh = self.dec3(xh)
        xh = self.dec4(xh)
        xh = self.final_resize(xh)
        return xh

    def forward(self, x):
        mu, logvar, cond = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar, cond


model = CVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=hpcfg.learning_rate)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, travmaps, x, mu, logvar, kt=0.5, kOriginalBCE=0.5):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')*kOriginalBCE

    x_binary = batch_traversable(x)  # original travmaps

    BCE_travmaps = F.binary_cross_entropy(travmaps, x_binary, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD + kt*BCE_travmaps, BCE, KLD, BCE_travmaps*kt


def save_combined_with_labels(data, recon_batch, travmaps, save_path, n=8):
    data = data[:n].cpu()
    recon_batch = recon_batch[:n].detach().cpu()
    travmaps = travmaps[:n].detach().cpu()
    extracted = batch_traversable(data)

    img_h, img_w = hpcfg.img_size, hpcfg.img_size
    # Convert each tensor [1,H,W] to PIL Image
    def tensor_to_pil(t):
        t = t.squeeze(0).numpy() * 255
        return Image.fromarray(t.astype('uint8')).convert("RGB")

    rows = []
    labels = ["Orig", "Recon", "Trav", "Extr"]
    for i in range(n):
        imgs = [
            tensor_to_pil(data[i]),
            tensor_to_pil(recon_batch[i]),
            tensor_to_pil(travmaps[i]),
            tensor_to_pil(extracted[i])
        ]
        # Stack vertically
        row = Image.new("RGB", (img_w, img_h*4))
        for j, im in enumerate(imgs):
            row.paste(im, (0, j*img_h))
        # Draw labels on left
        draw = ImageDraw.Draw(row)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        for j, lbl in enumerate(labels):
            draw.text((0, j*img_h+2), lbl, fill=(255,0,0), font=font)
        rows.append(row)

    # Stack all rows horizontally
    total_w = img_w * n
    total_h = img_h * 4
    combined = Image.new("RGB", (total_w, total_h))
    for i, row in enumerate(rows):
        combined.paste(row, (i*img_w, 0))

    combined.save(save_path)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, cond = model(data)

        # compute traversability maps for all reconstructions in batch
        with torch.no_grad():
            traversability_maps = batch_traversable(recon_batch, thresh=0.1, max_iters=200)

        loss, _, KLD, _ = loss_function(recon_batch, traversability_maps, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx == 0:
            n = n_fixed
            traversability_maps = batch_traversable(recon_batch, thresh=0.1, max_iters=200)
            save_combined_with_labels(data, recon_batch, traversability_maps,
                                      r"C:\Users\79140\PycharmProjects\TerrainVAE\results\reconstruction_" + str(
                                          epoch) + ".png", n=n)

        if batch_idx % hpcfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t KLD: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data),
                       KLD.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


"""
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
"""


if __name__ == "__main__":
    for epoch in range(1, hpcfg.epochs + 1):
        train(epoch)
        with torch.no_grad():
            sample = torch.randn(64, hpcfg.latent_dim).to(device)

            # supply condition manually
            # example: random normalized terrain feature counts
            cond = torch.rand(64, 10, device=device)
            cond = cond / cond.sum(dim=1, keepdim=True)

            sample = model.decode(sample, cond).cpu()
            save_image(sample.view(64, 1, hpcfg.img_size, hpcfg.img_size),
                       r"C:\Users\79140\PycharmProjects\TerrainVAE\results\sample_" + str(epoch) + '.png')

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": None,
        }, r"C:\Users\79140\PycharmProjects\TerrainVAE\checkpoints\vae_cnn.pt")

        print("Weights saved!")
