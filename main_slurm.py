from __future__ import print_function
import argparse
import os
import sys
import json
import torch
import traceback

# --- Runtime-safe setup for SLURM ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
RUN_DIR = os.path.join(PROJECT_ROOT, "runs")
os.makedirs(RUN_DIR, exist_ok=True)
STDOUT_PATH = os.path.join(RUN_DIR, "stdout.log")
STDERR_PATH = os.path.join(RUN_DIR, "stderr.log")
sys.stdout = open(STDOUT_PATH, "w", buffering=1)
sys.stderr = open(STDERR_PATH, "w", buffering=1)
print(f"Logging to: {RUN_DIR}")

# --- imports ---
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont

from betterconf import Config, field
from betterconf.config import AbstractProvider
from betterconf.caster import to_int, to_bool, to_float

from terrain_dataset import TerrainDataset
from classify_terrain_tensor import count_features_by_class, classify_terrain_tensor, terrain_counts_tensor

# --- safe file handling ---
HYPERPARAMS_FILE = os.path.join(PROJECT_ROOT, "hyperparams.json")
CHECKPOINTS_DIR = os.path.join(RUN_DIR, "checkpoints")
RESULTS_DIR = os.path.join(RUN_DIR, "results")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

checkpoint_path = os.path.join(CHECKPOINTS_DIR, "vae_cnn_classic.pt")
results_path = RESULTS_DIR

# --- config provider system ---
class JSONProvider(AbstractProvider):
    def __init__(self):
        self.HYPERPARAMS_FILE = HYPERPARAMS_FILE
        try:
            with open(self.HYPERPARAMS_FILE, "r") as f:
                self._settings = json.load(f)
        except FileNotFoundError:
            self._settings = {}

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

# --- device setup ---
try:
    use_accel = not hpcfg.no_accel and torch.cuda.is_available()
except AttributeError:
    use_accel = torch.cuda.is_available()

torch.manual_seed(hpcfg.seed)
device = torch.device("cuda" if use_accel else "cpu")
print(f"Using device: {device}")

# --- data setup ---
data_abs_path = os.path.join(PROJECT_ROOT, hpcfg.data_path)
print(f"Dataset path: {data_abs_path}")

try:
    dataset = TerrainDataset(root_dir=data_abs_path, img_size=hpcfg.img_size)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=hpcfg.batch_size, shuffle=True, drop_last=True)
except FileNotFoundError:
    print("Dataset not found.")
    sys.exit(1)

n_fixed = min(8, len(dataset))
fixed_data = torch.stack([dataset[i] for i in range(n_fixed)]).unsqueeze(1).to(device)

# --- CVAE and training logic ---
def batch_traversable(batch, thresh=0.1, max_iters=200):
    B, _, H, W = batch.shape
    device = batch.device
    min_vals = batch.view(B, -1).min(dim=1)[0].view(B,1,1,1)
    reachable = (batch == min_vals).float()
    kernel = torch.tensor([[0,1,0],[1,0,1],[0,1,0]], device=device).float().view(1,1,3,3)
    grid = batch.clone()
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
        nf = hpcfg.nf_min
        self.enc1 = nn.Sequential(nn.Conv2d(1, nf, 4, 2, 1), nn.BatchNorm2d(nf), nn.LeakyReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(nf, nf*2, 4, 2, 1), nn.BatchNorm2d(nf*2), nn.LeakyReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(nf*2, nf*4, 4, 2, 1), nn.BatchNorm2d(nf*4), nn.LeakyReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(nf*4, nf*8, 4, 2, 1), nn.BatchNorm2d(nf*8), nn.LeakyReLU())
        with torch.no_grad():
            dummy = torch.zeros(1,1,hpcfg.img_size,hpcfg.img_size)
            x = self.enc4(self.enc3(self.enc2(self.enc1(dummy))))
            _, _, H, W = x.shape
            self.H, self.W = H, W
        self.flat_dim = nf*8*H*W
        cond_dim = 10
        self.fc_mu = nn.Linear(self.flat_dim + cond_dim, hpcfg.latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim + cond_dim, hpcfg.latent_dim)
        self.fc_decode = nn.Linear(hpcfg.latent_dim + cond_dim, self.flat_dim)
        self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(nf*8, nf*4, 3,1,1), nn.BatchNorm2d(nf*4), nn.LeakyReLU())
        self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(nf*4, nf*2, 3,1,1), nn.BatchNorm2d(nf*2), nn.LeakyReLU())
        self.dec3 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(nf*2, nf, 3,1,1), nn.BatchNorm2d(nf), nn.LeakyReLU())
        self.dec4 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(nf,1,3,1,1), nn.Sigmoid())
        self.final_resize = nn.Upsample(size=(hpcfg.img_size, hpcfg.img_size))
    def encode(self,x):
        xh = self.enc4(self.enc3(self.enc2(self.enc1(x))))
        xh = torch.flatten(xh, start_dim=1)
        batch_size = x.shape[0]
        cond = torch.stack([terrain_counts_tensor(x[i,0], num_classes=10, device=x.device) for i in range(batch_size)])
        xc = torch.cat([xh, cond], dim=1)
        mu = self.fc_mu(xc)
        logvar = self.fc_logvar(xc)
        return mu, logvar, cond
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z, cond):
        zc = torch.cat([z, cond], dim=1)
        xh = F.leaky_relu(self.fc_decode(zc))
        xh = xh.view(-1, hpcfg.nf_min*8, self.H, self.W)
        for block in [self.dec1, self.dec2, self.dec3, self.dec4]:
            xh = block(xh)
        xh = self.final_resize(xh)
        return xh
    def forward(self,x):
        mu,logvar,cond = self.encode(x)
        z = self.reparameterize(mu,logvar)
        recon = self.decode(z,cond)
        return recon, mu, logvar, cond

model = CVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=hpcfg.learning_rate)

def loss_function(recon_x, travmaps, x, mu, logvar, kt=0.5, kOriginalBCE=0.5):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') * kOriginalBCE
    x_binary = batch_traversable(x)
    BCE_travmaps = F.binary_cross_entropy(travmaps, x_binary, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD + kt*BCE_travmaps, BCE, KLD, BCE_travmaps*kt

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, cond = model(data)
        with torch.no_grad():
            traversability_maps = batch_traversable(recon_batch, thresh=0.1, max_iters=200)
        loss, _, KLD, _ = loss_function(recon_batch, traversability_maps, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % hpcfg.log_interval == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item()/len(data):.6f}")
    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")

if __name__ == "__main__":
    try:
        for epoch in range(1, hpcfg.epochs + 1):
            train(epoch)
            with torch.no_grad():
                sample = torch.randn(64, hpcfg.latent_dim).to(device)
                cond = torch.rand(64, 10, device=device)
                cond = cond / cond.sum(dim=1, keepdim=True)
                sample = model.decode(sample, cond).cpu()
                save_image(sample.view(64, 1, hpcfg.img_size, hpcfg.img_size),
                           os.path.join(results_path, f"sample_{epoch}.png"))
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": None,
            }, checkpoint_path)
            print("Weights saved.")
    except Exception:
        traceback.print_exc()
