import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from main import CVAE
from main import hpcfg
from PIL import Image
import os
from math import tan, radians
from classify_terrain_tensor import count_features_by_class, classify_terrain_tensor, terrain_counts_tensor
from queue import Queue


def calculate_threshold(total_height, angle):
    # angle in degrees
    # total_height in meters
    horizontal = 512.0 / 100.0  # 512/100 meters between pixels
    return horizontal * tan(radians(angle)) / total_height  # derive threshold from acceptable slope angle


def gaussian_kernel(size: int, sigma: float):
    ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur(img: torch.Tensor, blur_size: int, sigma: float):
    # img shape: (batch, channels, H, W)
    kernel = gaussian_kernel(blur_size, sigma)
    kernel = kernel.expand(img.size(1), 1, blur_size, blur_size)
    return F.conv2d(img, kernel, padding=blur_size//2, groups=img.size(1))


device = torch.device("cuda")

# Load model and checkpoints
model = CVAE().to(device)
checkpoint = torch.load("checkpoints/vae_cnn.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load one input image from inputs/
img_path = "example1.png"
img = Image.open(os.path.join("inputs", img_path)).convert("L").resize((hpcfg.img_size, hpcfg.img_size))
x = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
x_orig = x.clone()  # save copy
x = gaussian_blur(x, blur_size=33, sigma=9)  # blur user input
x = torch.minimum(x, x_orig)  # save pre-defined black boundaries, add some "slope" via blurring
x = x.to(device)

print(terrain_counts_tensor(x.squeeze(), 10, device=device))

injection = torch.tensor([0, 50, 100, 4000, 0, 0, 0, 0, 0, 0]) / x.numel()

# Encode and reconstruct
with torch.no_grad():
    mu, logvar, cond = model.encode(x)
    z = model.reparameterize(mu, logvar)
    recon = model.decode(z, cond).cpu().squeeze()

# Linearly interpolate recon with input image
BLUR_SIZE = 25
sigma = 1.5

x_blur = gaussian_blur(x.cpu(), BLUR_SIZE, sigma)
recon_t = recon.unsqueeze(0).unsqueeze(0).cpu()

# Interpolate between blurred input and reconstruction
recon_mix = torch.lerp(x_blur, recon_t, 0.95).squeeze().cpu().numpy()

# Now compute traversability map using BFS with a threshold
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

print(f"Acceptable threshold: {calculate_threshold(total_height=10, angle=20)}% = {calculate_threshold(total_height=10, angle=20)*10}m")

traversable_map = compute_traversable(torch.tensor(recon_mix, dtype=torch.float32), thresh=calculate_threshold(total_height=10, angle=20))

# Scale 3D heights
recon_scaled = recon_mix * 0.2

# 2D + 4x 3D views
fig = plt.figure(figsize=(20, 5))

# 2D original
ax0 = fig.add_subplot(1, 8, 1)
ax0.imshow(x_orig.cpu().squeeze().numpy(), cmap='gray')
ax0.set_title("Original Input")
ax0.axis('off')

ax1 = fig.add_subplot(1, 8, 2)
ax1.imshow(x.cpu().squeeze().numpy(), cmap='gray')
ax1.set_title("Original Input (Blurred)")
ax1.axis('off')

# 2D reconstructed
ax2 = fig.add_subplot(1, 8, 3)
ax2.imshow(recon_mix, cmap='gray')
ax2.set_title("Decoded Terrain (with Interpolation)")
ax2.axis('off')

# 3D views at different angles
angles = [(30, 45), (60, 30), (15, 90), (75, 60)]
for i, (elev, azim) in enumerate(angles):
    ax = fig.add_subplot(1, 8, i + 4, projection='3d')
    h, w = recon_scaled.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    ax.plot_surface(X, Y, recon_scaled, cmap='terrain', linewidth=0, antialiased=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"3D View {i+1}")
    ax.set_zlim(0, 1)
    ax.set_axis_off()

# 2D traversability map
axt = fig.add_subplot(1, 8, 8)
axt.imshow(traversable_map, cmap='gray')
axt.set_title("Traversability Map")
axt.axis('off')

plt.tight_layout()

# Save the combined figure
os.makedirs("outputs", exist_ok=True)

# Extract identifier from filename
import os, re
base = os.path.splitext(os.path.basename(img_path))[0]
digits = re.findall(r"\d+", base)
identifier = digits[-1] if digits else base

# Save figure
out_path = os.path.join("outputs", f"{identifier}_full.png")
plt.savefig(out_path, bbox_inches="tight", dpi=250)
plt.close(fig)

print(f"Saved visualization to {out_path}")
