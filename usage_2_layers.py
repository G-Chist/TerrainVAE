import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from main import CVAE
from main import hpcfg
from PIL import Image
import os
from classify_terrain_tensor import count_features_by_class, classify_terrain_tensor, terrain_counts_tensor

LAYER_2_SCALE = 0.15


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


def weierstrass_mandelbrot_chunked(
    x_vals, y_vals, D_list, G_list, L, gamma, M_list, n_max,
    device='cuda', chunk_size=1500, dtype=torch.float32
):
    """
    Compute seamless Weierstrass-Mandelbrot terrain in chunks.

    Parameters:
    -----------
    x_vals, y_vals : 1D torch tensors of coordinates
    D_list, G_list, M_list : lists of parameters for each WM layer
    L, gamma : float
    n_max : int
    device : 'cuda' or 'cpu'
    chunk_size : int, max rows per chunk
    """
    res_x = x_vals.size(0)
    res_y = y_vals.size(0)

    # Prepare full result on CPU
    z_full = torch.zeros((res_y, res_x), dtype=dtype, device='cpu')

    # Precompute random phases and gamma powers for each layer
    layers = []
    for D, G, M in zip(D_list, G_list, M_list):
        phi_mn_list = [2 * torch.pi * torch.rand(n_max + 1, device=device, dtype=dtype) for _ in range(M)]
        layers.append({'D': D, 'G': G, 'M': M, 'phi_mn_list': phi_mn_list})

    # Process chunks along y-axis
    for start in range(0, res_y, chunk_size):
        end = min(start + chunk_size, res_y)
        y_chunk = y_vals[start:end]
        yy, xx = torch.meshgrid(y_chunk, x_vals, indexing='ij')
        xx = xx.to(device)
        yy = yy.to(device)

        z_chunk = torch.ones_like(xx, device=device, dtype=dtype)

        for layer in layers:
            D = layer['D']
            G = layer['G']
            M = layer['M']
            phi_mn_list = layer['phi_mn_list']
            A = L * (G / L) ** (D - 2) * (torch.log(torch.tensor(gamma, device=device, dtype=dtype)) / M).sqrt()

            r = torch.sqrt(xx**2 + yy**2)
            z_layer = torch.zeros_like(xx, device=device, dtype=dtype)

            for m in range(M):
                theta_m = torch.atan2(yy, xx) - (torch.pi * (m+1) / M)
                cos_theta = torch.cos(theta_m)
                phi_mn = phi_mn_list[m]

                for n in range(n_max + 1):
                    gamma_n = gamma ** n
                    term = torch.cos(phi_mn[n]) - torch.cos(2 * torch.pi * gamma_n * r / L * cos_theta + phi_mn[n])
                    z_layer += gamma ** ((D - 3) * n) * term

            z_layer *= A
            z_chunk *= z_layer  # multiply layers

        # Move chunk to CPU
        z_full[start:end, :] = z_chunk.cpu()

        # Free GPU memory
        del xx, yy, z_chunk, z_layer
        torch.cuda.empty_cache()

    # Normalize globally
    z_full = (z_full - z_full.min()) / (z_full.max() - z_full.min())
    return z_full


device = torch.device("cuda")

# Load model and checkpoints
model = CVAE().to(device)
checkpoint = torch.load("checkpoints/vae_cnn.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load one input image from inputs/
img_path = "example6.png"
img = Image.open(os.path.join("inputs", img_path)).convert("L").resize((hpcfg.img_size, hpcfg.img_size))
x = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
x = x.to(device)

print(terrain_counts_tensor(x.squeeze(), 10, device=device))

injection = torch.tensor([0, 50, 100, 4000, 0, 0, 0, 0, 0, 0]) / x.numel()

# Encode and reconstruct
with torch.no_grad():
    mu, logvar, cond = model.encode(x)
    z = model.reparameterize(mu, logvar)
    recon = model.decode(z, cond).cpu().squeeze()

size = 500
res = 6000  # large grid, 12 points per meter
x_vals = torch.linspace(0, size, res)
y_vals = torch.linspace(0, size, res)

# Layer parameters
D_list = [2.2, 2.45, 2.45]
G_list = [1e-6, 8e-8, 1e-8]
M_list = [16, 32, 64]
L = 100.0
gamma = 1.5
n_max = 10

z = weierstrass_mandelbrot_chunked(
        x_vals, y_vals, D_list, G_list, L, gamma, M_list, n_max,
        device=device, chunk_size=1500
)

# Linearly interpolate recon with input image
BLUR_SIZE = 25
sigma = 1.5

x_blur = gaussian_blur(x.cpu(), BLUR_SIZE, sigma)
recon_t = recon.unsqueeze(0).unsqueeze(0).cpu()

# Interpolate between blurred input and reconstruction
recon_mix = torch.lerp(x_blur, recon_t, 0.95)

# Upscale by 60x with smooth interpolation (blurry) (100x100 -> 6000x6000)
recon_mix = F.interpolate(recon_mix, scale_factor=60, mode='bilinear', align_corners=False).squeeze() + z*LAYER_2_SCALE  # add layer 2

recon_mix = recon_mix.cpu().numpy()

# Scale 3D heights
recon_scaled = recon_mix * 0.2

# 2D + 4x 3D views
fig = plt.figure(figsize=(20, 5))

# 2D original
ax1 = fig.add_subplot(1, 6, 1)
ax1.imshow(x.cpu().squeeze().numpy(), cmap='gray')
ax1.set_title("Original Input")
ax1.axis('off')

# 2D reconstructed
ax2 = fig.add_subplot(1, 6, 2)
ax2.imshow(recon_mix, cmap='gray')
ax2.set_title("Decoded Terrain (with Interpolation)")
ax2.axis('off')

# 3D views at different angles
angles = [(30, 45), (60, 30), (15, 90), (75, 60)]
for i, (elev, azim) in enumerate(angles):
    ax = fig.add_subplot(1, 6, i + 3, projection='3d')
    h, w = recon_scaled.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    ax.plot_surface(X, Y, recon_scaled, cmap='terrain', linewidth=0, antialiased=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"3D View {i+1}")
    ax.set_zlim(0, 1)
    ax.set_axis_off()

plt.tight_layout()

# Save the combined figure
os.makedirs("outputs", exist_ok=True)

# Extract identifier from filename
import os, re
base = os.path.splitext(os.path.basename(img_path))[0]
digits = re.findall(r"\d+", base)
identifier = digits[-1] if digits else base

# Save figure
out_path = os.path.join("outputs", f"{identifier}_rough_full.png")
plt.savefig(out_path, bbox_inches="tight", dpi=150)
plt.close(fig)

print(f"Saved visualization to {out_path}")
