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

device = torch.device("cpu")

# Load model and weights
model = CVAE().to(device)
checkpoint = torch.load("checkpoints/vae_cnn.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load one input image from inputs/
img_path = "example1.png"
img = Image.open(os.path.join("inputs", img_path)).convert("L").resize((hpcfg.img_size, hpcfg.img_size))
x = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
x = x.to(device)

# Prepare for "feature injection": use real-life landform feature ratio

# Original counts
counts = {0: 0, 1: 2, 2: 63, 3: 861, 4: 7739, 5: 0, 6: 0, 7: 70, 8: 0, 9: 0}

# Convert to tensor
keys_sorted = sorted(counts.keys())
vec = torch.tensor([counts[k] for k in keys_sorted], dtype=torch.float32)

# Normalize (divide by sum to get values between 0 and 1)
cond_vector = vec / vec.sum()

# Unsqueeze
cond_vector = cond_vector.unsqueeze(0)

# Encode and reconstruct
with torch.no_grad():
    mu, logvar, _ = model.encode(x)  # ignore real landform features
    cond = cond_vector
    z = model.reparameterize(mu, logvar)
    recon = model.decode(z, cond).cpu().squeeze().numpy()

# Scale 3D heights
recon_scaled = recon * 0.2

# 2D + 4x 3D views
fig = plt.figure(figsize=(20, 5))

# 2D original
ax1 = fig.add_subplot(1, 6, 1)
ax1.imshow(x.cpu().squeeze().numpy(), cmap='gray')
ax1.set_title("Original Input")
ax1.axis('off')

# 2D reconstructed
ax2 = fig.add_subplot(1, 6, 2)
ax2.imshow(recon, cmap='gray')
ax2.set_title("Decoded Terrain")
ax2.axis('off')

# 3D views at different angles
angles = [(30, 45), (60, 30), (45, 90), (75, 60)]
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
plt.show()
