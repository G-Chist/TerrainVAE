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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Encode and reconstruct
with torch.no_grad():
    mu, logvar, cond = model.encode(x)
    z = model.reparameterize(mu, logvar)
    recon = model.decode(z, cond).cpu().squeeze().numpy()

# Prepare 2D + 3D plot in one window
fig = plt.figure(figsize=(10, 5))

# 2D original and reconstructed
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(x.cpu().squeeze().numpy(), cmap='gray')
ax1.set_title("Original Input")
ax1.axis('off')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
h, w = recon.shape
X, Y = np.meshgrid(np.arange(w), np.arange(h))
ax2.plot_surface(X, Y, recon, cmap='terrain', linewidth=0, antialiased=False)
ax2.set_title("Decoded Terrain (3D)")

plt.tight_layout()
plt.show()
