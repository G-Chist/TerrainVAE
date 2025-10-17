"""
Weierstrass-Mandelbrot Fractal Terrain Generator (PyTorch, Seamless Chunked)
----------------------------------------------------------------------------

Generates a 2D fractal terrain heightmap using PyTorch with seamless chunking
for large grids. Visualizes the terrain using Matplotlib.

Dependencies:
-------------
torch, matplotlib
"""

import torch
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(123)

    size = 500
    res = 6000  # large grid
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

    # Plot terrain
    plt.figure(figsize=(8, 8))
    plt.imshow(z.numpy(), cmap='terrain', interpolation='lanczos')
    plt.colorbar(label='Height')
    plt.title('Seamless Chunked Weierstrass-Mandelbrot Fractal Terrain')
    plt.show()
