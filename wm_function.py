"""
Weierstrass-Mandelbrot Function Terrain Generator (PyTorch Version)
-------------------------------------------------------------------

This script generates a 2D Weierstrass-Mandelbrot fractal terrain heightmap
using PyTorch tensors and visualizes it with Matplotlib.

Key Components:
---------------
- Fractal Noise Generation (Weierstrass-Mandelbrot)
- Tensor operations using PyTorch
- Visualization using Matplotlib

Dependencies:
-------------
torch, numpy, matplotlib

Usage:
------
Run the script directly in any Python environment:
    python wm_function.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def weierstrass_mandelbrot_3d(x, y, D, G, L, gamma, M, n_max, device='cpu'):
    """
    Compute the 3D Weierstrass-Mandelbrot function z(x, y) using PyTorch.

    Parameters:
        x, y : 2D torch tensors
        D : float
            Fractal dimension (typically 2 < D < 3)
        G : float
            Amplitude roughness coefficient
        L : float
            Transverse width of the profile
        gamma : float
            Frequency scaling factor (>1)
        M : int
            Number of ridges (azimuthal angles)
        n_max : int
            Upper cutoff frequency index
    """
    A = L * (G / L) ** (D - 2) * (torch.log(torch.tensor(gamma)) / M) ** 0.5
    z = torch.zeros_like(x, device=device)

    r = torch.sqrt(x ** 2 + y ** 2)
    for m in range(1, M + 1):
        theta_m = torch.atan2(y, x) - np.pi * m / M
        phi_mn = 2 * np.pi * torch.rand(n_max + 1, device=device)

        for n in range(n_max + 1):
            gamma_n = gamma ** n
            term = (
                torch.cos(phi_mn[n])
                - torch.cos(2 * np.pi * gamma_n * r / L * torch.cos(theta_m) + phi_mn[n])
            )
            z += gamma ** ((D - 3) * n) * term

    z *= A
    return z


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    size = 512
    res = 2000
    torch.manual_seed(123)

    x_vals = torch.linspace(0, size, res, device=device)
    y_vals = torch.linspace(0, size, res, device=device)
    x, y = torch.meshgrid(x_vals, y_vals, indexing='xy')

    L = 100.0
    gamma = 1.5
    n_max = 10

    z1 = weierstrass_mandelbrot_3d(x, y, D=2.2,  G=1e-6, L=L, gamma=gamma, M=16, n_max=n_max, device=device)
    z2 = weierstrass_mandelbrot_3d(x, y, D=2.45, G=8e-8, L=L, gamma=gamma, M=32, n_max=n_max, device=device)
    z3 = weierstrass_mandelbrot_3d(x, y, D=2.45, G=1e-8, L=L, gamma=gamma, M=64, n_max=n_max, device=device)

    z = z1 * z2 * z3

    # Normalize to [0,1]
    z = (z - z.min()) / (z.max() - z.min())

    # Move to CPU for plotting
    z_cpu = z.cpu().numpy()

    # Plot heightmap
    plt.figure(figsize=(8, 8))
    plt.imshow(z_cpu, cmap="terrain", interpolation="lanczos")
    plt.colorbar(label="Height")
    plt.title("Weierstrass-Mandelbrot Fractal Terrain")
    plt.show()
