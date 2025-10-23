import rasterio
import numpy as np
import matplotlib.pyplot as plt

path = "/home/matvei/PycharmProjects/TerrainVAE/dot-tif/output_USGS1m.tif"

with rasterio.open(path) as src:
    elevation = src.read(1).astype(float)
    elevation[elevation == src.nodata] = np.nan

h, w = elevation.shape
win_h, win_w = 256, 256
stride = 256

rows = (h - win_h) // stride + 1
cols = (w - win_w) // stride + 1

def nan_stat(func):
    out = np.full((rows, cols), np.nan)
    for i in range(rows):
        for j in range(cols):
            patch = elevation[i*stride:i*stride+win_h, j*stride:j*stride+win_w]
            out[i, j] = func(patch)
    return out

max_map = nan_stat(np.nanmax)
min_map = nan_stat(np.nanmin)
range_map = max_map - min_map

# plot 2 maps + 2 line charts
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# top row: maps
axes[0, 0].imshow(max_map, cmap="terrain")
axes[0, 0].set_title("Max Height per Window")
fig.colorbar(axes[0, 0].images[0], ax=axes[0, 0])

axes[0, 1].imshow(range_map, cmap="plasma")
axes[0, 1].set_title("Height Range (Max–Min)")
fig.colorbar(axes[0, 1].images[0], ax=axes[0, 1])

# bottom row: linear plots
axes[1, 0].plot(np.nanmean(range_map, axis=0))
axes[1, 0].set_title("Height Range per Column")
axes[1, 0].set_xlabel("Column index")
axes[1, 0].set_ylabel("Elevation Δ (m)")

axes[1, 1].plot(np.nanmean(min_map, axis=0), label="Min")
axes[1, 1].plot(np.nanmean(max_map, axis=0), label="Max")
axes[1, 1].set_title("Min and Max Heights per Column")
axes[1, 1].set_xlabel("Column index")
axes[1, 1].set_ylabel("Elevation (m)")
axes[1, 1].legend()

plt.tight_layout()
plt.show()
