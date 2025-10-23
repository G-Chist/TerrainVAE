import rasterio
import numpy as np

path = "/home/matvei/PycharmProjects/TerrainVAE/dot-tif/output_USGS1m.tif"

with rasterio.open(path) as src:
    elevation = src.read(1).astype(float)
    elevation[elevation == src.nodata] = np.nan  # mask nodata

import matplotlib.pyplot as plt

# define crop size
h, w = elevation.shape
crop_h, crop_w = 256, 256
start_h = (h - crop_h) // 2
start_w = (w - crop_w) // 2

crop = elevation[start_h:start_h+crop_h, start_w:start_w+crop_w]

print("Shape:", crop.shape)
print("Min crop:", np.nanmin(crop))
print("Max crop:", np.nanmax(crop))
print("Max - min:", np.nanmax(crop) - np.nanmin(crop))
print("Mean crop:", np.nanmean(crop))
print("Std dev:", np.nanstd(crop))
print("NaN ratio:", np.isnan(crop).mean())

plt.figure(figsize=(8, 8))
plt.imshow(crop, cmap="terrain")
plt.title("1024x1024 Elevation Crop")
plt.colorbar(label="Elevation (m)")
plt.tight_layout()
plt.show()