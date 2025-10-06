import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import rgb_to_grayscale, rotate, resize


class TerrainDataset(Dataset):
    """Dataset for grayscale heightmaps with 0, 90, 180, 270 degree rotations (PyTorch-native)."""

    def __init__(self, root_dir, img_size=512):
        self.root_dir = root_dir
        self.img_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.png')]
        if len(self.img_files) == 0:
            raise ValueError(f"No PNG files found in {root_dir}")
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files) * 4

    def __getitem__(self, idx):
        img_idx = idx // 4
        rot_idx = idx % 4
        img_path = os.path.join(self.root_dir, self.img_files[img_idx])

        # Read and preprocess
        img = read_image(img_path)  # [C,H,W], uint8 0-255
        img = rgb_to_grayscale(img)  # [1,H,W]
        img = resize(img, [self.img_size, self.img_size])  # [1,img_size,img_size]

        # Apply rotation
        if rot_idx == 1:
            img = rotate(img, 90)
        elif rot_idx == 2:
            img = rotate(img, 180)
        elif rot_idx == 3:
            img = rotate(img, 270)

        # Convert to float before min/max operations
        img = img.float()

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # normalize 0-1 safely
        img = torch.clamp(img, 0.0, 1.0)  # ensure no out-of-bounds values

        return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    path = r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\datapoints_png_cropped"
    img_size = 28
    batch_size = 8

    dataset = TerrainDataset(root_dir=path, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(len(loader))

    imgs = next(iter(loader))  # shape: [b, 1, H, W]

    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 2, 2))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i, 0].cpu(), cmap="gray")
        print(imgs[i, 0].cpu())
        ax.axis("off")
    plt.show()

