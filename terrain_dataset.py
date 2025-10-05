import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import rgb_to_grayscale, rotate


class TerrainDataset(Dataset):
    """Dataset for grayscale heightmaps with 0, 90, 180, 270 degree rotations (PyTorch-native)."""

    def __init__(self, root_dir, img_size=512):
        self.root_dir = root_dir
        self.img_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.png')]
        self.img_size = img_size

        self.preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_files) * 4

    def __getitem__(self, idx):
        img_idx = idx // 4
        rot_idx = idx % 4
        img_path = os.path.join(self.root_dir, self.img_files[img_idx])

        # Read image (tensor: [C,H,W], dtype uint8)

        img = rgb_to_grayscale(read_image(img_path).float())  # now [1,H,W]
        img /= 255.0  # normalize [0,1]

        # Resize to target size
        img = transforms.functional.resize(img, [self.img_size, self.img_size])

        # Rotate tensor (no PIL)
        if rot_idx == 1:
            img = rotate(img, 90)
        elif rot_idx == 2:
            img = rotate(img, 180)
        elif rot_idx == 3:
            img = rotate(img, 270)

        return img / 255


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

