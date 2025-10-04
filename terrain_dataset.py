import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms


class TerrainDataset(Dataset):
    """Dataset for grayscale heightmaps with 0, 90, 180, 270 degree rotations."""

    def __init__(self, root_dir, transform=transforms.Compose([
            transforms.Resize((512, 512)) if isinstance(512, int) else transforms.Resize(512),
            transforms.ToTensor(),
        ])):

        self.root_dir = root_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.png')]

    def __len__(self):
        # Each image contributes 4 rotated versions
        return len(self.img_files) * 4

    def __getitem__(self, idx):
        # Map global idx to image and rotation
        img_idx = idx // 4
        rot_idx = idx % 4

        img_path = os.path.join(self.root_dir, self.img_files[img_idx])
        image = Image.open(img_path).convert("L")
        image = ImageOps.exif_transpose(image)

        # Apply rotation
        if rot_idx == 1:
            image = image.rotate(90, expand=True)
        elif rot_idx == 2:
            image = image.rotate(180, expand=True)
        elif rot_idx == 3:
            image = image.rotate(270, expand=True)

        if self.transform:
            image = self.transform(image)

        return image