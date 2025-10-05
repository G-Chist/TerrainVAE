import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from torchvision import transforms
import torch.utils


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
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path = "C:\\Users\\79140\\PycharmProjects\\procedural-terrain-generation\\data\\datapoints_png_cropped"
    img_size = 512
    batch_size = 1

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)) if isinstance(img_size, int)
        else transforms.Resize(img_size),
        transforms.ToTensor()])

    dataset = TerrainDataset(root_dir=path,
                             transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)

    # Get one batch
    imgs = next(iter(train_loader))  # shape: [b, 1, img_size, img_size]

    # Convert to grid display
    b = imgs.size(0)
    fig, axes = plt.subplots(1, b, figsize=(b * 2, 2))
    if b == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        img = imgs[i, 0].cpu()  # remove channel dimension
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


