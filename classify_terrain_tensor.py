import torch
import torch.nn.functional as F


def terrain_counts_tensor(H, num_classes=10, device="cpu"):
    feat = classify_terrain_tensor(H)  # shape (h, w)
    counts = count_features_by_class(feat, num_classes=num_classes)
    counts_tensor = torch.tensor([counts[i] for i in range(num_classes)],
                                 dtype=torch.float32, device=device)
    counts_tensor /= H.numel()  # normalize to [0,1]
    return counts_tensor


def classify_terrain_tensor(H, eps=1e-5):
    """
    Classify each pixel of a 2D terrain tensor into one of ten terrain feature types
    (flat, peak, ridge, shoulder, spur, slope, pit, valley, footslope, hollow).

    Parameters
    ----------
    H : torch.Tensor
        2D tensor of elevations, shape (h, w)
    eps : float
        Threshold for equality (treat values within +-eps as equal)

    Returns
    -------
    torch.Tensor
        Tensor of ints (h, w) with class codes 0–9
    """

    # prepare tensor
    device = H.device
    H = H.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
    Hpad = F.pad(H, (1, 1, 1, 1), mode="replicate")

    # extract 3x3 neighbourhoods
    neighs = torch.stack([
        Hpad[:, :, i:i+H.shape[2], j:j+H.shape[3]]
        for i in range(3) for j in range(3)
        if not (i == 1 and j == 1)
    ], dim=0)  # (8,1,1,h,w)

    center = H
    diffs = neighs - center  # (8,1,1,h,w)

    # Boolean masks
    eq = (diffs.abs() < eps)
    gt = (diffs > eps)
    lt = (diffs < -eps)

    # Counts
    n_eq = eq.sum(0)
    n_gt = gt.sum(0)
    n_lt = lt.sum(0)

    # Classify per pixel
    feat = torch.full(center.shape, 10, dtype=torch.int64, device=device)  # default class 10

    feat[n_eq == 8] = 0  # flat
    feat[n_lt == 8] = 1  # peak
    feat[n_gt == 8] = 6  # pit
    feat[n_lt == 6] = 2  # ridge
    feat[(n_lt == 3) & (n_gt == 5)] = 3  # shoulder
    feat[(n_lt == 4) & (n_gt == 4)] = 4  # spur
    feat[n_eq >= 6] = 5  # slope
    feat[n_gt == 6] = 7  # valley
    feat[(n_gt >= 4) & ((diffs.abs() > 0.1).sum(0) >= 4)] = 9  # hollow
    feat[(n_eq == 5) & (n_gt == 3)] = 8  # footslope

    return feat[0,0]


def count_features_by_class(feat_tensor, num_classes=11):
    """
    Count the number of pixels belonging to each terrain feature class.

    Parameters
    ----------
    feat_tensor : torch.Tensor
        2D tensor of integers 0-10 representing terrain features
    num_classes : int
        Total number of classes (default 11)

    Returns
    -------
    dict
        Mapping from class index to count
    """
    counts = torch.bincount(feat_tensor.flatten(), minlength=num_classes)
    return {i: int(counts[i]) for i in range(num_classes)}


if __name__ == "__main__":
    # Example usage
    data_path = "C:\\Users\\79140\\PycharmProjects\\procedural-terrain-generation\\data\\datapoints_png_cropped"
    use_accel = True
    img_size = 100
    batch_size = 32
    
    from terrain_dataset import TerrainDataset

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_accel else {}

    dataset = TerrainDataset(root_dir=data_path,
                             img_size=img_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)

    data = [i for i in train_loader][5]
    data = data.squeeze()[1].to(device)
    print(data.size())
    features = classify_terrain_tensor(data)
    print(features.flatten())

    import matplotlib.pyplot as plt


    def show_side_by_side(data, features):
        data_cpu = data.detach().cpu().numpy()
        feats_cpu = features.detach().cpu().numpy()

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.title("Elevation")
        plt.imshow(data_cpu, cmap="terrain")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1, 3, 2)
        plt.title("Feature Class")
        plt.imshow(feats_cpu, cmap="tab10", vmin=0, vmax=9)
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1, 3, 3)
        plt.title("Feature Class Histogram")
        feature_counts = count_features_by_class(features)

        plt.bar(feature_counts.keys(), feature_counts.values())
        plt.xlabel("Terrain Feature Class")
        plt.ylabel("Pixel Count")
        plt.title("Terrain Feature Counts")
        plt.xticks(range(11),
                   ["flat", "peak", "ridge", "shoulder", "spur", "slope", "pit", "valley", "footslope", "hollow", "N/A"],
                   rotation=45)

        plt.tight_layout()
        plt.show()


    show_side_by_side(data, features)
