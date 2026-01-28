import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import numpy as np
import json
import os, math

file_paths = np.load(r"/extra/xielab0/wuat2/AryaQualityViewProjectData/ExternalValData/file_paths.npy")
labels = np.load(r"/extra/xielab0/wuat2/AryaQualityViewProjectData/ExternalValData/labels.npy")

class XrayDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        base_path = os.path.splitext(self.file_paths[idx])[0]
        img_path = base_path + ".png"
        
        image = read_image(img_path, mode=ImageReadMode.RGB)
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

def load_dataset(image_size: int, batch_size: int = 32, test_transform = None):
    config_path = r"/extra/xielab0/wuat2/AryaQualityViewProjectData/ExternalValData/ext_val_torch_loader_config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    torch.manual_seed(cfg["seed"])

    if not test_transform: #if test_transform remains none, use default
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.ConvertImageDtype(torch.float32), # float conversion
        ])

    dataset = XrayDataset(file_paths, labels, transform=test_transform)

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=cfg["shuffle"],
        pin_memory=True
    )

    return dataset, loader

class PercentileDomainAdaptation(object):
    def __init__(self, train_p1, train_p99):
        # The target range we want to force our test images into
        self.target_min = train_p1
        self.target_max = train_p99
        self.target_range = train_p99 - train_p1
        self.limit = 10_000_000 #note the underscores are just for readibility; this is still an int

    def __call__(self, img):

        if not img.is_floating_point():
            img = img.type(torch.float32)

        num_pixels = img.numel() #reduce size of img if too big for outlier calculation
        if num_pixels > self.limit:
            factor = int(math.ceil(math.sqrt(num_pixels / self.limit)))
            proxy = img[:, ::factor, ::factor].reshape(-1)
        else:
            proxy = img.reshape(-1)
            
        img_p1 = torch.quantile(proxy, 0.01)
        img_p99 = torch.quantile(proxy, 0.99)
        img_range = img_p99 - img_p1 + 1e-6

        img = torch.clamp(img, img_p1, img_p99)

        img_normalized = (img - img_p1) / img_range

        img_adapted = (img_normalized * self.target_range) + self.target_min

        return torch.clamp(img_adapted, 0.0, 1.0).contiguous() #clamp again just in case; force contiguous memory
