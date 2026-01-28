import os
import matplotlib.pyplot as plt

import random
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from torchvision import transforms
from timm.models import create_model
from sklearn.metrics import roc_curve, auc

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


def set_seeds(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


import json
from collections import Counter


def data_summary(dataset, test_size=0.2, batch_size=2, seed=3000):
    set_seeds(seed)

    x_indices, y_labels = dataset.idx_representation()

    train_indices, test_indices, _, _ = train_test_split(
        x_indices,
        y_labels,
        stratify=y_labels,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    # Create datasets and dataloaders
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # DATA SUMMARY
    print("X dtype:", next(iter(train_loader))[0].dtype)
    print("Y dtype:", next(iter(train_loader))[1].dtype)

    print("Test size:", len(test_loader.dataset))
    print("Train size:", len(train_loader.dataset))

    print("Single batch shape:", next(iter(train_loader))[0].shape)

    print()
    print("Complete class distribution:")
    print(json.dumps(dataset.label_distribution, indent=2))

    print("\nTrainset class distribution:")

    try:
        c = Counter(map(lambda d: d[1].item(), train_dataset))
    except RuntimeError:
        c = Counter(map(lambda d: str(tuple(d[1].tolist())), train_dataset))
    distribution = {k: v / len(train_dataset) for k, v in c.items()}
    print(json.dumps(dict(c.items()), indent=2))
    print(json.dumps(distribution, indent=2))


def make_dataset(
    path_to_index,
    image_size=(600, 600),
    seed=88,
):
    set_seeds(seed)

    # PyTorch Dataset Definition (hardcoded for all)
    class CustomDataset(Dataset):
        def __init__(self, pairs, transform=None):
            """
            Args:
                xml_root: Parsed XML root element.
                image_dir: Directory containing images.
                transform: Transformations for images and masks.
            """
            self.pairs = pairs
            self.transform = transform

            self.images = [pair["path"] for pair in pairs]
            self.labels = [pair["label"] for pair in pairs]

        def __len__(self):
            return len(self.images)

        def idx_representation(self):
            return range(len(self.images)), self.labels

        def __getitem__(self, idx):
            image_path = self.images[idx]
            image = Image.open(image_path).convert("RGB")
            quality = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, quality

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),  # Resize to uniform size
            transforms.ToTensor(),
        ]
    )

    with open(path_to_index) as file:
        dataset_index = json.load(file)

    dataset = CustomDataset(dataset_index, transform=transform)

    return dataset
