from pathlib import Path
from collections import Counter

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset


# VARIABLES
# NUM_WORKERS Causes issues on Windows; set to 0
BATCH_SIZE = 256
NUM_WORKERS = 0 
PIN_MEMORY = True 

# SETUP PATHS
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'

# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# ************************
# HELPERS FUNCTION
# ************************
def print_class_counts(dataset_path, transform=None):
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    counts = Counter(dataset.targets)
    for class_name, class_index in dataset.class_to_idx.items():
        print(f"{class_name:<20}{counts[class_index]:>15}")
    print(f"Total images: {len(dataset)}\n")
    
    avg_samples = np.mean(list(counts.values()))
    min_samples = np.min(list(counts.values()))
    max_samples = np.max(list(counts.values()))
    print(f"Min: {min_samples}")
    print(f"Max: {max_samples}")
    print(f"Avg: {avg_samples:.2f}")


def subsample_per_class(data: datasets.ImageFolder, n: int = 0, fraction: float = 1):
    """
    Subsample an ImageFolder dataset to `n` samples per class, or a fraction per class.
    
    Args:
        data: torchvision.datasets.ImageFolder
        n: number of samples per class
        fraction: fraction of samples per class (ignored if n > 0)
    
    Returns:
        torch.utils.data.Subset with the selected samples
    """
    # Get all class labels
    targets = np.array(data.targets)
    indices = []
    # Pick up to n samples per class
    if n > 0:
        for cls in np.unique(targets):
            cls_idx = np.where(targets == cls)[0]
            np.random.shuffle(cls_idx)
            selected = cls_idx[:n] if n <= len(cls_idx) else cls_idx
            indices.extend(selected)
    # Pick fraction of samples per class
    elif 0 < fraction < 1:
        for cls in np.unique(targets):
            cls_idx = np.where(targets == cls)[0]
            np.random.shuffle(cls_idx)
            subset_size = max(1, int(len(cls_idx) * fraction))
            indices.extend(cls_idx[:subset_size])
    # No subsampling
    else:   return data

    np.random.shuffle(indices)
    return Subset(data, indices)

def unwrap_subset(subset):
    """Flatten nested Subsets to original dataset and absolute indices."""
    indices = np.arange(len(subset))
    base_dataset = subset
    while isinstance(base_dataset, Subset):
        indices = np.array(base_dataset.indices)[indices]
        base_dataset = base_dataset.dataset
    return base_dataset, indices

def train_val_split_per_class(subset, val_fraction=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    base_dataset, indices = unwrap_subset(subset)
    targets = np.array(base_dataset.targets)[indices]

    train_indices, val_indices = [], []

    for cls in np.unique(targets):
        cls_mask = targets == cls
        cls_idx = indices[cls_mask]
        np.random.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_fraction))
        val_indices.extend(cls_idx[:n_val])
        train_indices.extend(cls_idx[n_val:])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return Subset(base_dataset, train_indices), Subset(base_dataset, val_indices)


def make_dataloader(input_data):
    dataloader = DataLoader(
        input_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return dataloader

# ************************
# INTERFACE FUNCTION
# ************************
def combine_loaders(loader1: DataLoader, loader2: DataLoader):
    ds1, ds2 = loader1.dataset, loader2.dataset
    ds3 = ConcatDataset([ds1, ds2])
    return make_dataloader(ds3)

def load_dataset(lang: str, class_n: int = 0, class_frac: float = 1, val_frac: float = 0.1, seed: int = 42):
    """
    Load dataset for a given language with optional subsampling.
    
    Args:
        lang: Language code ('urdu', etc.)
        n_per_class: Number of samples per class
        fraction_per_class: Fraction of samples per class
    Returns:
        train_loader, val_loader, test_loader
    """
    # SETUP PATHS
    test_path = DATA_DIR / lang / 'test'
    train_path = DATA_DIR  / lang / 'train'

    # Check if paths exist
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Data paths for language '{lang}' not found.")

    # LOAD DATA
    full_train_data = datasets.ImageFolder(root=train_path, transform=transform)
    test_data = datasets.ImageFolder(root=test_path, transform=transform)

    # SUBSAMPLE IF NEEDED
    if class_n > 0 or (0 < class_frac < 1):
        full_train_data = subsample_per_class(full_train_data, n=class_n, fraction=class_frac)

    # SPLIT TRAIN → TRAIN + VALIDATION
    train_data, val_data = train_val_split_per_class(full_train_data, val_fraction=val_frac, seed=seed)

    # print(len(train_data))
    # print(len(val_data))
    # print(len(test_data))

    # DATA LOADERS
    train_loader = make_dataloader(train_data)
    val_loader = make_dataloader(val_data)
    test_loader = make_dataloader(test_data)
    return train_loader, val_loader, test_loader

# ************************
# MAIN FUNCTION
# ************************
if __name__ == "__main__":
    import time

    start = time.time()
    # train_loader, val_loader, test_loader = load_dataset('english', class_n=100)

    # print(f"Training batches: {len(train_loader)}")
    # print(f"Validation batches: {len(val_loader)}")
    # print(f"Testing batches: {len(test_loader)}")

    print_class_counts(DATA_DIR / 'ur' / 'train', transform=transform)
    end = time.time()
    print(f"Data loading took {end - start:.2f} seconds")
