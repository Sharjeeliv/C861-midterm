from pathlib import Path
from collections import Counter

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset



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


def train_val_split_per_class(subset, val_fraction=0.1, seed=None):
    """
    Split a subset (already possibly class-subsampled) into train and validation
    sets, ensuring each class is represented.
    
    Args:
        subset: torch.utils.data.Subset or ImageFolder
        val_fraction: fraction of samples per class to use for validation
        seed: optional random seed for reproducibility
    Returns:
        train_subset, val_subset
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get the targets for this subset
    if isinstance(subset, Subset):
        targets = np.array([subset.dataset.targets[i] for i in subset.indices])
        indices = np.array(subset.indices)
    else:
        targets = np.array(subset.targets)
        indices = np.arange(len(subset))
    
    train_indices = []
    val_indices = []

    for cls in np.unique(targets):
        cls_idx = indices[targets == cls]
        np.random.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_fraction))  # at least 1 sample
        val_indices.extend(cls_idx[:n_val])
        train_indices.extend(cls_idx[n_val:])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return Subset(subset.dataset, train_indices), Subset(subset.dataset, val_indices)


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

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))

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

    print_class_counts(DATA_DIR / 'urdu' / 'train', transform=transform)
    end = time.time()
    print(f"Data loading took {end - start:.2f} seconds")
