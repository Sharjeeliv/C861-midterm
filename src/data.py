from typing import Tuple
from pathlib import Path
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader




# Load Arabic Digit Dataset for CNN

# Notes:
#  Dataloader is used for shuffling and batching the data
# Transforms are used for data augmentation and normalization


# VARIABLES
BATCH_SIZE = 64


# SETUP PATHS
ROOT = Path(__file__).parent.parent
TEST_PATH = ROOT / 'data' / f'urdu_test'
TRAIN_PATH = ROOT / 'data' / f'urdu_train'

# def filename_label_to_subdir(path: Path, exts: Tuple=('png')):
#     for p in path.iterdir():
#         if not p.is_file(): continue
#         if not p.name.lower().endswith(exts): continue

#         # Extract label from filename
#         label = p.name.split('_')[-1].split('.')[0]
#         subdir = path / label
#         subdir.mkdir(exist_ok=True)
#         p.rename(subdir / p.name) 

# filename_label_to_subdir(TRAIN_PATH)
# filename_label_to_subdir(TEST_PATH)


# # Define transforms - Might need another for other dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# train_transform = transforms.Compose([
#     transforms.RandomRotation(10),
#     transforms.RandomAffine(0, translate=(0.1, 0.1)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# train_data = datasets.ImageFolder(root=TRAIN_PATH, transform=train_transform)
# test_data = datasets.ImageFolder(root=TEST_PATH, transform=test_transform)

# print(os.listdir(TEST_PATH / '1')[:10])

# Load datasets
train_data = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
test_data = datasets.ImageFolder(root=TEST_PATH, transform=transform)


# Data loaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False) 

print(f"Classes: {train_data.classes}")
print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")

# # Visualize one batch
# import matplotlib.pyplot as plt
# images, labels = next(iter(train_loader))
# plt.imshow(utils.make_grid(images[:8], nrow=8).permute(1, 2, 0))
# plt.show()
