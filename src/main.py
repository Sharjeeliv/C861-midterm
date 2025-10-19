import torch

# Relative Imports
from .models.CNN import BasicCNN
from .train import train, evaluate
from .data import load_dataset
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASET_NCLS = {
    "arabic":   28,
    "urdu":     40,
    "english":  26
}

def finetune():
    weights = "weights\BasicCNN_Arabic_model.pth"
    model = BasicCNN(num_classes=28)
    model.load_state_dict(torch.load(weights))

    # Change final layer for Urdu dataset (36-40 classes)
    num_ftrs = model.fc2.in_features
    model.fc2 = torch.nn.Linear(num_ftrs, 40)  # Adjust output classes for Urdu

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last conv layer and fc layer
    for param in model.conv3.parameters():
        param.requires_grad = True
    for param in model.fc2.parameters():
        param.requires_grad = True
    
    # Continue with training and evaluation
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    train(model, optimizer, criterion, train_loader, "BasicCNN_Finetuned", "Urdu")
    evaluate(model, test_loader)


def main():
    LANG = 'arabic'  # Change as needed

    train_loader, val_loader, test_loader = load_dataset(LANG, class_n=100)
    model = BasicCNN(num_classes=DATASET_NCLS[LANG])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train(model, optimizer, criterion, train_loader, "BasicCNN", LANG)
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()