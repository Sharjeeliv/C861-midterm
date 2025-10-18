import torch

# Relative Imports
from .models.CNN import BasicCNN
from .train import train, evaluate
from .data import train_loader, test_loader



def run_all():
    pass



def main():
    # Initialize model, loss function, and optimizer
    # Arabic=28, Urdu=36-40
    model = BasicCNN(num_classes=40)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, optimizer, criterion, train_loader)
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()