import torch
from time import time
import os

EPOCHS = 15
WEIGHTS_DIR = "weights"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def finetune_setup(model):
    # Freeze all layers except the last fully connected layer
    for param in model.parameters(): param.requires_grad = False
    # Last layers are named fcf (for custom models):
    if hasattr(model, 'fcf'):
        for param in model.fcf.parameters():
            param.requires_grad = True
    else:
        # If the model has a different architecture, adjust accordingly
        raise ValueError("Model does not have a 'fcf' layer. Adjust finetune_setup accordingly.")


def train(model, optimizer, criterion, train_loader, name, dataset_name):
    cul_time = 0
    for epoch in range(EPOCHS):
        starttime = time()
        model.to(device)
        model.train()
        running_loss = 0.0
        correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        total_time = time() - starttime
        cul_time += total_time
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, Time: {total_time:.2f}s")
    print(f"Total Training Time: {cul_time:.2f}s")
    
    # Save the model
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{WEIGHTS_DIR}/{name}_{dataset_name}_model.pth")


def evaluate(model, test_loader):
    # Evaluate on test set
    model.to(device)
    model.eval()  # set model to evaluation mode
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

