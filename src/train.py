import torch

EPOCHS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train(model, optimizer, criterion, train_loader):
    for epoch in range(EPOCHS):
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
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")


def evaluate(model, test_loader):
    # Evaluate on test set
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

