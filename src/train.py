import torch
from time import time
import os


from optuna import Trial
import optuna

from .utils.optuna import get_trial_params


EPOCHS = 15
WEIGHTS_DIR = "weights"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = None


# TODO:
# 1. Complete the objective function for Optuna hyperparameter tuning
# 2. Update custom models with dict. based init
# 3. Complete training and validation
# 4. Implement early stopping based on validation performance
# 5. Custom CNN via tuning
# 6. Saving and loading models
# 7. Evaluation function


# train_loader, val_loader, test_loader = load_dataset(LANG, class_n=100)
# model = BasicCNN(num_classes=DATASET_NCLS[LANG])
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train(model, optimizer, criterion, train_loader, "BasicCNN", LANG)


def objective(trial: Trial, model_name: str, tr_loader, val_loader):
    model_cls = models[model_name]
    trial_params = get_trial_params(trial, model_name)

    # Instantiate model with trial params
    model = model_cls(**trial_params)
    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Epoch Loop:
    cul_time = 0
    for epoch in range(EPOCHS):
        starttime = time()
        model.to(device)
        model.train()
        running_loss = 0.0
        correct = total = 0
        # Train
        train(model, optimizer, criterion, tr_loader, model_name, "Trial")

        # Validation
        validate(model, val_loader, criterion)
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        total_time = time() - starttime
        cul_time += total_time
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, Time: {total_time:.2f}s")
    print(f"Total Training Time: {cul_time:.2f}s")


    # Train the model
    epoch, val_loss = 0, 0

    # Early stopping setup via Optuna pruning
    trial.report(val_loss, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()


    # Validate the model and return validation accuracy




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


def validate(model, val_loader, criterion):
    model.to(device)
    model.eval()
    val_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_loss /= total
    val_acc = 100.0 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    return val_loss, val_acc


def train(model, optimizer, criterion, train_loader, name, dataset_name):
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

        
    
    # # Save the model
    # os.makedirs(WEIGHTS_DIR, exist_ok=True)
    # torch.save(model.state_dict(), f"{WEIGHTS_DIR}/{name}_{dataset_name}_model.pth")


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

