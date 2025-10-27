# Local
from .utils.optuna import get_trial_params

# First-party
from time import time
import os

# Third-party
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
import optuna
from optuna import Trial
from torch.utils.data import DataLoader

# Relative Imports
from .models.CNN import BasicCNN, LeNet5, CNN
from .models.ResNet import ResNet
from .models.VG import VGG16
from .utils.utils import EarlyStopping
from .data import combine_loaders


# TODO:
# 1. Complete the objective function for Optuna hyperparameter tuning
# 2. Update custom models with dict. based init
# 3. Complete training and validation
# 4. Implement early stopping based on validation performance
# 5. Custom CNN via tuning
# 6. Saving and loading models
# 7. Evaluation function


# ********************************
# CONSTANTS, VARIABLES AND SETUP
# ********************************
EPOCHS = 15
N_TRIALS = 5
WEIGHTS_DIR = "weights"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {
    "BasicCNN": BasicCNN,
    "LeNet5": LeNet5,
    "ResNet": ResNet,
    "VGG16": VGG16,
    "CNN": CNN
}

# ********************************
# HELPER FUNCTIONS
# ********************************
# def _train(model, optimizer, criterion, train_loader):
#     model.train()
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

def _train(model, optimizer, criterion, train_loader, log=False):
    model.train()
    nclasses = model.nclasses
    tr_loss, total = 0.0, 0
    tr_acc = Accuracy(task="multiclass", num_classes=nclasses).to(device)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if not log: continue
        tr_loss += loss.item() * images.size(0)
        total += labels.size(0)
        preds = outputs.argmax(dim=1)
        tr_acc.update(preds, labels)

    if not log: return
    tr_loss /= total
    tr_acc = tr_acc.compute().item() * 100
    print(f"Train:\tLoss: {tr_loss:.4f} \t Accuracy: {tr_acc:.2f}%")
    return tr_loss


def _train_loop(model, optimizer, criterion, train_loader):
    early_stopping = EarlyStopping()
    total_time = 0.0
    for epoch in range(EPOCHS):
        start_time = time()
        # Training & Validation
        print(f"E={epoch + 1}", end="\t")
        tr_loss = _train(model, optimizer, criterion, train_loader, log=True)
        # Timing and reporting
        epoch_time = time() - start_time
        total_time += epoch_time
        # Early Stopping
        early_stopping(tr_loss)
        if early_stopping.early_stop: 
            print(f"Early stop! Total Training Time: {total_time:.2f}s")
            return
        
    print(f"Total Training Time: {total_time:.2f}s")


def _validate(model, val_loader, criterion):
    model.eval()
    nclasses = model.nclasses
    val_loss, total = 0.0, 0
    val_acc = Accuracy(task="multiclass", num_classes=nclasses).to(device)
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            total += labels.size(0)
            preds = outputs.argmax(dim=1)
            val_acc.update(preds, labels)
    val_loss /= total
    val_acc = val_acc.compute().item() * 100
    print(f"Val:\tLoss: {val_loss:.4f} \t Accuracy: {val_acc:.2f}%")
    return val_loss, val_acc


def save_model(model, title):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{WEIGHTS_DIR}/{title}_mdl.pth")


# ********************************
# INTERFACE FUNCTIONS
# ********************************
def objective(trial: Trial, model_name: str, tr_loader: DataLoader, val_loader: DataLoader, n_classes: int):
    # Guard Clause
    err_string = f"Model {model_name} not found in models dictionary."
    if model_name not in MODELS: raise ValueError(err_string)
    
    # Retrieve and instantiate model
    model_cls = MODELS[model_name]
    trial_params = get_trial_params(trial, model_name)
    # Omit invalid configurations:
    try: model = model_cls(**trial_params, num_classes=n_classes)
    except RuntimeError: raise optuna.exceptions.TrialPruned()
    model.to(device)

    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Epoch Loop:
    total_time = 0.0
    for epoch in range(EPOCHS):
        start_time = time()
        # Training & Validation
        print(f"E={epoch + 1}", end="\t")
        _train(model, optimizer, criterion, tr_loader)
        vloss, vacc = _validate(model, val_loader, criterion)
        # Timing and reporting
        epoch_time = time() - start_time
        total_time += epoch_time
        # print(f"Epoch {epoch+1}/{EPOCHS} - Val Loss: {vloss:.4f}, Val Acc: {vacc:.2f}%, Time: {epoch_time:.2f}s")

        # Prune (i.e., early stopping) based on validation loss
        trial.report(vloss, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
    print(f"Total Training Time: {total_time:.2f}s")
    return vloss


def evaluate(model, test_loader):
    model.to(device)
    model.eval()
    n_classes = model.nclasses
    
    metrics = {
        "Accuracy": Accuracy(task="multiclass", num_classes=n_classes, average='macro').to(device),
        "Precision": Precision(task="multiclass", num_classes=n_classes, average='macro').to(device),
        "Recall": Recall(task="multiclass", num_classes=n_classes, average='macro').to(device),
        "F1Score": F1Score(task="multiclass", num_classes=n_classes, average='macro').to(device)
    }

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            # Update all metrics
            for metric in metrics.values():
                metric.update(preds, labels)

    # Compute & print metrics
    results = {name: metric.compute().item() for name, metric in metrics.items()}
    for name, value in results.items(): print(f"{name:<15}{value:.4f}")

    return results


def finetune(model):
    # Freeze all layers except the last fully connected layer
    for param in model.parameters(): param.requires_grad = False
    # Last layers are named fcf (for custom models):
    if hasattr(model, 'fcf'):
        for param in model.fcf.parameters():
            param.requires_grad = True
    else:
        # If the model has a different architecture, adjust accordingly
        raise ValueError("Model does not have a 'fcf' layer. Adjust finetune_setup accordingly.")


def tune(model_name, tr_loader, val_loader, n_classes):
    # Hyperparameter Tuning
    study = optuna.create_study(direction='minimize')
    print(len(tr_loader))
    print(len(val_loader))
    study.optimize(lambda trial: objective(trial, model_name, tr_loader, val_loader, n_classes), n_trials=N_TRIALS)
    
    return study.best_params

def train(model_name, tr_loader, n_classes, optimal_params):
    model_cls = MODELS[model_name]
    model = model_cls(**optimal_params, num_classes=n_classes)
    model.to(device)

    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    _train_loop(model, optimizer, criterion, tr_loader)
    return model
