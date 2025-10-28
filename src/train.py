# Local
from .utils.optuna import get_trial_params
from .models.CNN import BasicCNN, LeNet5, CNN
from .models.ResNet import ResNet
from .models.VG import VGG16
from .utils.utils import EarlyStopping

# First-party
from time import time
import os
from pathlib import Path

# Third-party
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
import optuna
from optuna import Trial
import optuna.visualization as vis
from torch.utils.data import DataLoader


# ********************************
# CONSTANTS, VARIABLES AND SETUP
# ********************************
FT_EPOCHS = 30
EPOCHS = 15
N_TRIALS = 5
WEIGHTS_DIR = "weights"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).parent.parent

MODELS = {
    "BasicCNN": BasicCNN,
    "LeNet5": LeNet5,
    "ResNet18": ResNet,
    "VGG16": VGG16,
    "CNN": CNN
}

# ********************************
# HELPER FUNCTIONS
# ********************************
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


def _train_loop(model, optimizer, criterion, train_loader, epochs=EPOCHS):
    early_stopping = EarlyStopping()
    total_time = 0.0
    for epoch in range(epochs):
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


def get_model(model_name, trial_params, n_classes):
    model_cls = MODELS[model_name]
    if model_name != "CNN": return model_cls(num_classes=n_classes)
    return model_cls(**trial_params, num_classes=n_classes)


def get_optimizer(model_name, model, trial_params):
    if model_name != "CNN": return torch.optim.Adam(model.parameters(), lr=0.001)
    # Unpack parameters
    optimizer_type = trial_params.get("optimizer", "Adam")
    lr = trial_params.get("lr", 1e-3)
    weight_decay = trial_params.get("weight_decay", 0.0)
    # Build optimizer for remaining models
    if optimizer_type == "SGD":
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=lr, weight_decay=weight_decay)
    else: raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def objective(trial: Trial, model_name: str, tr_loader: DataLoader, val_loader: DataLoader, n_classes: int):
    # Guard Clause
    err_string = f"Model {model_name} not found in models dictionary."
    if model_name not in MODELS: raise ValueError(err_string)
    
    # Retrieve and instantiate model
    trial_params = get_trial_params(trial, model_name)
    # Omit invalid configurations:
    try: model = get_model(model_name, trial_params, n_classes)
    except RuntimeError: raise optuna.exceptions.TrialPruned()
    model.to(device)

    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model_name, model, trial_params)

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


def _update_classifier(model, n_letters):
    # last layer is always called fcf, replace classifier head
    num_feats = model.fcf.in_features
    model.fcf = torch.nn.Linear(num_feats, n_letters)
    # Freeze all parameters
    for param in model.parameters(): 
        param.requires_grad = False
    # Update classifier
    for param in model.fcf.parameters():
        param.requires_grad = True
    return model

# ********************************
# INTERFACE FUNCTIONS
# ********************************
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
    

def finetune(model_name, weights_path, n_letters, tr_loader, optimal_params, nclasses):
    # Retrieve, initialize and update model
    model = get_model(model_name, optimal_params, nclasses)
    model.load_state_dict(torch.load(weights_path))
    _update_classifier(model, n_letters)
    model.nclasses = n_letters
    model.to(device)

    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model_name, model, optimal_params)
    _train_loop(model, optimizer, criterion, tr_loader, epochs=FT_EPOCHS)
    return model
    

def tune(model_name, tr_loader, val_loader, n_classes, lang, split):
    # Hyperparameter Tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model_name, tr_loader, val_loader, n_classes), 
                   n_trials=N_TRIALS)
    # Figures & Plotting
    figs = {
        "optim_history": vis.plot_optimization_history(study),
        "param_importance": vis.plot_param_importances(study),
        "parallel_coord": vis.plot_parallel_coordinate(study),
        "slice": vis.plot_slice(study),
        "contour": vis.plot_contour(study),
    }

    fig_dir = ROOT / "results" / "figures"
    os.makedirs(fig_dir, exist_ok=True)

    for name, fig in figs.items():
        fig.write_image(fig_dir / f"{lang}{split}_{model_name}_{name}.png")
    return study.best_params


def train(model_name, tr_loader, n_classes, optimal_params):
    model = get_model(model_name, optimal_params, n_classes)
    model.to(device)

    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model_name, model, optimal_params)
    _train_loop(model, optimizer, criterion, tr_loader)
    return model
