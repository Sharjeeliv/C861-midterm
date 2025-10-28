import torch

# Relative Imports
from .models.CNN import BasicCNN, LeNet5
from .models.ResNet import ResNet
from .models.VG import VGG16

from .train import tune, evaluate, train
from .data import load_dataset, combine_loaders
from .utils.utils import save_output, save_model
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASET_NCLS = {
    "ar": 28,
    "ur": 40,
    "en": 26
}
MODELS = {
    "BasicCNN": BasicCNN,
    "LeNet5": LeNet5,
    "ResNet": ResNet,
    "VGG16": VGG16,
}

def finetune():
    print("Finetuning BasicCNN pre-trained on English to Urdu dataset...")
    weights = "weights\BasicCNN_english_model.pth"
    model = BasicCNN(num_classes=26)
    model.load_state_dict(torch.load(weights))

    # Change final layer for Urdu dataset (36-40 classes)
    num_ftrs = model.fcf.in_features
    model.fcf = torch.nn.Linear(num_ftrs, 40)  # Adjust output classes for Urdu

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last conv layer and fc layer
    for param in model.conv3.parameters():
        param.requires_grad = True
    for param in model.fcf.parameters():
        param.requires_grad = True
    
    LANG = 'urdu'
    # Continue with training and evaluation
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    train_loader, val_loader, test_loader = load_dataset(LANG, class_n=480)
    tune(model, optimizer, criterion, train_loader, "BasicCNN_Finetuned-en", LANG)
    evaluate(model, test_loader)


def main():
    
    LANG = 'arabic'  # Change as needed
    train_loader, val_loader, test_loader = load_dataset(LANG, class_n=480)

    for model_name in MODELS.keys():
        if model_name in {'BasicCNN', 'LeNet5', 'ResNet'}:
            print(f"Skipping {model_name}")
            continue
        print(f"Training model: {model_name} on {LANG} dataset")
        model = MODELS[model_name](num_classes=DATASET_NCLS[LANG])
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        tune(model, optimizer, criterion, train_loader, model_name, LANG)
        evaluate(model, test_loader)



def testing():
    model_name = 'LeNet5'
    split = 480
    for lang in ['ar']:
        train_loader, val_loader, test_loader = load_dataset(lang, class_n=split)
        optimal_params = tune(model_name, train_loader, val_loader, DATASET_NCLS[lang], lang, split)

        combined_loader = combine_loaders(train_loader, val_loader)
        model = train(model_name=model_name, tr_loader=combined_loader, n_classes=DATASET_NCLS[lang], optimal_params=optimal_params)
        res = evaluate(model, test_loader)

        title = f"{lang}{split}_{model_name}"
        save_output(res, optimal_params, title, ROOT)
        save_model(model, model_name, lang, split, ROOT)


if __name__ == "__main__":
    testing()