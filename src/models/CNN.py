import torch
import torch.nn as nn
import torch.nn.functional as F

# Adaptable CNN Model:
class CNN(nn.Module):
    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        num_conv_layers=3,
        filters=32,
        kernel_size=3,
        activation="ReLU",
        fc_size=128,
        dropout=0.5,
    ):
        super(CNN, self).__init__()

        self.nclasses = num_classes

        # Activation function
        if activation == "ReLU":        act_fn = nn.ReLU
        elif activation == "LeakyReLU": act_fn = nn.LeakyReLU
        elif activation == "ELU":       act_fn = nn.ELU
        else:   raise ValueError(f"Unknown activation {activation}")

        # --- Feature extractor ---
        # Can also experiment with batch norm and pooling mechanism
        layers = []
        in_channels = input_channels

         # Automatically scale filters per layer
        for i in range(num_conv_layers):
            out_channels = filters * (2 ** i)  # 32 → 64 → 128 → 256 ...
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(act_fn())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)

        # compute flatten size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)  # batch=1, 3 channels
            dummy_output = self.feature_extractor(dummy_input)
            self.flatten_size = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Pytorch Basic CNN Model:
class BasicCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super(BasicCNN, self).__init__()
        self.nclasses = num_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fcf = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fcf(x)
        return x


# Pytorch LeNet-5 Model:
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.nclasses = num_classes
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fcf = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fcf(x)
        return x