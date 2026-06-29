import torch
import torch.nn as nn
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)

        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):

    x = self.conv1(x)

    x = self.relu(x)

    x = self.pool(x)

    x = self.conv2(x)

    x = self.relu(x)

    x = self.pool(x)

    x = torch.flatten(x, 1)

    x = self.fc1(x)

    x = self.relu(x)

    x = self.fc2(x)

    return x