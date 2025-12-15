import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMNISTNet(nn.Module):
    """
    Simple feedforward neural network for MNIST digit classification.
    Input: 28x28 grayscale images (flattened to 784)
    Output: 10 classes (digits 0-9)
    """

    def __init__(self):
        super(SimpleMNISTNet, self).__init__()
        # Flatten 28x28 = 784 input features
        self.fc1 = nn.Linear(784, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer
        self.fc3 = nn.Linear(64, 10)  # Output layer (10 classes)

    def forward(self, x):
        # Flatten the image if not already flattened
        x = x.view(-1, 784)

        # First layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Second layer with ReLU activation
        x = F.relu(self.fc2(x))

        # Output layer (no activation, used with CrossEntropyLoss)
        x = self.fc3(x)

        return x