import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import transforms

from common import WORKER_BATCH_SIZE
from worker_model import WorkerModel


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

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 784)
        x = torch_f.relu(self.fc1(x))
        x = torch_f.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class WorkerMNistModel(WorkerModel):
    """
    Example WorkerModel that will use a shard of the MNIST dataset
    with each step to train a small feed-forward linear NN.
    """
    @staticmethod
    def make_shard_loader(worker_id: int, num_workers: int, batch_size: int) -> DataLoader:
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

        shard = Subset(ds, list(range(worker_id, len(ds), num_workers)))
        loader = DataLoader(shard, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False)

        return loader

    def __init__(self, worker_id: int, num_workers: int, batch_size: int):
        super().__init__(SimpleMNISTNet(), torch.nn.functional.cross_entropy, self.make_shard_loader(worker_id, num_workers, batch_size))