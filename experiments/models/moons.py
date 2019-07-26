import torch
from torch import nn
from torch.nn import functional as F


class SimpleNet(nn.Module):
    """Dummy network: input -> hidden1 -> hidden2 -> softmax(output)"""
    def __init__(self, n_fts, n_classes, hidden1=10, hidden2=5):
        super().__init__()
        self.fc1 = nn.Linear(n_fts, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def metric(prediction: torch.Tensor, y_true: torch.Tensor) -> dict:
        y_hat = prediction.argmax(-1)
        acc = (y_hat == y_true).sum().item()
        return {'accuracy': acc}
