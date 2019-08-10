from typing import List
import torch.nn as nn
import torch.nn.functional as F

from models import shapes


class CifarNet(nn.Module):
    def __init__(self,
                 img_sz=32,
                 n_channels=3,
                 n_classes=10,
                 filters_1=32,
                 kernel_1=5,
                 filters_2=64,
                 kernel_2=5,
                 fc_size=64):
        """ConvNet formed by two conv blocks (conv + by max-pooling)"""
        super().__init__()

        self.conv1 = nn.Conv2d(n_channels, filters_1, kernel_size=kernel_1)
        self.conv2 = nn.Conv2d(filters_1, filters_2, kernel_size=kernel_2)
        self.conv2_drop = nn.Dropout2d()

        # Computing the square image flattened descriptor shape
        # Sanity check for filter size, since this may be random
        block_1_sz = shapes.block_2d_output(img_sz, kernel_1, 2)
        if block_1_sz <= kernel_2:
            kernel_2 = block_1_sz

        block_2_sz = shapes.block_2d_output(block_1_sz, kernel_2, 2)
        self.flat_shape = block_2_sz ** 2 * filters_2

        self.fc1 = nn.Linear(self.flat_shape, fc_size)
        self.fc2 = nn.Linear(fc_size, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.flat_shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=-1)

    @staticmethod
    def metric(y_pred, y_true):
        return dict(accuracy=(y_pred.argmax(-1) == y_true).sum().item())

    @staticmethod
    def learnable_hyperparams() -> List[str]:
        return [
            'filters_1',
            'kernel_1',
            'filters_2',
            'kernel_2'
        ]
