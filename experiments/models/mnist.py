import torch.nn as nn
import torch.nn.functional as F

from . import shapes


class ConvNet(nn.Module):
    def __init__(self,
                 img_sz=28,
                 n_classes=10,
                 filters_1=10,
                 kernels_1=5,
                 filters_2=20,
                 kernels_2=5,
                 fc_size=50):
        """Simple ConvNet formed by two convolutional blocks (ie: conv followed by max-pooling)"""
        super().__init__()
        self.conv1 = nn.Conv2d(1, filters_1, kernel_size=kernels_1)
        self.conv2 = nn.Conv2d(filters_1, filters_2, kernel_size=kernels_2)
        self.conv2_drop = nn.Dropout2d()

        # Computing the square image flattened descriptor shape
        block_1_sz = shapes.block_2d_output(img_sz, kernels_1, 2)
        block_2_sz = shapes.block_2d_output(block_1_sz, kernels_2, 2)
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
