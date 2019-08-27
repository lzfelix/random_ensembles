from typing import List
from torch import nn
from torch.nn import functional as F

from models import shapes


class KMnistNet(nn.Module):
    """From https://github.com/rois-codh/kmnist/blob/master/benchmarks/kuzushiji_mnist_cnn.py"""
    def __init__(self,
                 img_sz=32,
                 in_channels=1,
                 n_classes=10,
                 filters_1=32,
                 kernel_1=3,
                 filters_2=64,
                 kernel_2=3,
                 fc_size=128,
                 p_drop=0.25):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, filters_1, kernel_size=kernel_1)
        self.conv2 = nn.Conv2d(filters_1, filters_2, kernel_size=kernel_2)
        self.dropout = nn.Dropout(p=p_drop)

        # Ensuring that the 2nd kernel is not larger than the 1st layer output
        conv1_output = shapes.conv_2d_output(img_sz, kernel_1)
        kernel_2 = min(conv1_output, kernel_2)

        self.flat_shape = shapes.maxpool_2d_output(
            shapes.conv_2d_output(conv1_output, kernel_2),
            kernel_sz=2
        ) ** 2 * filters_2

        self.dense1 = nn.Linear(self.flat_shape, fc_size)
        self.dense2 = nn.Linear(fc_size, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        return F.log_softmax(self.dense2(x), dim=-1)

    @staticmethod
    def metric(y_pred, y_true):
        return dict(accuracy=(y_pred.argmax(-1) == y_true).sum().item())

    @staticmethod
    def learnable_hyperparams() -> List[str]:
        return [
            'filters_1',
            'kernel_1',
            'filters_2',
            'kernel_2',
            'fc_size',
            'p_drop'
        ]
