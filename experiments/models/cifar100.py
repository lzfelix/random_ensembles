from typing import List
import torch.nn as nn
import torch.nn.functional as F

from models import shapes


class Cifar100Net(nn.Module):
    """From https://github.com/tensorflow/models/blob/master/research/slim/nets/cifarnet.py"""
    def __init__(self,
                 img_size=32,
                 in_channels=3,
                 n_classes=100,
                 filters_1=64,
                 kernel_1=5,
                 filters_2=64,
                 kernel_2=5,
                 fc_1=384,
                 fc_2=192,
                 p_drop=0.5):
        super().__init__()

        # Sanity check
        block1_output = shapes.block_2d_output(img_size, kernel_1, maxpool_kernel_sz=2)
        kernel_2 = min(kernel_2, block1_output)

        # Inferring flattened layer size
        block2_output = shapes.block_2d_output(block1_output, kernel_2, maxpool_kernel_sz=2)
        self.flat_shape = block2_output ** 2 * filters_2

        self.conv1 = nn.Conv2d(in_channels, filters_1, kernel_1)
        self.conv2 = nn.Conv2d(filters_1, filters_2, kernel_2)

        self.lrn = nn.LocalResponseNorm(size=4)
        self.dropout = nn.Dropout(p_drop)
        self.fc1 = nn.Linear(self.flat_shape, fc_1)
        self.fc2 = nn.Linear(fc_1, fc_2)
        self.fc3 = nn.Linear(fc_2, n_classes)

    @staticmethod
    def conv_relu(conv, x):
        return F.relu(conv(x))

    def forward(self, x):
        x = self.lrn(F.max_pool2d(self.conv_relu(self.conv1, x), 2))
        x = F.max_pool2d(self.lrn(self.conv_relu(self.conv2, x)), 2)

        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=-1)

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
            'fc_1',
            'fc_2',
        ]


# Leaving this here just in case, but the current CifarNET presents better results
# class Old_CifarNet(nn.Module):
#     def __init__(self,
#                  img_sz=32,
#                  n_channels=3,
#                  n_classes=10,
#                  filters_1=16,
#                  kernel_1=5,
#                  filters_2=32,
#                  kernel_2=5,
#                  fc1_size=120,
#                  fc2_size=84):
#         """ConvNet formed by two conv blocks (conv + by max-pooling)"""
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(n_channels, filters_1, kernel_size=kernel_1)
#         self.conv2 = nn.Conv2d(filters_1, filters_2, kernel_size=kernel_2)
#
#         # Computing the square image flattened descriptor shape
#         # Sanity check for filter size, since this may be random
#         block_1_sz = shapes.block_2d_output(img_sz, kernel_1, 2)
#         if block_1_sz <= kernel_2:
#             kernel_2 = block_1_sz
#         block_2_sz = shapes.block_2d_output(block_1_sz, kernel_2, 2)
#
#         self.flat_shape = block_2_sz ** 2 * filters_2
#
#         self.fc1 = nn.Linear(self.flat_shape, fc1_size)
#         self.fc2 = nn.Linear(fc1_size, fc2_size)
#         self.fc3 = nn.Linear(fc2_size, n_classes)
#
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), 2)
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, self.flat_shape)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return F.log_softmax(self.fc3(x), dim=-1)
#
#     @staticmethod
#     def metric(y_pred, y_true):
#         return dict(accuracy=(y_pred.argmax(-1) == y_true).sum().item())
#
#     @staticmethod
#     def learnable_hyperparams() -> List[str]:
#         return [
#             'filters_1',
#             'kernel_1',
#             'filters_2',
#             'kernel_2',
#             'fc1_size',
#             'fc2_size'
#         ]
