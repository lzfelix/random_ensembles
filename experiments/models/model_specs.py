from typing import NamedTuple, List
from torch import nn
from models import ConvNet, MpegNet, CifarNet, Cifar100Net, KMnistNet


class Experiment(NamedTuple):
    net: nn.Module
    lb: List[float]
    ub: List[float]


experiment_configs = {
    'mnist': Experiment(
        net=ConvNet,
        lb=[1, 2, 1, 2, 50, 1e-3, 0],
        ub=[20, 9, 20, 9, 100, 1e-2, 1]
        # filters_1, kernel_1, filters_2, kernel_2, fc_size, lr, momentum
    ),
    'kmnist': Experiment(
        # net=KMnistNet,
        # lb=[1, 2, 1, 2, 50, 0, 1e-3, 0],
        # ub=[20, 9, 20, 9, 200, 0.9, 1e-2, 1]

        net=KMnistNet,
        lb=[ 1, 2,  1, 2,  50,   0, 1e-2],
        ub=[20, 9, 20, 9, 100, 0.9, 2]
        # filters_1, kernel_1, filters_2, kernel_2, fc_size, p_drop, lr
    ),
    'cifar10': Experiment(
        net=CifarNet,
        # lb=[1, 2, 1, 2, 1e-3, 0],
        # ub=[32, 9, 32, 9, 1e-2, 1]
        # filters_1, kernel_1, filters_2, kernel_2, lr, momentum

        lb=[1, 2, 1, 2, 50, 25, 1e-3, 0],
        ub=[32, 9, 32, 9, 200, 100, 1e-2, 1]
        # filters_1, kernel_1, filters_2, kernel_2, fc1, fc2, lr, momentum
    ),
    'cifar100': Experiment(
        net=Cifar100Net,
        # lb=[1, 2, 1, 2, 1e-3, 0],
        # ub=[32, 9, 32, 9, 1e-2, 1]
        # filters_1, kernel_1, filters_2, kernel_2, lr, momentum

        lb=[1, 2, 1, 2, 50, 25, 1e-3, 0],
        ub=[32, 9, 32, 9, 200, 100, 1e-2, 1]
        # filters_1, kernel_1, filters_2, kernel_2, fc1, fc2, lr, momentum
    ),
    'mpeg7': Experiment(
        net=MpegNet,
        lb=[1, 2, 1, 2, 1e-3, 0],
        ub=[20, 9, 20, 9, 1e-2, 1]
        # filters_1, kernel_1, filters_2, kernel_2, lr, momentum
    )
}


def get_experiment_setup(ds_name: str) -> Experiment:
    try:
        return experiment_configs[ds_name]
    except KeyError:
        print('Unknown dataset')
        exit(-1)
