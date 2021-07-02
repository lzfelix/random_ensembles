from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from datasets import mpeg7


def mnist_loaders(batch_sz: int,
                  trn_split_sz: float = 0.8,
                  seed: int = 1337,
                  pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:

    return _load_dataset(datasets.MNIST, (0.1307,), (0.3081,), batch_sz, trn_split_sz, seed, pin_memory)


def kmnist_loaders(batch_sz: int,
                  trn_split_sz: float = 0.8,
                  seed: int = 1337,
                  pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:

    # How to compute mean and std values:
    #     transfs = transforms.Compose([transforms.ToTensor()])
    #     train_set = datasets.KMNIST(root='../data', train=True, download=True, transform=transfs)
    #     imgs = torch.cat([sample for sample, label in train_set])
    #     imgs.mean().item(), imgs.std().item()
    return _load_dataset(datasets.KMNIST, (0.1918,), (0.3483,), batch_sz, trn_split_sz, seed, pin_memory)


def cifar10_loaders(batch_sz: int,
                    trn_split_sz: float = 0.8,
                    seed: int = 1337,
                    pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:

    return _load_dataset(datasets.CIFAR10, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), batch_sz, trn_split_sz, seed, pin_memory)


def cifar100_loaders(batch_sz: int,
                    trn_split_sz: float = 0.8,
                    seed: int = 1337,
                    pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:

    return _load_dataset(datasets.CIFAR100, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), batch_sz, trn_split_sz, seed, pin_memory)


def _load_dataset(dataset_fn,
                  means: Tuple,
                  stds: Tuple,
                  batch_sz: int,
                  trn_split_sz: float = 0.8,
                  seed: int = 1337,
                  pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates reproducible train, valid, test splits from a given torchvision dataset.

    # Return
        trn_loader: The train loader, granted that samples are shuffled.
        val_loader: Tre validation loader, granted that samples are shuffled.
        tst_loader: The test loader, samples are *not* shuffled.
    """

    transfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    all_train = dataset_fn('../data/', train=True, transform=transfs, download=True)
    all_test = dataset_fn('../data/', train=False, transform=transfs, download=True)

    n_trn_samples = len(all_train)
    trn_sz = int(trn_split_sz * n_trn_samples)
    val_sz = n_trn_samples - trn_sz

    # Just the shuffling is exactly reproducible
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    trn_ds, val_ds = torch.utils.data.random_split(all_train, [trn_sz, val_sz])
    torch.set_rng_state(rng_state)

    trn_dl = DataLoader(trn_ds, batch_size=batch_sz, pin_memory=pin_memory, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_sz, pin_memory=pin_memory, shuffle=False)
    tst_dl = DataLoader(all_test, batch_size=batch_sz, pin_memory=pin_memory, shuffle=False)

    return trn_dl, val_dl, tst_dl


def mpeg7_loaders(batch_sz: int,
                  trn_split_sz: float = 0.8,
                  seed: int = 1337,
                  pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:

    # How to compute mean and std values:
    #     from datasets import mpeg7
    #     train_set = mpeg7.MPEG7('../data/')
    #     mean = train_set.x.astype(float).mean() / 255
    #     std = train_set.x.astype(float).std() / 255
    return _load_dataset(mpeg7.MPEG7, (0.1596,), (0.3469,), batch_sz, trn_split_sz, seed, pin_memory)
