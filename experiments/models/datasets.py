from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def mnist_loaders(batch_sz: int,
                  trn_split_sz: float = 0.8,
                  seed: int = 1337,
                  pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates reproducible train, valid, test splits from the MNIST dataset.

    # Return
        trn_loader: The train loader, granted that samples are shuffled.
        val_loader: Tre validation loader, granted that samples are shuffled.
        tst_loader: The test loader, samples are *not* shuffled.
    """
    transfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
    )

    all_train = datasets.MNIST('../data/', train=True, transform=transfs, download=True)
    all_test = datasets.MNIST('../data/', train=False, transform=transfs, download=True)

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
