"""
Computes the accuracy mean and standard deviation for the experiment
that combines several neural networks trained with random hyperparameters
in a single ensemble.

This script assumes the following tree structure:

./
    CIFAR10/                        <- dataset name
        10/                         <- ensemble size
            experiment_0.txt        <- experiment id. Filename doesn't matter
            ...
            experiment_n.txt
        15/
            experiment_0.txt
            ...
            experiment_n.txt
        ...
    MNIST/
        10/
            ...

# Notes:
    - All files, except ones starting with '.' are read.
    - The name of the experiment files does not matter.

# Todo:
    - Compute mean activation for each model in the ensembles
"""

import re
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=str, help='Experiment results root folder.')
    return parser.parse_args()


def get_metrics(experiment_log: Path) -> Tuple[float, float]:
    iterator = experiment_log.open('r')
    val_acc = 0
    tst_acc = 0

    for line in iterator:
        if 'val. accuracy:' in line:
            val_acc = float(line.split()[-1])
        elif 'tst. accuracy:' in line:
            tst_acc = float(line.split()[-1])
    return val_acc, tst_acc


def compute_experiment_metrics(base_path: Path) -> Tuple[float, float, float, float]:
    all_val_acc = []
    all_tst_acc = []
    for experiment_log in base_path.iterdir():
        if experiment_log.stem[0] == '.':
            continue
        cur_val_acc, cur_tst_acc = get_metrics(experiment_log)
        all_val_acc.append(cur_val_acc)
        all_tst_acc.append(cur_tst_acc)
    
    return np.mean(all_val_acc), np.std(all_val_acc),\
           np.mean(all_tst_acc), np.std(all_tst_acc)


def sort_by_ensemble_size(filepath: Path):
    numbers = re.findall(r'\d+', filepath.stem)
    if not numbers:
        return 1000
    return int(numbers[0])


if __name__ == "__main__":
    args = get_args()
    path = Path(args.root)

    for dataset in path.iterdir():
        if dataset.stem[0] == '.':
            continue

        print(f'Dataset: {dataset.stem}')
        for experiment in sorted(dataset.iterdir(), key=sort_by_ensemble_size):
            if experiment.stem[0] == '.':
                continue
            
            mean_val, std_val, mean_tst, std_tst = compute_experiment_metrics(experiment)
            print(f'\t# Models: {experiment.stem:2}. ' +
                  f'Accuracy (val): {mean_val:4.4f} ± {std_val:4.4f} ' +
                  f'Accuracy (tst): {mean_tst:4.4f} ± {std_tst:4.4f}')
