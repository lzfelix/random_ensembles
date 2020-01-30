import sys
import glob
from typing import *
from pathlib import Path


import numpy as np


def parse_file(filename: str) -> List[float]:
    with open(filename) as file:
        metrics = file.readline().split()

    assert len(metrics) == 4
    return [float(metric) for metric in metrics]


def compute_metrics(folder: Path) -> Tuple[List[float], List[float], List[float], List[float]]:
    def spread(dest, data):
        for dd, da in zip(dest, data):
            dd.append(da)

    # test_accuracy, test_loss, test_time, train_time
    stats = ([], [], [], [])
    logs = glob.glob(str(folder / '*.txt'))
    for filename in logs:
        if '_tst_' in filename:
            data = parse_file(filename)
            spread(stats, data)

    return stats


def print_report(name: str, data: List[float]) -> None:
    def basic_stats(l):
        return np.mean(l), np.std(l)

    mean, std = basic_stats(data)
    print(f'{name.capitalize():20}\t{mean:4.4} ± {std:4.4}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python parse_baseline.py logs_folder/')
        exit(-1)

    folder = Path(sys.argv[1])
    metrics = compute_metrics(folder)

    print(f'Dataset: {folder.stem}')
    names = ['[test] accuracy', '[test] loss', '[test] time (s)', '[train] time (s)']
    for name, metric in zip(names, metrics):
        print_report(name, metric)
