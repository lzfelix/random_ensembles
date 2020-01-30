import sys
import glob
from typing import *
from pathlib import Path


import numpy as np


def parse_folder(folder: str):
    files = glob.glob(str(Path(folder) / '*.txt'))
    values = [parse_file(file) for file in files]
    print(values)
    print('Accuracy: {} {}'.format(np.mean(values), np.std(values)))


def parse_file(filename: str) -> float:
    with open(filename) as file:
        data = file.readlines()[-1].split()
    return float(data[-1])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python parse_opt_ensemble.py logs_folder/')
        exit(-1)

    folder = Path(sys.argv[1])
    parse_folder(folder)
