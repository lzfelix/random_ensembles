import re
import argparse
from typing import Tuple, List
from pathlib import Path

import numpy as np


def get_filepath() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'root',
        type=str,
        help='Path to the *ensemble* log folder using 15 models.',
        default='../results/random_ensemble/CIFAR10/15/'
    )
    return parser.parse_args()


def extract_metrics(log_filepath: Path) -> Tuple[List[float], List[float]]:
    # Reading file upside down, since model metrics are at the bottom
    selected_lines = []
    for line in log_filepath.open('r').readlines()[::-1]:
        if '---' in line:
            break
        selected_lines.append(line)
    
    # Model metrics should contain two float numbers: val. accuracy
    # and tst. accuracy. Filter lines with two numbers bc of this.
    line_numbers = map(lambda s: re.findall(r'(\d\.\d+)', s), selected_lines)
    lines_with_accuracy = list(filter(lambda s: len(s) == 2, line_numbers))

    val_acc = list(map(lambda metric: float(metric[0]), lines_with_accuracy))
    tst_acc = list(map(lambda metric: float(metric[1]), lines_with_accuracy))

    return list(val_acc), list(tst_acc)


def mean_std(x: List[float]) -> Tuple[float, float]:
    return np.mean(x), np.std(x)


if __name__ == "__main__":
    exec_args = get_filepath()
    root_folder = Path(exec_args.root)

    # All files for the largest ensemble contain the test metrics for each
    # network that form it. If any file is parsed all metrics are retrieved
    experiment_logs = list(filter(lambda x: x.stem[0] != '.', root_folder.iterdir()))
    val_acc, tst_acc = extract_metrics(experiment_logs[0])

    print('Val accuracy: {:4.4f} ± {:4.4f}'.format(*mean_std(val_acc)))
    print('Tst accuracy: {:4.4f} ± {:4.4f}'.format(*mean_std(tst_acc)))
    print(f'# entries: {len(val_acc)}')
