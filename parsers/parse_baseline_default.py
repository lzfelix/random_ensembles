"""
Computes accuracy mean and standard deviation, as well as training time for 
predefined architectures using the predefined hyperparameters as well. This
scripts assumes the following tree structure:

./
    CIFAR10/
        cifar10_tst_metrics_1.txt
        cifar10_trn_metrics_1.txt
        ...
    MNIST/
        ...

# Notes
    - Files containing the substring 'trn' are ignored
    - The name of the experiment files do not matter, as long as tst files do not
    contain the "trn" substring
    - The format for each test log file should be a single line containing
        (test_accuracy, test_loss, test_time, train_time)
"""

import argparse
from pathlib import Path

import numpy as np


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=str, help='Experiment results root folder.')
    return parser.parse_args()


def get_metrics(experiment_dir: Path):
    all_acc = []
    all_time = []

    # Fields order: (test_accuracy, test_loss, test_time, train_time)
    for exp_log in experiment_dir.iterdir():
        if 'trn' in exp_log.stem:
            continue

        data = exp_log.read_text().rstrip()
        tst_acc, _, _, trn_time = list(map(float, data.split()))
        all_acc.append(tst_acc)
        all_time.append(trn_time)

    return np.mean(all_acc), np.std(all_acc), \
           np.mean(all_time), np.std(all_time)


if __name__ == "__main__":
    args = get_args()
    path = Path(args.root)

    for dataset_experiment in path.iterdir():
        ds_name = dataset_experiment.stem
        if ds_name[0] == '.' or not dataset_experiment.is_dir():
            continue

        print(ds_name)
        mean_tst_acc, std_tst_acc, mean_trn_time, std_trn_time = get_metrics(dataset_experiment)
        print(f'\t{mean_tst_acc:4.4f} ± {std_tst_acc:4.4f} '+
              f'(Trn. time {mean_trn_time:4.4f} ± {std_trn_time:4.2f} seconds)')
