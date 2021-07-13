import re
import argparse
from pathlib import Path

import numpy as np


def get_arsg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract metrics from logs output by runner.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=str, help='Folder containing a set of outputs. Each network run should be a file')
    return parser.parse_args()


def extract_metrics(p: Path):
    lines = list(filter(lambda x: len(x) > 0, p.open('r').readlines()))
    test_metrics = eval(lines[-1])

    train_time = lines[-3]
    train_time_in_seconds = re.findall(r'(\d+\.\d+) seconds', train_time)[0]

    return test_metrics['accuracy'], test_metrics['loss'], float(train_time_in_seconds)


if __name__ == '__main__':
    args = get_arsg()
    path = Path(args.root)

    counter = 0
    all_acc = []
    all_loss = []
    all_train_time = []
    for experiment_log in path.iterdir():
        # Skip hidden files
        ds_name = experiment_log.stem
        if ds_name[0] == '.':
            continue

        print(f'Reading {experiment_log}')
        counter += 1
        test_accuracy, test_loss, train_time = extract_metrics(experiment_log)
        print('\t', test_accuracy, test_loss, train_time, '\n')

        all_acc.append(test_accuracy)
        all_loss.append(test_loss),
        all_train_time.append(train_time)

    print(f'Read {counter} experiment logs')
    print(f'Test accuracy:  {np.mean(all_acc)} ± {np.std(all_acc)}')
    print(f'Test loss:      {np.mean(all_loss)} ± {np.std(all_loss)}')
    print(f'Train time:     {np.mean(all_train_time)} ± {np.std(all_train_time)}')
