import re
import argparse
from pathlib import Path

import numpy as np


def get_arsg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract metrics from logs output by hyper_learner.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=str, help='Folder containing a set of outputs. Each optimization '
                                               'experiment should be a file')
    return parser.parse_args()


def extract_metrics(p: Path):
    lines = list(filter(lambda x: len(x) > 0, p.open('r').readlines()))

    # The first line after --- contains the fittest network with learnt hyperparameters
    last_line = ''
    for line in reversed(lines):
        if '-----' in line:
            break
        last_line = line

    accuracies = re.findall(r'(\d+\.\d+)', last_line)
    test_accuracy = float(accuracies[2])

    # Get experiment running time
    for line in reversed(lines):
        if 'It took' in line:
            running_time = float(re.findall(r'took (\d+\.\d+) se', line)[0])
            break
    else:
        raise RuntimeError('Could not find execution time for experiment!')

    return test_accuracy, running_time


if __name__ == '__main__':
    args = get_arsg()
    path = Path(args.root)

    counter = 0
    all_acc = []
    all_running_times = []
    for experiment_log in path.iterdir():
        # Skip hidden files
        ds_name = experiment_log.stem
        if ds_name[0] == '.':
            continue

        print(f'Reading {experiment_log}')
        counter += 1
        test_accuracy, running_time = extract_metrics(experiment_log)
        print('\t', test_accuracy, running_time, '\n')

        all_acc.append(test_accuracy)
        all_running_times.append(running_time)

    print(f'Read {counter} experiment logs')
    print(f'Test accuracy:  {np.mean(all_acc)} ± {np.std(all_acc)}')
    print(f'Train time:     {np.mean(all_running_times)} ± {np.std(all_running_times)}')
