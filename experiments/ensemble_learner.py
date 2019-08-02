import argparse
from typing import List

import numpy as np
from models import utils
from models import ensemble

from opytimizer.optimizers.pso import PSO


def get_exec_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage='Combines different neural nets into a single ensemble')
    parser.add_argument('val_ground', help='Path to the validation ground truth labels', type=str)
    parser.add_argument('tst_ground', help='Path to the test ground truth labels', type=str)
    parser.add_argument('-val_preds', help='List of candidate predictions in the validation set', type=str, nargs='+',
                        required=True)
    parser.add_argument('-n_agents', help='MH amount of agents', type=int, default=20)
    parser.add_argument('-n_iters', help='MH amount of iterations', type=int, default=10)

    return parser.parse_args()


def _show_files(split_name: str, ground, preds):
    print(f'{split_name.capitalize()} ground: {ground}')
    print('Prediction files:')
    for filename in preds:
        print(f'\t- {filename}')
    print()


def deduce_test_files(filepaths: List[str]) -> List[str]:
    test_files = list()
    for filepath in filepaths:
        extension_dot = filepath.rfind('.')
        test_file = filepath[:extension_dot] + '_tst' + filepath[extension_dot:]
        test_files.append(test_file)
    return test_files


if __name__ == '__main__':
    exec_args = get_exec_args()
    tst_preds = deduce_test_files(exec_args.val_preds)

    _show_files('validation', exec_args.val_ground, exec_args.val_preds)
    _show_files('test', exec_args.tst_ground, tst_preds)
    print(exec_args)

    val_all_preds, val_y_true = ensemble.load_candidates_preds(exec_args.val_preds, exec_args.val_ground)
    tst_all_preds, tst_y_true = ensemble.load_candidates_preds(tst_preds, exec_args.tst_ground)

    if val_all_preds.shape[0] != tst_all_preds.shape[0]:
        raise RuntimeError('Amount of predictions for validation and test sets differ')

    ensemble_fun = ensemble.make_combination(val_all_preds, val_y_true)

    n_variables = val_all_preds.shape[0]
    lb = [0] * n_variables
    ub = [1] * n_variables
    mh_hyperparams = dict(w=1.7, c1=2, c2=1.7)

    history = utils.optimize(PSO,
                             target=ensemble_fun,
                             n_agents=exec_args.n_agents,
                             n_variables=n_variables,
                             n_iterations=exec_args.n_iters,
                             lb=lb,
                             ub=ub,
                             hyperparams=mh_hyperparams)

    best_weights = np.asarray(history.best_agent[0][0]).flatten()
    print('Learned weights')
    print('---------------')
    print(best_weights)

    print('Inidividual val. model accuracies')
    print('---------------------------------')
    for i, pred in enumerate(val_all_preds):
        acc = utils.accuracy(pred, val_y_true)
        print(f'\t. {i:<3} - {acc}')

    val_acc = ensemble.evaluate_ensemble(best_weights, val_all_preds, val_y_true)
    print(f'Ensemble val. accuracy: {val_acc}')

    tst_acc = ensemble.evaluate_ensemble(best_weights, tst_all_preds, tst_y_true)
    print(f'Ensemble tst. accuracy: {tst_acc}')
