import argparse
from typing import List

import numpy as np
from misc import utils
from models import ensemble

from opytimizer.optimizers.swarm import PSO


def get_exec_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage='Combines different neural nets into a single ensemble')
    parser.add_argument('val_ground', help='Path to the validation ground truth labels', type=str)
    parser.add_argument('tst_ground', help='Path to the test ground truth labels', type=str)
    parser.add_argument('-val_preds', help='List of candidate predictions in the validation set', type=str, nargs='+',
                        required=True)
    parser.add_argument('-n_iters', help='MH amount of iterations', type=int, default=10)
    parser.add_argument('--show_test', help='Shows model accuracy @ train test in the end summary', action='store_true')

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

    n_agents = val_all_preds.shape[0]
    if val_all_preds.shape[0] != tst_all_preds.shape[0]:
        raise RuntimeError('Amount of candidates for validation and test sets differ')

    ensemble_fun = ensemble.make_combination(val_all_preds, val_y_true)

    n_variables = val_all_preds.shape[0]
    lb = [0] * n_variables
    ub = [1] * n_variables
    mh_hyperparams = dict(w=0.7, c1=1.7, c2=1.7)

    history = utils.optimize(PSO,
                             target=ensemble_fun,
                             n_agents=n_agents,
                             n_variables=n_variables,
                             n_iterations=exec_args.n_iters,
                             lb=lb,
                             ub=ub,
                             hyperparams=mh_hyperparams)

    best_weights = np.asarray(history.best_agent[-1][0]).flatten()
    print('Learned weights')
    print('---------------')
    print(best_weights)

    print('Individual val. model accuracies')
    print('{:10} {:10} {:10}'.format('Model ID', 'acc @ val', 'acc @ tst'))
    print('-' * 33)
    for i, (val_pred, tst_pred) in enumerate(zip(val_all_preds, tst_all_preds)):
        val_acc = utils.accuracy(val_pred, val_y_true)
        tst_acc = utils.accuracy(tst_pred, tst_y_true) if exec_args.show_test else '????'
        print(f'{i:<10} {val_acc:>10.4}{tst_acc:>10.4}')

    val_acc = ensemble.evaluate_ensemble(best_weights, val_all_preds, val_y_true)
    tst_acc = ensemble.evaluate_ensemble(best_weights, tst_all_preds, tst_y_true)

    tst_acc = tst_acc if exec_args.show_test else '????'
    print(f'\nEnsemble val. accuracy: {val_acc}')
    print(f'Ensemble tst. accuracy: {tst_acc}')
