import argparse
import time

import numpy as np

import torch
from torch import optim as torch_opt
from torch.nn import functional as F

from flare.callbacks import Checkpoint
from flare import trainer

from misc import utils
from misc import logs
from models import utils as ut
from models import model_specs
from datasets import specs


def get_exec_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage='Trains and learns a NN hyperparameters.')
    parser.add_argument('ds_name', help='Dataset', type=str, default='mnist')
    parser.add_argument('n_models', help='Amount of models to train', type=int, default=5)
    parser.add_argument('-n_epochs', help='NN Amount of epochs to train each candidate', type=int, default=3)
    parser.add_argument('-batch_sz', help='NN Batch size', type=int, default=128)
    parser.add_argument('-trn_split', help='Fraction of train data used for training', type=float, default=0.8)
    parser.add_argument('--no_gpu', help='Uses CPU instead of GPU', action='store_true')
    parser.add_argument('--show_test', help='Shows model accuracy @ train test in the end summary', action='store_true')
    return parser.parse_args()


def _sample_value(lower: float, upper: float) -> float:
    return lower + np.random.rand() * (upper - lower)


if __name__ == '__main__':
    exec_params = get_exec_params()
    print(exec_params)

    device, pin_memory = ut.get_device(exec_params.no_gpu)

    ds_specs = specs.get_specs(exec_params.ds_name)
    use_sgd = exec_params.ds_name.lower() != 'kmnist'
    print(f'Using SGD: {use_sgd}')

    train_loader, val_loader, tst_loader = ds_specs.loading_fn(exec_params.batch_sz,
                                                               trn_split_sz=exec_params.trn_split,
                                                               pin_memory=pin_memory)
    experiment = model_specs.experiment_configs[exec_params.ds_name]
    n_hyperparams = len(experiment.lb)
    assert len(experiment.lb) == len(experiment.ub)

    h_names = experiment.net.learnable_hyperparams()
    model_prefix = f'./trained/{exec_params.ds_name}_random'

    val_accuracies = list()
    tst_accuracies = list()
    start_time = time.time()
    for model_no in range(exec_params.n_models):
        print(f'------------------------- Model {model_no + 1} / {exec_params.n_models} -------------------------')

        h_values = [_sample_value(experiment.lb[i], experiment.ub[i]) for i in range(n_hyperparams)]
        logs.print_hyperparams(h_names, h_values)

        model_hyperparams = {name: int(round(value)) for name, value in zip(h_names, h_values)}

        model = experiment.net(ds_specs.img_size, ds_specs.n_channels, ds_specs.n_classes, **model_hyperparams)
        print(model_hyperparams)
        print(model)

        model = model.to(device)
        loss_fn = F.nll_loss

        # The last two hyperparams are LR and momentum
        if use_sgd:
            nn_optimizer = torch_opt.SGD(model.parameters(), lr=h_values[-1], momentum=h_values[-2])
        else:
            nn_optimizer = torch_opt.Adadelta(model.parameters(), lr=h_values[-1])

        filename = '{}_{}'.format(model_prefix, model_no)
        cbs = [Checkpoint('val_accuracy', min_delta=1e-3, filename=filename, save_best=True, increasing=True)]

        # Training and getting the best model for evaluation
        history = trainer.train_on_loader(model, train_loader, val_loader, loss_fn, nn_optimizer,
                                          n_epochs=exec_params.n_epochs, batch_first=True, device=device,
                                          callbacks=cbs)

        best_model = torch.load(filename + '.pth').to(device)

        # Predicting on validation and test sets for ensemble learning / evaluation
        val_acc = utils.predict_persist(best_model, val_loader, device,
                                        f'predictions/{exec_params.ds_name}_random_{model_no}.txt')
        tst_acc = utils.predict_persist(best_model, tst_loader, device,
                                        f'predictions/{exec_params.ds_name}_random_{model_no}_tst.txt')
        val_accuracies.append(val_acc)
        tst_accuracies.append(tst_acc)

    end_time = time.time()

    # Persisting labels for ensemble training / evaluation
    utils.store_labels(val_loader, f'./predictions/{exec_params.ds_name}_random_labels.txt')
    utils.store_labels(tst_loader, f'./predictions/{exec_params.ds_name}_random_labels_tst.txt')

    # Printing the report
    print(f'Finished. It took {end_time - start_time} seconds')

    # Had to rewrite the same code a thousand times, now I have a thousand versions of it :)
    print('Complete model scores')
    print('{:10} {:10} {:10}'.format('Model ID', 'acc @ val', 'acc @ tst'))
    print('-' * 33)
    if not exec_params.show_test:
        tst_accuracies = ['????'] * len(tst_accuracies)

    for model_no, (val_acc, tst_acc) in enumerate(zip(val_accuracies, tst_accuracies)):
        print(f'{model_no:<10} {val_acc:10.4} {tst_acc:10.4}')
    print('Done.')
