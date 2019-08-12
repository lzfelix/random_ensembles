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
from models import model_specs
from datasets import specs

IMAGE_SZ = 28
N_CLASSES = 10


def get_exec_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage='Trains and learns a NN hyperparameters.')
    parser.add_argument('ds_name', help='Dataset', type=str, default='mnist')
    parser.add_argument('n_models', help='Amount of models to train', type=int, default=5)
    parser.add_argument('-n_epochs', help='NN Amount of epochs to train each candidate', type=int, default=3)
    parser.add_argument('-batch_sz', help='NN Batch size', type=int, default=128)
    parser.add_argument('-trn_split', help='Fraction of train data used for training', type=float, default=0.8)
    parser.add_argument('--no_gpu', help='Uses CPU instead of GPU', action='store_true')
    return parser.parse_args()


def _sample_value(lower: float, upper: float) -> float:
    return lower + np.random.rand() * (upper - lower)


if __name__ == '__main__':
    exec_params = get_exec_params()
    print(exec_params)

    device = torch.device('cpu')
    pin_memory = False
    if torch.cuda.is_available() and not exec_params.no_gpu:
        print('Running model on GPU')
        device = torch.device('cuda')
        pin_memory = True
        torch.backends.cudnn.benchmark = True

    ds_specs = specs.get_specs(exec_params.ds_name)
    train_loader, val_loader, tst_loader = ds_specs.loading_fn(exec_params.batch_sz,
                                                               trn_split_sz=exec_params.trn_split,
                                                               pin_memory=pin_memory)

    experiment = model_specs.experiment_configs[exec_params.ds_name]
    n_hyperparams = len(experiment.lb)
    assert len(experiment.lb) == len(experiment.ub)

    h_names = experiment.net.learnable_hyperparams()
    model_prefix = f'./trained/{exec_params.ds_name}_random'

    accuracies = list()
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
        nn_optimizer = torch_opt.SGD(model.parameters(), lr=h_values[-1], momentum=h_values[-2])

        filename = '{}_{}'.format(model_prefix, model_no)
        cbs = [Checkpoint('val_accuracy', min_delta=1e-3, filename=filename, save_best=True, increasing=True)]

        # Training and getting the best model for evaluation
        history = trainer.train_on_loader(model, train_loader, val_loader, loss_fn, nn_optimizer,
                                          n_epochs=exec_params.n_epochs, batch_first=True, device=device,
                                          callbacks=cbs)

        best_model = torch.load(filename + '.pth').to(device)
        eval_metrics = trainer.evaluate_on_loader(best_model, val_loader, loss_fn, batch_first=True,
                                                  device=device, verbosity=0)
        accuracies.append(eval_metrics['accuracy'])

        # Predicting on validation and test sets for ensemble learning / evaluation
        utils.predict_persist(best_model, val_loader, device,
                              f'predictions/{exec_params.ds_name}_random_{model_no}.txt')
        utils.predict_persist(best_model, tst_loader, device,
                              f'predictions/{exec_params.ds_name}_random_{model_no}_tst.txt')
    end_time = time.time()

    # Persisting labels for ensemble training / evaluation
    utils.store_labels(val_loader, f'./predictions/{exec_params.ds_name}_random_labels.txt')
    utils.store_labels(tst_loader, f'./predictions/{exec_params.ds_name}_random_labels_tst.txt')

    print(f'Finished. It took {end_time - start_time} seconds')
    for model_no, accuracy in enumerate(accuracies):
        print(f'\t. {model_no:<3} - {accuracy}')