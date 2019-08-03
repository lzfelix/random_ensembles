import argparse
import time

import numpy as np

import torch
from torch import optim as torch_opt
from torch.nn import functional as F

from flare.callbacks import Checkpoint
from flare import trainer

from models import utils
from models import datasets
from models.mnist import ConvNet

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
    # TODO: Show model learning rate / momentum
    # TODO: Keep track of time

    exec_params = get_exec_params()
    print(exec_params)

    device = torch.device('cpu')
    pin_memory = False
    if torch.cuda.is_available() and not exec_params.no_gpu:
        print('Running model on GPU')
        device = torch.device('cuda')
        pin_memory = True
        torch.backends.cudnn.benchmark = True

    trn_loader, val_loader, tst_loader = datasets.mnist_laoders(exec_params.batch_sz,
                                                                trn_split_sz=exec_params.trn_split,
                                                                pin_memory=pin_memory)

    # filters_1, kernel_1, filters_2, kernel_2, lr, momentum
    lower_bound = [1, 2, 1, 2, 1e-3, 0]
    upper_bound = [20, 9, 20, 9, 1e-2, 1]
    n_hyperparams = len(lower_bound)
    assert len(lower_bound) == len(upper_bound)

    h_names = ConvNet.learnable_hyperparams()
    model_prefix = f'./trained/{exec_params.ds_name}_random'

    accuracies = list()
    start_time = time.time()
    for model_no in range(exec_params.n_models):
        print(f'------------------------- Model {model_no + 1} / {exec_params.n_models} -------------------------')

        h_values = [_sample_value(lower_bound[i], upper_bound[i]) for i in range(n_hyperparams)]
        model_hyperparams = {name: round(value) for name, value in zip(h_names, h_values)}

        model = ConvNet(IMAGE_SZ, N_CLASSES, **model_hyperparams)
        print(model_hyperparams)
        print(model)

        model = model.to(device)
        loss_fn = F.nll_loss

        # The last two hyperparams are LR and momentum
        nn_optimizer = torch_opt.SGD(model.parameters(), lr=h_values[-1], momentum=h_values[-2])

        filename = '{}_{}'.format(model_prefix, model_no)
        cbs = [Checkpoint('val_accuracy', min_delta=1e-3, filename=filename, save_best=True, increasing=True)]

        # Training and getting the best model for evaluation
        history = trainer.train_on_loader(model, trn_loader, val_loader, loss_fn, nn_optimizer,
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