import argparse

import numpy as np
import torch
from torch.nn import functional as F
from torch import optim as torch_opt

from flare import trainer
from flare.callbacks import Checkpoint

from opytimizer.optimizers.fa import FA

from models.mnist import ConvNet
from datasets import datasets as ds
from datasets import specs
from misc import utils

callno = 0
scoreboard = dict()


def get_exec_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage='Trains and learns a NN hyperparameters.')
    parser.add_argument('ds_name', help='Dataset', type=str, default='mnist')
    parser.add_argument('mh_name', help='MH name?', type=str)
    parser.add_argument('-n_agents', help='MH amount of agents', type=int, default=20)
    parser.add_argument('-n_iters', help='MH amount of iterations', type=int, default=3)
    parser.add_argument('-n_epochs', help='NN Amount of epochs to train each candidate', type=int, default=3)
    parser.add_argument('-batch_sz', help='NN Batch size', type=int, default=128)
    parser.add_argument('-top_k', help='Retrieve the top k best neural nets found', type=int, default=10)
    parser.add_argument('-trn_split', help='Fraction of train data used for training', type=float, default=0.8)
    parser.add_argument('--no_gpu', help='Uses CPU instead of GPU', action='store_true')
    return parser.parse_args()


def make_target_fn(model_prefix, device, model_class, trn_gen, val_gen, n_epochs, image_sz, n_channels, n_classes, hyperparams):
    """Creates a target function to be optimized based on some neural net, data and learnable hyperparams

    # Arguments
        model_prefix: All model predictions will be saved with this prefix, for instance the prefix `trained/pso_1`
            causes models to be persisted as `trained/pso_1.pth`.
        device: Either GPU or CPU.
        model_class: The class of the neural net to be trained and optimized.
        trn_gen: A PyTorch DataLoader with training samples.
        val_gen: A PyTorch DataLoader with validation samples. The models are selected based on the accuracy on this
            split of the dataset.
        n_epochs: For how many epochs train each candidate model.
        image_sz: Samples should be square image sized.
        n_classes:
        hyperparams: Names of the hyperparams to be learned via the metaheuristic.

    # Return
        A closure that should be called with an array of model hyperparams. This second function returns
            `1 - (model_accuracy @ val_set)`.
    """
    def target_fn(hyperparam_values):
        global callno
        global scoreboard

        # Ensuring that hyperparams is a 1D-tensor
        hyperparam_values = np.asarray(hyperparam_values).ravel()

        model_hyperparams = {hname: int(round(hvalue)) for hname, hvalue in zip(hyperparams, hyperparam_values)}
        model = model_class(image_sz, n_channels, n_classes, **model_hyperparams)
        print(model)

        model = model.to(device)
        loss_fn = F.nll_loss

        # The last two hyperparams are LR and momentum
        nn_optimizer = torch_opt.SGD(model.parameters(), lr=hyperparam_values[-1], momentum=hyperparam_values[-2])

        filename = '{}_{}'.format(model_prefix, callno)
        cbs = [Checkpoint('val_accuracy', min_delta=1e-3, filename=filename, save_best=True, increasing=True)]

        # Training
        history = trainer.train_on_loader(model, trn_gen, val_gen, loss_fn, nn_optimizer,
                                          n_epochs=n_epochs, batch_first=True, device=device,
                                          callbacks=cbs)

        # Getting the best model during training to evaluate
        best_model = torch.load(filename + '.pth').to(device)
        eval_metrics = trainer.evaluate_on_loader(best_model, val_gen, loss_fn, batch_first=True,
                                                  device=device, verbosity=0)

        # Opytimizer minimizes functions
        fitness = 1 - eval_metrics['accuracy']
        scoreboard[callno] = fitness
        callno += 1
        return fitness
    return target_fn


if __name__ == '__main__':
    # TODO: Figure out mean and std values for MPEG7
    # TODO: Add support for more metaheuristics
    # TODO: Add support for metaheuristics hyperparams selection
    # TODO: Show model learning rate / momentum
    # TODO: check log-maths

    exec_params = get_exec_params()
    print(exec_params)

    device = torch.device('cpu')
    pin_memory = False
    if torch.cuda.is_available() and not exec_params.no_gpu:
        device = torch.device('cuda')
        pin_memory = True
        torch.backends.cudnn.benchmark = True

    ds_specs = specs.get_specs(exec_params.ds_name)
    print(ds_specs)

    train_loader, val_loader, tst_loader = ds_specs.loading_fn(exec_params.batch_sz,
                                                               trn_split_sz=exec_params.trn_split,
                                                               pin_memory=pin_memory)

    target_fn = make_target_fn(f'./trained/{exec_params.ds_name}_{exec_params.mh_name}',
                               device,
                               ConvNet,
                               train_loader,
                               val_loader,
                               n_epochs=exec_params.n_epochs,
                               image_sz=ds_specs.img_size,
                               n_channels=ds_specs.n_channels,
                               n_classes=ds_specs.n_classes,
                               hyperparams=ConvNet.learnable_hyperparams())

    # filters_1, kernel_1, filters_2, kernel_2, lr, momentum
    lower_bound = [1, 2, 1, 2, 1e-3, 0]
    upper_bound = [20, 9, 20, 9, 1e-2, 1]

    n_variables = len(lower_bound)
    mh_hyperparams = dict(alpha=0.5, beta=0.2, gamma=1.0)

    # Learning the model
    history = utils.optimize(FA,
                             target=target_fn,
                             n_agents=exec_params.n_agents,
                             n_variables=n_variables,
                             n_iterations=exec_params.n_iters,
                             lb=lower_bound,
                             ub=upper_bound,
                             hyperparams=mh_hyperparams)

    # Keeping the top_k models. More than one model can be selected from each metaheuristic iteration
    top_indices, top_fitness = utils.get_top_models(scoreboard, exec_params.top_k)
    for ti, tf in zip(top_indices, top_fitness):
        print(f'{ti:<5} {tf:5}')

    print('Predicting on validation and test sets for ensemble learning')
    print('THERE IS STILL WORK PENDING. DO NOT KILL THIS PROCESS')

    # Predicting on validation
    best_models = utils.load_models(f'./trained/{exec_params.ds_name}_{exec_params.mh_name}', top_indices)
    for idx, model in zip(top_indices, best_models):
        utils.predict_persist(model, val_loader, device,
                              f'predictions/{exec_params.ds_name}_{exec_params.mh_name}_{idx}.txt')

    # Predicting on test
    for idx, model in zip(top_indices, best_models):
        utils.predict_persist(model, tst_loader, device,
                              f'predictions/{exec_params.ds_name}_{exec_params.mh_name}_{idx}_tst.txt')

    # Persisting labels for ensemble training / testing
    utils.store_labels(val_loader, f'./predictions/{exec_params.ds_name}_{exec_params.mh_name}_labels.txt')
    utils.store_labels(tst_loader, f'./predictions/{exec_params.ds_name}_{exec_params.mh_name}_labels_tst.txt')

    print('Done.')
