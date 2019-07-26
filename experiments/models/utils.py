from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch import nn

from opytimizer import Opytimizer
from opytimizer.utils.history import History
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.spaces.search import SearchSpace


def optimize(metaheuristic: Optimizer,
             target: callable,
             n_agents: int,
             n_variables: int,
             n_iterations: int,
             lb: List[float],
             ub: List[float],
             hyperparams: Dict[str, Any]) -> History:
    """Abstract all the Opytimizer mechanics into a single method."""

    space = SearchSpace(n_agents, n_variables, n_iterations, lb, ub)
    optimizer = metaheuristic(hyperparams=hyperparams)

    task = Opytimizer(space, optimizer, Function(target))
    return task.start()


def get_top_models(history: History,
                   n_agents: int) -> List[Tuple[int, int, list]]:
    """Gets all the best models found during optimization."""

    best_indices = list()
    for iter_no, iteration_best in enumerate(history.best_agent):
        # Finding the actual index
        position, fitness, index = iteration_best
        cumulative_index = index + iter_no * n_agents

        # Opytimizer discards the current iteration best if it is
        # worse than the previous last solution. Check for that so
        # each model is selected only once.
        # Despite the name, history.agents contains all the agents
        # values for the i-th iteration
        actual_fitness = history.agents[iter_no][index][-1]
        if np.isclose(actual_fitness, fitness) or iter_no == 0:
            best_indices.append((index, cumulative_index, fitness))

    return best_indices


def load_models(models_home: str,
                meta_prefix: str,
                model_indices: List[int]) -> List[nn.Module]:
    """Loads all models

    # Arguments
        models_home: The root folder of all models.
        meta_prefix: All model filenames must start with this
            prefix, for instance
            `models_home/meta_prefix_[model_index].txt`
        model_indices: Index of the models to be loaded.

    # Returns
        List of the loaded models.
    """
    mask = meta_prefix + '_{}.pth'
    models = list()
    for index in model_indices:
        filepath = Path(models_home, mask.format(index))
        models.append(torch.load(filepath))
    return models


def predict_on_val(model: nn.Module,
                   x_val: torch.Tensor,
                   y_val: torch.LongTensor,
                   destination: str) -> None:
    """Computes the model predictions (outputs) for x_val.

    # Arguments
        model: PyTorch model.
        x_val: Features tensor with shape `[n_samples, *]`.
        y_val: Labels tensor with shape `[n_samples]`.
        destination: Path of the text file with the computed
            predictions.

    # File layout:
            n_samples, val_accuracy
            logit_{00} logit_{01} ... logit_{0n}
            logit_{10} logit_{11} ... logit_{1n}
            ...
            logit_{m0} logit_{m1} ... logit_{mn}
    """
    logits = model(x_val)
    acc = ((logits.argmax(-1) == y_val).sum().float() / y_val.numel()).item()

    def tensor2str(tensor):
        return ' '.join(map(str, tensor.tolist()))

    with open(destination, 'w') as dfile:
        dfile.write('{} {}\n'.format(y_val.numel(), acc))
        for logit in logits:
            dfile.write('{}\n'.format(tensor2str(logit)))


def load_predictions(filepath: str) -> np.ndarray:
    """Loads predictions from the filepath.

    # File layout:
        n_samples, val_accuracy
        logit_{00} logit_{01} ... logit_{0n}
        logit_{10} logit_{11} ... logit_{1n}
        ...
        logit_{m0} logit_{m1} ... logit_{mn}
    """
    all_preds = list()
    for lineno, line in enumerate(open(filepath, 'r')):
        if lineno > 0:
            probabilities = list(map(float, line.split()))
            all_preds.append(probabilities)
    return np.asarray(all_preds)


def store_labels(y_val: torch.LongTensor,
                      destination: str) -> None:
    """Store the ground truth labels for the validation set.

    # Arguments
        y_val: Labels tensor with shape `[n_samples]`.
        destination: Path of the text file with the computed
            predictions.

    # File layout:
        n_samples
        y_0
        y_1
        ...
        y_n
    """
    with open(destination, 'w') as dfile:
        dfile.write(f'{y_val.numel()}\n')
        for label in y_val:
            dfile.write(f'{label.item()}\n')


def load_labels(filepath: str) -> np.ndarray:
    """Loads ground truth labels from the filepath."""
    labels = list()
    for lineno, line in enumerate(open(filepath, 'r')):
        if lineno > 0:
            labels.append(int(line.strip()))
    return np.asarray(labels, dtype=np.int32)


def accuracy(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Computes y_pred accuracy."""
    return (y_pred.argmax(-1) == y).sum() / y.size
