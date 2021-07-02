from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch import nn

from opytimizer import Opytimizer
from opytimizer.utils.history import History
from opytimizer.core import Function, Optimizer
from opytimizer.spaces import SearchSpace

from flare import trainer


def optimize(metaheuristic: Optimizer,
             target: callable,
             n_agents: int,
             n_variables: int,
             n_iterations: int,
             lb: List[float],
             ub: List[float],
             hyperparams: Dict[str, Any]) -> History:
    """Abstract all the Opytimizer mechanics into a single method."""

    space = SearchSpace(n_agents, n_variables, lb, ub)
    optimizer = metaheuristic(hyperparams)

    task = Opytimizer(space, optimizer, Function(target))

    task.start(n_iterations)

    return task.history


def get_top_models(scoreboard: Dict[int, int],
                   top_k: int) -> List[Tuple[int]]:
    """Gets the `min(top_k, len(scoreboard))` best models with different fitness values.

    # Arguments
        scoreboard: A dict mapping callno to fitness value.
        top_k:
    # Return
        A: List of model indices
        B: List of fitness values.
    """

    sorted_scores = sorted(scoreboard.items(), key=lambda x: x[1])
    selected = [sorted_scores[0]]
    i = 1
    while i < len(sorted_scores) and len(selected) < top_k:
        if sorted_scores[i][1] != selected[-1][1]:
            selected.append(sorted_scores[i])
        i += 1
    return list(zip(*selected))


def load_models(models_prefix: str,
                model_indices: List[int]) -> List[nn.Module]:
    """Loads all models

    # Arguments
        models_prefix: All model filenames must start with this
            prefix, for instance `models_home/meta_prefix_[model_index].txt`
        model_indices: Index of the models to be loaded.

    # Return
        List of yTorch models.
    """
    models = list()
    for index in model_indices:
        filepath = f'{models_prefix}_{index}.pth'
        models.append(torch.load(filepath))
    return models


def predict_persist(model,
                    eval_gen,
                    device,
                    destination: str) -> float:
    """Computes the model predictions and stores them in a text file.

    # Arguments
        model: PyTorch model.
        eval_gen:
        device:
        destination: Path of the text file with the computed predictions.

    # Return
        The model accuracy on the provided generator.

    # File layout:
            n_samples, val_accuracy
            logsoftmax_{00} logsoftmax_{01} ... logsoftmax_{0n}
            logsoftmax_{10} logsoftmax_{11} ... logsoftmax_{1n}
            ...
            logsoftmax_{m0} logsoftmax_{m1} ... logsoftmax_{mn}
    """

    logits = trainer.predict_on_loader(model, eval_gen, device, verbosity=0)
    y_val = [batch[1] for batch in eval_gen]
    y_val = torch.cat(y_val).to(device)

    # max(p(x)) == max(log(p(x)))
    acc = ((logits.argmax(-1) == y_val).sum().float() / y_val.numel()).item()

    def tensor2str(tensor):
        return ' '.join(map(str, tensor.tolist()))

    with open(destination, 'w') as dfile:
        dfile.write('{} {}\n'.format(y_val.numel(), acc))
        for logit in logits:
            dfile.write('{}\n'.format(tensor2str(logit)))
    return acc


def load_predictions(filepath: str) -> np.ndarray:
    """Loads predictions from the filepath. Recall that the returned values
    are `log(softmax(z)) = log(p(y_i | x_j))`.

    # File layout:
        n_samples, val_accuracy
        logsoftmax_{00} logsoftmax_{01} ... logsoftmax_{0n}
        logsoftmax_{10} logsoftmax_{11} ... logsoftmax_{1n}
        ...
        logsoftmax_{m0} logsoftmax_{m1} ... logsoftmax_{mn}
    """
    all_preds = list()
    for lineno, line in enumerate(open(filepath, 'r')):
        if lineno > 0:
            probabilities = list(map(float, line.split()))
            all_preds.append(probabilities)
    return np.asarray(all_preds)


def store_labels(eval_gen,
                 destination: str) -> None:
    """Store the ground truth labels for the validation set.

    # Arguments
        eval_gen:
        destination: Path of the text file with the computed
            predictions.

    # File layout:
        n_samples
        y_0
        y_1
        ...
        y_n
    """
    y_val = torch.cat([batch[1] for batch in eval_gen])
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


def accuracy(y_pred: np.ndarray,
             y: np.ndarray) -> float:
    """Computes y_pred accuracy."""
    return (y_pred.argmax(-1) == y).sum() / y.size
