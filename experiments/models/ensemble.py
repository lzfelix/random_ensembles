from typing import List, Tuple

import numpy as np
from misc import utils


def make_combination(y_pred: np.ndarray,
                     y_true: np.ndarray) -> callable:
    """Creates a target function to be minimized with the format
        `z = 1 - accuracy([w_i * M_i(x)], y_true)`

    # Arguments
        y_pred: ndarray with logits for each class with shape
            `[n_candidates, n_samples, n_classes]`
        y_true: ndarray with true labels and shape `[n_samples]`

    # Return
        A function.
    """

    def fun(w_: list) -> float:
        # Ensuring that the sum of the candidate weights is one
        w = np.asarray(w_).flatten()
        w = w / w.sum()
        n_candidates = w.shape[0]

        y_combined = np.zeros_like(y_pred[0])
        for i in range(n_candidates):
            y_combined += y_pred[i] * w[i]
        y_hat = y_combined.argmax(-1)

        acc = (y_hat == y_true).sum() / y_true.shape[0]
        return 1 - acc
    return fun


def evaluate_ensemble(w: np.ndarray,
                      y_pred: np.ndarray,
                      y_true: np.ndarray) -> float:
    """Multiplies each prediction by the model weight and compute its accuracy.

    # Arguments
        w: ndarray with the weight of the i-th model.
        y_pred: Tensor with shape `[n_candidates, n_samples, n_classes]`.
        y_true: Tensor with the ground-truth labels and shape `[n_samples, n_classes]`.

    # Return
        The ensemble accuracy.
    """
    w = w.flatten() / w.sum()
    n_candidates = w.shape[0]

    y_combined = np.zeros_like(y_pred[0])
    for i in range(n_candidates):
        y_combined += y_pred[i] * w[i]
    y_hat = y_combined.argmax(-1)

    acc = (y_hat == y_true).sum() / y_true.shape[0]
    return acc


def _validate_predictions(preds: np.ndarray,
                          ground: np.ndarray) -> None:
    n_samples = [x.shape[0] for x in preds]
    if not all(n_samples[0] == ns for ns in n_samples):
        raise RuntimeError('Not all candidates have evaluated the same amount of samples.')
    print(ground.shape[0], n_samples[0])
    if ground.shape[0] != n_samples[0]:
        raise RuntimeError('Amount of ground truth labels differs from the amount of samples.')


def load_candidates_preds(preds_path: List[str],
                          ground_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads cancidate predictions and ground truth from the disk."""
    y_true = utils.load_labels(ground_path)
    all_preds = [utils.load_predictions(preds) for preds in preds_path]

    # Combining a list of `(n_samples, n_classes)` tensors into a
    # `(n_candidates, n_samples, n_classes)` tensor
    all_preds = np.asarray(all_preds)

    _validate_predictions(all_preds, y_true)

    return all_preds, y_true
