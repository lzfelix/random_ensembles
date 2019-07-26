import numpy as np

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

    # File layout:
        n_samples
        y_0
        y_1
        ...
        y_n
    """

    def fun(w: np.ndarray) -> float:
        w = np.asarray(w).flatten()
        w /= w.sum()
        n_candidates = w.shape[0]

        y_combined = np.zeros_like(y_pred[0])
        for i in range(n_candidates):
            y_combined += y_pred[i] * w[i]
        y_hat = y_combined.argmax(-1)

        acc = (y_hat == y_true).sum() / y_true.shape[0]
        return 1 - acc
    return fun
