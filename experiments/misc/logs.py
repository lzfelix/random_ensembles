import json
from typing import List, Sized

import numpy as np


def print_hyperparams(h_names: List[str], h_values: Sized) -> None:
    n_model_hyperparams = len(h_names)
    model_hyperpar_values = h_values[:n_model_hyperparams]
    opt_hyperpar_values = h_values[n_model_hyperparams:]

    o_names = ['lr', 'momentum']
    assert len(o_names) == len(opt_hyperpar_values)

    table = dict()
    for hn, hv in zip(h_names, model_hyperpar_values):
        table[hn] = hv

    for on, ov in zip(o_names, opt_hyperpar_values):
        table[on] = ov
    print(json.dumps(table, indent=4))
