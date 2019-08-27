from opytimizer.optimizers import bha
from opytimizer.optimizers import fa
from opytimizer.core.optimizer import Optimizer


class MHSpecs:
    def __init__(self, mh_method: Optimizer, hyperparams: dict) -> None:
        self.mh_method = mh_method
        self.hyperparams = hyperparams

    def __repr__(self):
        return str(self.__dict__)


_specs = dict(
    fa=MHSpecs(fa.FA, dict(alpha=0.5, beta=0.2, gamma=1.0)),
    bha=MHSpecs(bha.BHA, dict())
)


def get_specs(mh_name: str) -> MHSpecs:
    try:
        return _specs[mh_name]
    except:
        print(f'Metaheuristic "{mh_name}" specs not specified in mh/specs.py')
        exit(-1)