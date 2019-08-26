import argparse

from torch import optim
from torch.nn import functional as F
from flare import trainer

from models import utils
from models import model_specs
from datasets import specs


def get_exec_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage='Runs a model with its default hyperparameters')
    parser.add_argument('ds_name', help='Dataset name, also selects the neural net')
    parser.add_argument('-n_epochs', help='For how long to train the model', default=15, type=int)
    parser.add_argument('-batch_sz', help='Batch size', default=128, type=int)
    parser.add_argument('-trn_split', help='Fraction of train data used for training', default=0.8, type=float)
    parser.add_argument('-lr', help='SGD learning rate', default=1.0, type=float)
    parser.add_argument('-momentum', help='SGD momentum', default=0, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    exec_params = get_exec_params()
    device, pin_memory = utils.get_device(no_gpu=False)

    ds_specs = specs.get_specs(exec_params.ds_name)
    trn_gen, val_gen, tst_gen = ds_specs.loading_fn(exec_params.batch_sz, exec_params.trn_split, pin_memory)

    experiment_setup = model_specs.get_experiment_setup(exec_params.ds_name)
    model = experiment_setup.net().to(device)

    if exec_params.ds_name == 'kmnist':
        opt = optim.Adadelta(model.parameters(), lr=exec_params.lr)
    else:
        opt = optim.SGD(model.parameters(), lr=exec_params.lr, momentum=exec_params.momentum)

    trainer.train_on_loader(model, trn_gen, val_gen, F.nll_loss, opt,
                            n_epochs=exec_params.n_epochs, batch_first=True, device=device)
