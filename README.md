# Random Ensembles

To reproduce a smaller version of the code and get an overall idea, please run
`hyperparam_learning.ipynb` and then `ensemble_learning.ipynb`.

# Scripts

The following scripts are available to run the experiments. Running
either of them with the `-h` flag shows their running options. 

   - `hyper_learner.py` Learns model hyperparameters;
   - `ensemble_learner.py` Learns weighted ensembles using some metaheuristic;
   - `runner.py` Runs the baseline models with their default hyperparameters

# Reproducing results

## Setup

- Requires `Python==3.7`
- Run `pip install -r requirements.txt`

## Training models with default hyperparameters

1. Run `experiments/runner.py`. The `-h` flag shows instructions on how to run experiments.
    It's important to save all output logs for results extraction. Such step can be achieved
    with `python runner.py mnist -n_epochs 1 > experiment_1.txt 2>&1 &`, for instance;
2. Move all experiment logs to a single folder, for instance `../results/cifar100/default/`;
3. Then run the parsing script to extract test accuracy, test lost and training time from 
    multiple experiments with `python parser/parse_baseline_output_logs.py results/cifar100/default/`.
