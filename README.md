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

- Each experiment should be executed 15 times to compute proper mean and standard deviation



## Setup

- Requires `Python==3.7`
- Run `pip install -r requirements.txt`



## 1. Training models with default hyperparameters

1. Run `experiments/runner.py`. The `-h` flag shows instructions on how to run experiments.
    It's important to save all output logs for results extraction. For instance, it's possible
    to run MNIST network training and save its logs with
    `python runner.py mnist -n_epochs 3 > ../logs/results/mnist/default/run_1 2>&1 &`;
2. Such a procedure must be repeated 15 times to compute the statistical tests from the paper;
3. Next, move all logs to a single folder, in step 1 that is already done, so all logs are
   stored in `../logs/results/mnist/`, so we can compute average test accuracy, test loss and training
   time with `python parser/parse_baseline_output_logs.py results/mnist/default/`.



## 2. Training models with hyperparameters lerned via PSO/BH (weak learners)

Before running this experiments, notice that:

> 1. This step will store individual network predictions for training and test set at `experiments/predictions/[dataset-name]\_[metaheuristic-name]\_[n-agents]_[keep-k-neural-nets].txt`. Test set predictions terminate with `_tst` suffix. These files are necessary to run the ensemble experiments. 
> 2. Be careful to not let multiple experiment runs override previous predictions in this folder!



1. Run `experiments/hyper_learner.py` to learn the good hyperparameters for each neural network individually. As before, the `-h` flag shows instructions on how to set parameters to match the setup described in the paper. Don't forget to store the produced logs to extract results, as aforementioned;
2. Move all experiment logs to a single folder, for instance `../results/cifar100/week_learners/5_agents/`
3. Then run the parsing script to extract test accuracy and optimization time from multiple experiments with `python parser/parse_hyper_learner_output_logs.py ../results/cifar100/week_learners/5_agents/`

