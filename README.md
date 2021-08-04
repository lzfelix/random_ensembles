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
   stored in `../logs/results/mnist/`. Further, it's possible to compute averages for test accuracy,
   test loss and training times with
   `python parser/parse_baseline_output_logs.py results/mnist/default/`.



## 2. Training models with hyperparameters lerned via PSO/BH (weak learners)

Before running this experiments, notice that:

- This step will store the top k individual network predictions for training and test set at `experiments/predictions/{dataset-name}_{metaheuristic-name}_{n-agents}_{agent-id}.txt`. Test set predictions terminate with `_tst` suffix. These files are necessary to run the ensemble experiments. Other two files will be generated containing validation and test sets ground truths;
- Be careful to not let multiple experiment runs override previous predictions in this folder!
- With that in mind:
   1. Run `experiments/hyper_learner.py` to fine-tune individual neural networks hyperparameters. As before, the `-h` flag shows instructions on how to set parameters to match the setup described in the paper. Don't forget to store the produced logs to extract results, which are used to compute fine tuned models accuracy;
   2. Just don't forget to add the `--show_test` flag to the experiment, otherwise test metrics won't be displayed in the final logs;
   3. This script should be executed only once, since it will generate 15 models. In the final lines of the logs it's possible to retrieve a list of accuracies to compute the mean and standard error.
