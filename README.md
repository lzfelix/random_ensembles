# Random Ensembles
To reproduce a smaller version of the code and get an overall idea, please run
`hyperparam_learning.ipynb` and then `ensemble_learning.ipynb`.


# Table of Contents
- [Random Ensembles](#random-ensembles)
- [Scripts](#scripts)
- [Setup](#setup)
- [Reproducing results](#reproducing-results)
  * [Part I: Weak learners](#part-i--weak-learners)
    + [1. Training models with default hyperparameters](#1-training-models-with-default-hyperparameters)
    + [2. Training models with hyperparameters fine-tuned via PSO and BH](#2-training-models-with-hyperparameters-fine-tuned-via-pso-and-bh)
    + [3. Training models with random hyperparameters](#3-training-models-with-random-hyperparameters)
  * [Part II: Ensembles](#part-ii--ensembles)
    + [1. Optimized weights ensembles](#1-optimized-weights-ensembles)
    + [2. Majority-vote ensembles](#2-majority-vote-ensembles)
    + [3. 1/K ensembles](#3-1k-ensembles)


# Scripts
The following scripts are available to run the experiments. Running
either of them with the `-h` flag shows their running options. 

   - `hyper_learner.py` Learns model hyperparameters;
   - `ensemble_learner.py` Learns weighted ensembles using some metaheuristic;
   - `runner.py` Runs the baseline models with their default hyperparameters

# Setup
   - Requires `Python==3.7`
   - Run `pip install -r requirements.txt`


# Reproducing results

Results reproduction is divided in two parts: first we describe how to reproduce weak learners
(ie: individual models) results and then how to combine such models outputs into ensemble results


## Part I: Weak learners

### 1. Training models with default hyperparameters
1. Run `experiments/runner.py`. The `-h` flag shows instructions on how to run experiments.
    It's important to save all output logs for results extraction. For instance, it's possible
    to run MNIST network training and save its logs with
    `python runner.py mnist -n_epochs 3 > ../logs/results/mnist/default/run_1 2>&1 &`;
2. Such a procedure must be repeated 15 times to compute the statistical tests from the paper;
3. Next, move all logs to a single folder, in step 1 that is already done, so all logs are
   stored in `../logs/results/mnist/`. Further, it's possible to compute averages for test accuracy,
   test loss and training times with
   `python parser/parse_baseline_output_logs.py results/mnist/default/`.



### 2. Training models with hyperparameters fine-tuned via PSO and BH
Before running this experiments, notice that:

- This step will store the top `K` individual network predictions for training and test set at `experiments/predictions/{dataset-name}_{metaheuristic-name}_{n-agents}_{agent-id}.txt`. Test set predictions terminate with `_tst` suffix. These files are necessary to run the ensemble experiments. Other two files will be generated containing validation and test sets ground truths;
- Be careful to not let multiple experiment runs override previous predictions in this folder!
- With that in mind:
   1. Run `experiments/hyper_learner.py` to fine-tune individual neural networks hyperparameters. As before, the `-h` flag shows instructions on how to set parameters to match the setup described in the paper. Don't forget to store the produced logs to extract results, which are used to compute fine tuned models accuracy;
   2. Just don't forget to add the `--show_test` flag to the experiment, otherwise test metrics won't be displayed in the final logs;
   3. This script should be executed only once, since it will generate 15 models. In the final lines of the logs it's possible to retrieve a list of accuracies to compute the mean and standard error.


### 3. Training models with random hyperparameters
1. To train models with random hyperparameters run `python random_learner.py` and store the logs in an appropriate location;
2. Predictions will be stored in `experiments/predictions/{dataset-name}_random_{model-id}.txt` along with two additional files with ground truth labels for the validation and test sets, respectivelly;
3. Once again, there's no parser for this experiment since the logs will contain the relevant results in its last rows.


## Part II: Ensembles

To learn an ensemble it's necessary to first train a set of K weak learners, where K stands for the number of models to be combined. Recall that after training a set of weak learners some text files are generated under `./prediction/` folder with the outputs of each models to the validation and test sets. Ensemble learning is based on the validation set and final metrics are computed on the test set.

### 1. Optimized weights ensembles
1. Simply run `python experiments/ensemble_learner.py {path-to-validation-ground-truth-labels} {path-to-test-ground-truth-labels} -val_preds L --show_tests`. Running the script with the `-h` flag shows extra details on how to use it. In this case, `L` stands for a list of `K` networks outputs in the validation set. So, to train a ensemble based on three models, `L = ./predictions/mnist_bh_3_i.txt ./predictions/mnist_bh_3_j.txt ./predictions/mnist_bh_3_k.txt`, where `i`, `j`, `k` are weak learners' ids. The script `experiments/ensemble_learner.sh` also shows how the Python script can be invoked with multiple models;
2. This process is usually pretty fast and it's possible to find in the end of the logs:
  - The weight given to each model as a Python list
  - Each model individual accuracy on valiadtion and test set
  - The ensemble performance on the validation and test sets
  - How long it took to train the ensemble
3. This process must be repeated 15 times to compute the mean accuracy and standard deviation reported in the paper.
4. Just to recap: this step can be performed with the fine-tuned models (from Part I, step 2) and with models trained with random hyperparameters (from Part I, step 3). By doing and varying `K=[5, 10, 15]` it's possible to reproduce the results for all columns in row `Optimized` from Table 3 in the manuscript.

### 2. Majority-vote ensembles
TODO

### 3. 1/K ensembles
TODO
