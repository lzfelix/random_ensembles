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
The following scripts are available to run the experiments. For more details on usage, please run
any of them with the `-h` flag
 - `hyper_learner.py` Learns model hyperparameters;
 - `runner.py` Runs the baseline models with their default hyperparameters;
 - `ensemble_learner.py` Learns weighted ensembles using some metaheuristic;
 - `ensemble_baseline.py` Computes 1/k and majority voting-based ensembles.

# Setup
   - Requires `Python==3.7` (using other python versions may lead to conflicts when installing `pytorch`)
   - Run `pip install -r requirements.txt`

# Reproducing results
The guide to reproduce results reported in the manuscript is divided in two parts: first we describe how to
train weak learners (ie: individual neural networks) and reproduce their results. Next, it's described how
to combine such outputs on ensemble and compute their metrics.


## Part I: Weak learners

### 1. Training models with default hyperparameters
1. Run `experiments/runner.py`. The `-h` flag shows instructions on how to run experiments.
    It's important to save all output logs for results extraction. For instance, it's possible
    to run MNIST network training and save its logs with
    `python runner.py mnist -n_epochs 3 > ../logs/results/mnist/default/run_1 2>&1 &`;
2. Such a procedure must be repeated 15 times to compute the statistical tests from the paper;
3. Next, move all logs to a single folder, for instance, `./logs/results/mnist/default/`.  This will make
  it possible to compute the averages for test set accuracy, tset set loss and training time with
  `python parser/parse_baseline_output_logs.py results/mnist/default/`.


### 2. Training models with hyperparameters fine-tuned via PSO and BH
Before running this experiments, notice that:

- This step will store the top `K` individual network predictions for training and test set at `experiments/predictions/{dataset-name}_{metaheuristic-name}_{n-agents}_{agent-id}.txt`. Test set predictions terminate with `_tst` suffix. These files are necessary to run the ensemble experiments in Part 2. Additionally, other two files will be generated holding validation and test sets ground truth labels;
- Be careful to not let multiple experiment runs override previous predictions in this folder!
- With that in mind:
   1. Run `experiments/hyper_learner.py` to fine-tune individual neural networks hyperparameters. As before, the `-h` flag shows instructions on how to set parameters to match the setup described in the paper. Don't forget to store the produced logs to extract results, such as model accuracy;
   2. Don't forget to add the `--show_test` flag when running the experiment, otherwise test metrics won't be displayed in the final logs;
   3. This script should be executed only once, since it will generate 15 models. In the final lines of the logs it's possible to retrieve a list of accuracies to compute the mean and standard error.


### 3. Training models with random hyperparameters
1. To train models with random hyperparameters run `python random_learner.py` and store the logs in an appropriate location. Notice this script can train multiple neural networks at once;
2. All models predictions will be stored in `experiments/predictions/{dataset-name}_random_{model-id}.txt` along with two additional files with ground truth labels for the validation and test sets, respectivelly;
3. Once again, there's no parser for this experiment since the logs will contain the relevant results in its last rows.


## Part II: Ensembles

To learn an ensemble it's necessary to first train a set of `K` weak learners to be combined. Recall that after training a set of weak learners some text files are generated under `./prediction/` folder. These files contian the outputs of each model in the validation and test sets. Ensemble learning is based on the validation set and final metrics are computed on the test set.

### 1. Optimized weights ensembles
1. Run `python experiments/ensemble_learner.py {path-to-validation-ground-truth-labels} {path-to-test-ground-truth-labels} -val_preds L --show_tests`. Running the script with the `-h` flag shows extra details on how to use it. In this case, `L` stands for a list of `K` networks outputs in the validation set. So, to train a ensemble based on three models, `L = ./predictions/mnist_bh_3_i.txt ./predictions/mnist_bh_3_j.txt ./predictions/mnist_bh_3_k.txt`, where `i`, `j`, `k` are weak learners' ids. The script `experiments/ensemble_learner.sh` also shows how the Python script can be invoked with multiple models;
2. This process is usually pretty fast and it's possible to find in the end of the logs:
  - The weight given to each model as a Python list
  - Each model individual accuracy on valiadtion and test set
  - The ensemble performance on the validation and test sets
  - How long it took to train the ensemble
3. This process must be repeated 15 times to compute the mean accuracy and standard deviation reported in the paper.
4. Just to recap: this step can be performed with the fine-tuned models (from Part I, step 2) and with models trained with random hyperparameters (from Part I, step 3). By doing and varying `K=[5, 10, 15]` it's possible to reproduce the results for all columns in row `Optimized` from Table 3 in the manuscript.

### 2. Majority-vote ensembles
1. This procedure is similar as the one in Step 1, but the script is different. Use `python experiments/ensemble_baseline.py [majority|uniform] {path-to-validation-ground-truth-labels} {path-to-test-ground-truth-labels} -val_preds L --show_tests`. The difference is that now it's necessary to supply the ensemling strategy to compute metrics, which is either `majority` (for this kind ensemble) or `uniform` for the ensembles from Part II, step 3;
2. Once again: all models supplied in the list `L` will be used to compute the ensemble;
3. Since this strategy is deterministic, there's no need to repeat it multiple times, since given the same models the output will always be the same, thus yielding zero stddev and mean=single observation value.

### 3. 1/K ensembles
1. Please refer to instructions in Part II, step 2.
