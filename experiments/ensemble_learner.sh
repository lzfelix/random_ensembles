#!/bin/sh

# Initial variables
DATASET=mnist
N_MODELS=10

# Looping for every possible model
for (( i=0; i<$N_MODELS; i++ ))
do
    # Concatenating the string
    VAL_PREDS=""$VAL_PREDS"predictions/"$DATASET"_random_"$i".txt "
done

# Running the actual ensemble learner
python ensemble_learner.py predictions/"$DATASET"_random_labels.txt predictions/"$DATASET"_random_labels_tst.txt -val_preds $VAL_PREDS > mnist_random_ensemble.txt 2>&1 &
