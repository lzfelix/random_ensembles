#!/bin/sh

# Initial variables
DATASET=mnist
N_RUNS=15
N_EPOCHS=15

# Iterates through all possible seeds
for RUN in $(seq 1 $N_RUNS); do
    # Running the actual ensemble learner
    python runner.py ${DATASET} -n_epochs ${N_EPOCHS} > "${DATASET}"_default_"${RUN}".txt 2>&1 &
done

mkdir -p ../results/${DATASET}/default/; mv "${DATASET}"_default_*.txt ../results/${DATASET}/default/