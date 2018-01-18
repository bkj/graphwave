#!/bin/bash

# run.sh

# --
# Run on synthetic graphs

mkdir -p {_data,_results}/synthetic

# Generate synthetic graphs
python utils/make-graph.py --n-nodes 3200 --outpath ./_data/synthetic/3200.edgelist
python utils/make-graph.py --n-nodes 51200 --outpath ./_data/synthetic/51200.edgelist


# Compute features for all nodes in small graph (2 seconds)
python main.py --n-jobs 32 \
    --inpath ./_data/synthetic/3200.edgelist \
    --outpath _results/synthetic/3200


# Compute features for sample of nodes in larger graph (25 seconds)
python main.py --n-jobs 32 \
    --inpath ./_data/synthetic/51200.edgelist \
    --outpath _results/synthetic/51200 \
    --pct-queries 0.01

# --
# Run on POKEC graph