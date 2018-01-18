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
    --outpath ./_results/synthetic/3200


# Compute features for sample of nodes in larger graph (25 seconds)
python main.py --n-jobs 32 \
    --inpath ./_data/synthetic/51200.edgelist \
    --outpath ./_results/synthetic/51200 \
    --num-queries 512

# --
# Run on POKEC graph

mkdir -p {_data,_results}/pokec
wget --header "Authorization:$TOKEN" https://hiveprogram.com/data/_v0/generic/pokec.edgelist.gz
gunzip pokec.edgelist.gz && mv pokec.edgelist ./_data/pokec

python main.py --n-jobs 32 \
    --inpath ./_data/pokec/pokec.edgelist \
    --outpath ./_results/pokec/pokec \
    --num-queries 64 