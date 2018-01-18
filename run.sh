#!/bin/bash

# run.sh

# --
# Run on synthetic graphs

python utils/make-graph.py --n-nodes 3200 --outpath ./_data/synthetic/3200.edgelist
python utils/make-graph.py --n-nodes 51200 --outpath ./_data/synthetic/51200.edgelist

python main.py --n-jobs 32 --inpath ./_data/synthetic/3200.edgelist # 2 seconds
python main.py --n-jobs 32 --inpath ./_data/synthetic/51200.edgelist # 2 seconds
