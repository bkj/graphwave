#!/bin/bash

# run.sh

# !! This assumes undirected graphs

# --
# There are serial, parallel and CUDA versions -- can see them used here

python test.py

# --
# Very small (synthetic) example to understand what's going on

python simple-example.py --plot

# --
# Run in parallel on a larger (synthetic) graph

python parallel-example.py --n-jobs 32 --n-nodes 25600
python parallel-example.py --n-jobs 64 --n-nodes 51200
python parallel-example.py --n-jobs 128 --n-nodes 102400
python parallel-example.py --n-jobs 256 --n-nodes 205600
