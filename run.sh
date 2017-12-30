#!/bin/bash

# run.sh

# --
# Look at different versions
# There are serial, parallel and CUDA versions

python test.py

# --
# Very small (synthetic) example to understand what's going on

python simple-example.py --plot

# --
# Run in parallel on a larger (synthetic) graph

python parallel-example.py --n-jobs 32 --n-nodes 3200