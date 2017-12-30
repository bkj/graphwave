#!/bin/bash

# run.sh

# --
# Very small (synthetic) example to understand what's going on

python simple-example.py

# --
# Run in parallel on a larger (synthetic) graph

python parallel-example.py --n-jobs 32 --n-nodes 3200

# --
# Run in parallel on a real graph