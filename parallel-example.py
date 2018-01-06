#!/usr/bin/env python

"""
    parallel-example.py
"""

from __future__ import division, print_function

import sys
import argparse
import numpy as np
import networkx as nx
from time import time

import heat
from helpers import par_graphwave


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-nodes', type=int, default=3200)
    parser.add_argument('--p', type=float, default=0.01)
    
    parser.add_argument('--taus', type=str, default="0.5")
    
    parser.add_argument('--n-chunks', type=int, default=32)
    parser.add_argument('--n-jobs', type=int, default=32)
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()


# --
# Run

if __name__ == "__main__":
    args = parse_args()
    
    np.random.seed(args.seed)
    
    print("parallel-example.py: creating graph", file=sys.stderr)
    W = nx.adjacency_matrix(nx.gnp_random_graph(args.n_nodes, args.p, seed=args.seed + 1))
    keep = np.asarray(W.sum(axis=0) > 0).squeeze()
    W = W[keep]
    W = W[:,keep]
    W.eliminate_zeros()
    
    taus = map(float, args.taus.split(','))
    
    print("parallel-example.py: running on graph w/ %d edges" % W.nnz, file=sys.stderr)
    t = time()
    hk = heat.Heat(W=W, taus=taus)
    pfeats = par_graphwave(hk, n_chunks=args.n_chunks, n_jobs=args.n_jobs, verbose=10)
    run_time = time() - t
    
    print("parallel-example.py: took %f seconds" % run_time, file=sys.stderr)
