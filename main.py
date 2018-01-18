#!/usr/bin/env python

"""
    parallel-example.py
"""

from __future__ import division, print_function

import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from time import time
from scipy import sparse

import heat
from helpers import par_graphwave


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, required=True)
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
    
    edgelist = np.array(pd.read_csv(args.inpath, sep='\t', header=None, dtype=int))
    W = sparse.csr_matrix((np.arange(edgelist.shape[0]), (edgelist[:,0], edgelist[:,1])))
    
    assert (np.sum(W, axis=0) > 0).all(), 'cannot have empty rows'
    assert (np.sum(W, axis=1) > 0).all(), 'cannot have empty columns'
    
    taus = map(float, args.taus.split(','))
    
    print("parallel-example.py: running on graph w/ %d edges" % W.nnz, file=sys.stderr)
    t = time()
    hk = heat.Heat(W=W, taus=taus)
    pfeats = par_graphwave(hk, n_chunks=args.n_chunks, n_jobs=args.n_jobs, verbose=10)
    run_time = time() - t
    
    print("parallel-example.py: took %f seconds" % run_time, file=sys.stderr)
