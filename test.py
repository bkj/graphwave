#!/usr/bin/env python

"""
    example.py
"""

from __future__ import division, print_function


import os
import sys
import pygsp
import numpy as np
import networkx as nx
from joblib import Parallel, delayed

from simple import SimpleHeat
from helpers import featurize, delayed_featurize, characteristic_function

# --
# Create graph

np.random.seed(123)
W = nx.adjacency_matrix(nx.gnp_random_graph(100, 0.1))
W.eliminate_zeros()

taus = [0.5, 0.6, 0.7]
s = np.eye(100)

# --
# Run original

G = pygsp.graphs.Graph(W, lap_type='normalized')
G.estimate_lmax()


# New
heat_kernel = pygsp.filters.Heat(G, taus, normalize=False)
heat_print = heat_kernel.analyze(s).transpose((2, 0, 1))
feats = featurize(heat_print)

# --
# Simple

heat_kernel2 = SimpleHeat(W=W, lmax=G.lmax, taus=taus)
heat_print2 = heat_kernel2.filter(s)
feats2 = featurize(heat_print2)

assert (heat_print == heat_print2).all()
assert (feats == feats2).all()

# >>

from simple import _filter

def par_featurize(hk, num_chunks=10):
    assert hk.num_nodes % num_chunks == 0
    
    global _runner
    def _runner(chunk):
        return delayed_featurize(_filter(hk.L, hk.num_nodes, hk.lmax, hk.taus, chunk))
    
    chunks = np.array_split(np.eye(hk.num_nodes), num_chunks, axis=1)
    
    jobs = [delayed(_runner)(chunk) for chunk in chunks]
    tmp = np.array(Parallel(n_jobs=10)(jobs)).mean(axis=0)
    
    pfeats = np.empty((tmp.shape[0], tmp.shape[1] * 2))
    pfeats[:,0::2] = tmp.real
    pfeats[:,1::2] = tmp.imag
    
    return pfeats

hk = SimpleHeat(W=W, lmax=G.lmax, taus=taus)
pfeats = par_featurize(hk)
assert np.allclose(feats, pfeats)

