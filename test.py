#!/usr/bin/env python

"""
    test.py
"""

from __future__ import division, print_function

import pygsp
import numpy as np
import networkx as nx
from time import time

import heat
from helpers import featurize, par_graphwave

# --
# Create graph

np.random.seed(123)

num_nodes = 100
W = nx.adjacency_matrix(nx.gnp_random_graph(num_nodes, 0.1, seed=123 + 1))
W.eliminate_zeros()

taus = [0.5, 0.6, 0.7]
s = np.eye(num_nodes)


# --
# Simple

heat_kernel = heat.Heat(W=W, taus=taus)
heat_print = heat_kernel.filter(s)
feats = featurize(heat_print)

# --
# Parallel test

par_feats = par_graphwave(heat_kernel, n_chunks=2, n_jobs=2, verbose=10)

assert np.allclose(feats, par_feats)

# --
# Cupy test

cupy_heat_kernel = heat.CupyHeat(W=W, lmax=heat_kernel.lmax, taus=taus)
cupy_heat_print = cupy_heat_kernel.filter(s)
cupy_feats = featurize(cupy_heat_print)

assert np.allclose(heat_print, cupy_heat_print)
assert np.allclose(feats, cupy_feats)
assert np.allclose(par_feats, cupy_feats)

