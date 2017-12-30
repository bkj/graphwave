#!/usr/bin/env python

"""
    test.py
    
    Test to verify that SimpleHeat and `par_graphwave` yield same results
    as original code
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
W = nx.adjacency_matrix(nx.gnp_random_graph(num_nodes, 0.1))
W.eliminate_zeros()

taus = [0.5, 0.6, 0.7]
s = np.eye(num_nodes)

G = pygsp.graphs.Graph(W, lap_type='normalized')
G.estimate_lmax()

# --
# Run original

heat_kernel = pygsp.filters.Heat(G, taus, normalize=False)
heat_print = heat_kernel.analyze(s).transpose((2, 0, 1))
feats = featurize(heat_print)

# --
# Simple

simple_heat_kernel = heat.Heat(W=W, lmax=G.lmax, taus=taus)
simple_heat_print = simple_heat_kernel.filter(s)
simple_feats = featurize(simple_heat_print)

assert (heat_print == simple_heat_print).all()
assert (feats == simple_feats).all()

# --
# Parallel test

par_feats = par_graphwave(simple_heat_kernel)
assert np.allclose(feats, par_feats)
assert np.allclose(simple_feats, par_feats)
