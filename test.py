#!/usr/bin/env python

"""
    example.py
"""


import os
import sys
import pygsp
import numpy as np
import pandas as pd
import networkx as nx

from simple import SimpleHeat, featurize

# --
# Create graph

adj = nx.adjacency_matrix(nx.gnp_random_graph(100, 0.1))

taus = [0.5, 0.6]


G = pygsp.graphs.Graph(adj, lap_type='normalized')
G.estimate_lmax()

s = np.eye(G.N)[:,:20]

# New
heat_kernel = pygsp.filters.Heat(G, taus, normalize=False)
heat_print = heat_kernel.analyze(s).transpose((2, 0, 1))
feats = featurize(heat_print)

# --
# Simple

from scipy import sparse

W = sparse.csr_matrix(adj)
W.eliminate_zeros()
dw = np.asarray(W.sum(axis=1)).squeeze()
L = compute_laplacian(W, dw)
num_nodes = W.shape[0]

assert (G.W.toarray() == W.toarray()).all()
assert (G.dw == dw).all()
assert (G.L.toarray() == L.toarray()).all()


lmax = G.lmax

heat_kernel2 = SimpleHeat(L=L, num_nodes=num_nodes, lmax=lmax, taus=taus)
heat_print2 = heat_kernel2.filter(s)
feats2 = featurize(heat_print2)

assert (heat_print == heat_print2).all()
assert (feats == feats2).all()
