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

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from rsub import *
from matplotlib import pyplot as plt

from utils.build_graph import build_regular_structure

np.random.seed(123)

# --
# Helpers

def characteristic_function(s, t=np.arange(0, 100, 2)):
    return (np.exp(np.complex(0, 1) * s) ** t.reshape(-1, 1)).mean(axis=1)

# --
# Create graph

G, colors = build_regular_structure(
    width_basis=15,
    basis_type="star",
    nb_shapes=5,
    shape=["house"],
    start=0,
    add_random_edges=0
)

adj = nx.adjacency_matrix(G)

taus = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

# Create kernel
pG = pygsp.graphs.Graph(adj, lap_type='normalized')
pG.estimate_lmax()
heat_kernel = pygsp.filters.Heat(pG, taus, normalize=False)

# Apply kernel at every node
heat_print = heat_kernel.analyze(np.eye(pG.N)).transpose((2, 0, 1))

# Create features
feats = []
for i, sig in enumerate(heat_print):
    sig_feats = []
    for node_sig in sig:
        node_feats = characteristic_function(node_sig)
        node_feats = np.column_stack([node_feats.real, node_feats.imag]).reshape(-1)
        sig_feats.append(node_feats)
    
    feats.append(np.vstack(sig_feats))

feats = np.hstack(feats)

feats.sum()

# --
# Cluster resulting features

nfeats = feats - feats.mean(axis=0, keepdims=True)
nfeats /= (1e-10 + nfeats.std(axis=0, keepdims=True))
nfeats[np.isnan(nfeats)] = 0

pca_feats = PCA(n_components=10).fit_transform(nfeats)

clus = KMeans(n_clusters=len(set(colors))).fit(pca_feats).labels_

jitter_pca_feats = pca_feats + np.random.uniform(0, 1, pca_feats.shape)

_ = plt.scatter(jitter_pca_feats[:,0], jitter_pca_feats[:,1], alpha=0.25, c=clus, cmap='rainbow')
show_plot()

np.random.seed(1235)
_ = nx.draw(G, pos=nx.spring_layout(G, iterations=200), 
    node_color=clus, node_size=50, cmap='rainbow', ax=plt.figure().add_subplot(111))

show_plot()
