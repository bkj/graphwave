#!/usr/bin/env python

"""
    simple-example.py
"""

import numpy as np
import networkx as nx

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt

from utils.build_graph import build_regular_structure

from simple import SimpleHeat
from helpers import featurize

np.random.seed(123)

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

W = nx.adjacency_matrix(G)
W.eliminate_zeros()
taus = [0.5, 0.6 , 0.7, 0.8, 0.9, 1.0, 1.1]

# Apply kernel at every node
signal = np.eye(W.shape[0])
heat_kernel = SimpleHeat(W=W, taus=taus)
heat_print = heat_kernel.filter(signal)
feats = featurize(heat_print)

# --
# Cluster resulting features

# Normalize features
nfeats = feats - feats.mean(axis=0, keepdims=True)
nfeats /= (1e-10 + nfeats.std(axis=0, keepdims=True))
nfeats[np.isnan(nfeats)] = 0

# Reduce dimension
pca_feats = PCA(n_components=10).fit_transform(nfeats)

# Cluster
clus = KMeans(n_clusters=len(set(colors))).fit(pca_feats).labels_

# Plot features in first 2 PCA dimensions
jitter_pca_feats = pca_feats + np.random.uniform(0, 1, pca_feats.shape)
_ = plt.scatter(jitter_pca_feats[:,0], jitter_pca_feats[:,1], alpha=0.25, c=clus, cmap='rainbow')
plt.show()

# Show roles on graph
np.random.seed(1235)
_ = nx.draw(G, pos=nx.spring_layout(G, iterations=200), 
    node_color=clus, node_size=50, cmap='rainbow', ax=plt.figure().add_subplot(111))

plt.show()
