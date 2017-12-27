#!/usr/bin/env python

"""
    simple-example.py
"""

from __future__ import print_function

import argparse
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from simple import SimpleHeat
from helpers import featurize
from utils.build_graph import build_regular_structure

np.random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, default='./simple-feats')
    parser.add_argument('--plot', action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    
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
    
    print('simple-example.py: saving feats to %s.npy' % args.outpath)
    np.save(args.outpath, feats)
    
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
    
    if args.plot:
        # Plot features in first 2 PCA dimensions
        jitter_pca_feats = pca_feats + np.random.uniform(0, 1, pca_feats.shape)
        _ = plt.scatter(jitter_pca_feats[:,0], jitter_pca_feats[:,1], alpha=0.25, c=clus, cmap='rainbow')
        plt.show()
        
        # Show roles on graph
        np.random.seed(1235)
        _ = nx.draw(G, pos=nx.spring_layout(G, iterations=200), 
            node_color=clus, node_size=50, cmap='rainbow', ax=plt.figure().add_subplot(111))
        plt.show()
