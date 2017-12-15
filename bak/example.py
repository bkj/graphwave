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

from GraphWave.shapes import build_graph

np.random.seed(123)

# --
# Helpers

def heat_diffusion(adj, taus):
    pG = pygsp.graphs.Graph(adj, lap_type='normalized')
    pG.compute_fourier_basis(recompute=True)
    
    Hk = pygsp.filters.Heat(pG, taus, normalize=False)
    
    heat = {i : pd.DataFrame(np.zeros((pG.N, pG.N))) for i in range(len(taus))}
    
    for v in range(pG.N):
            f = np.zeros(pG.N)
            f[v] = 1
            
            Sf_vec = Hk.analyze(f)
            Sf = Sf_vec.reshape((Sf_vec.size / len(taus), len(taus)), order='F')
            for i in range(len(taus)):
                heat[i].iloc[:,v] = Sf[:,i]
    
    return [heat[i] for i in range(len(taus))]


def characteristic_function(sig, t):
    f = np.zeros((len(t), 3))
    
    if type(t) is list:
        f = np.zeros((len(t), 3))
        f[0,:] = [0, 1, 0]
        vec1 = [np.exp(complex(0, sig[i])) for i in range(len(sig))]
        for tt in range(1, len(t)):
            f[tt,0] = t[tt]
            vec = [x ** t[tt] for x in vec1]
            
            c = np.mean(vec)
            f[tt,1] = c.real
            f[tt,2] = c.imag
    else:
        c = np.mean([np.exp(complex(0,t*sig[i])) for i in range(len(sig))])
        f = [t,np.real(c),np.imag(c)]
    
    return f


def featurize_characteristic_function(heat_print, t):
    chi = np.empty((heat_print[0].shape[0], 2 * len(t) * len(heat_print)))
    
    for tau in range(len(heat_print)):
        sig = heat_print[tau]
        for ind in range(heat_print[0].shape[0]):
            s = sig.iloc[:, ind].tolist()
            c = characteristic_function(s, t)
            chi[ind, tau * 2 * len(t):(tau+1) * 2 * len(t)] = np.reshape(c[:, 1:], [1, 2 * len(t)])
            
    return chi



# --
# Create graph

G, colors = build_graph.build_regular_structure(
    width_basis=15,
    basis_type="cycle",
    nb_shapes=5,
    shape=["house"],
    start=0,
    add_random_edges=0
)

# --
# Choose taus

# --
# Compute the heat wavelets

np.random.seed(123)
adj = nx.adjacency_matrix(G)
heat_print = heat_diffusion(adj, taus=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
chi = featurize_characteristic_function(heat_print, t=range(0, 100, 2))

# --
# Cluster resulting features

nchi = chi - chi.mean(axis=0, keepdims=True)
nchi /= nchi.std(axis=0, keepdims=True)
nchi[np.isnan(nchi)] = 0

feats = PCA(n_components=10).fit_transform(nchi)

clus = KMeans(n_clusters=len(set(colors))).fit(feats).labels_

jitter_feats = feats + np.random.uniform(0, 1, feats.shape)

_ = plt.scatter(jitter_feats[:,0], jitter_feats[:,1], alpha=0.25, c=clus, cmap='rainbow')
show_plot()

np.random.seed(1235)
_ = nx.draw(G, pos=nx.spring_layout(G, iterations=200), 
    node_color=clus, node_size=50, cmap='rainbow', ax=plt.figure().add_subplot(111))

show_plot()
