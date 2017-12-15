#!/usr/bin/env python

"""
    graphwave.py
"""

import sys
import pygsp
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

def heat_diffusion(G, taus, diff_type="immediate", b=1):
    
    A  = nx.adjacency_matrix(G)
    N  = G.number_of_nodes()
    pG = pygsp.graphs.Graph(A, lap_type='normalized')
    pG.compute_fourier_basis(recompute=True)
    
    Hk = pygsp.filters.Heat(pG, taus, normalize=False)
    
    heat = {i : pd.DataFrame(np.zeros((N,N))) for i in range(len(taus))}
    
    for v in range(N):
            ### for each node v , create a signal that corresponds to a Dirac of energy
            ### centered around v and whic propagates through the network
            f = np.zeros(N)
            f[v] = 1
            Sf_vec = Hk.analyze(f) ### creates the associated heat wavelets
            Sf = Sf_vec.reshape((Sf_vec.size / len(taus), len(taus)), order='F')
            for i in range(len(taus)):
                heat[i].iloc[:,v] = Sf[:,i] ### stores in different dataframes the results
    
    return [heat[i] for i in range(len(taus))]


def characteristic_function(sig, t):
    f = np.zeros((len(t),3))
    
    if type(t) is list:
        f=np.zeros((len(t),3))
        f[0,:]=[0,1,0]
        vec1=[np.exp(complex(0,sig[i])) for i in range(len(sig))]
        for tt in range(1,len(t)):
            f[tt,0]=t[tt]
            vec=[x**t[tt] for x in vec1]
            c=np.mean(vec)
            f[tt,1]=c.real
            f[tt,2]=c.imag
    else:
        c=np.mean([np.exp(complex(0,t*sig[i])) for i in range(len(sig))])
        f=[t,np.real(c),np.imag(c)]
    
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


def graphwave(G, taus, t=range(0, 100, 2), type_graph="nx", **kwargs):
    if taus is None:
        taus = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 2.1, 2.3, 2.5, 2.7] + range(3,5)
        
        # Compute the optimal embedding
        pG = pygsp.graphs.Graph(nx.adjacency_matrix(G), lap_type='normalized')
        pG.compute_fourier_basis(recompute=True)
        
        l1 = np.where(pG.e > 0.1 / pG.N) # safety check to ensure that the graph is indeed connected
        l1 = pG.e[l1[0][0]]
        smax, smin = -np.log([0.90, 0.99]) * np.sqrt(pG.e[-1] / l1)
        
        max_ind = len(taus)
        if np.any(taus > smax):
            max_ind = np.where(taus > smax)[0][0]
        
        min_ind = 0
        if np.any(taus < smin):
            min_ind = np.where(taus < smin)[0][-1]
        
        taus = taus[min_ind:max_ind]
    
    # Compute the heat wavelets
    heat_print = heat_diffusion(G, taus)
    chi = featurize_characteristic_function(heat_print, t)
    return chi, heat_print, taus


