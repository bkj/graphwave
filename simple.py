#!/usr/bin/env python

"""
    simple.py
"""

from __future__ import division, print_function

import numpy as np
from scipy import sparse

from helpers import estimate_lmax, compute_laplacian

# --
# Heat kernel

class SimpleHeat(object):
    
    def __init__(self, W, taus, lmax=None):
        assert isinstance(taus, list)
        
        self.num_nodes = W.shape[0]
        
        dw = np.asarray(W.sum(axis=1)).squeeze()
        self.L = compute_laplacian(W, dw)
        
        self.lmax = estimate_lmax(self.L) if lmax is None else lmax
        self.taus = taus
    
    def filter(self, signal, order=30):
        return _filter(self.L, self.num_nodes, self.lmax, self.taus, signal, order)


def _compute_cheby_coeff(lmax, taus, i, order=30):
    N = order + 1
    a = lmax / 2.
    
    tmpN = np.arange(N)
    num  = np.cos(np.pi * (tmpN + 0.5) / N)
    
    c = np.zeros(N)
    for o in range(N):
        kernel = lambda x: np.exp(-taus[i] * x / lmax)
        c[o] = 2. / N * np.dot(kernel(a * num + a), np.cos(np.pi * o * (tmpN + 0.5) / N))
        
    return c

def _filter(L, num_nodes, lmax, taus, signal, order=30):
    assert signal.shape[0] == num_nodes
    n_signals = signal.shape[1]
    n_features_out = len(taus)
    
    # --
    # compute_cheby_coeff
    
    c = [_compute_cheby_coeff(lmax, taus, i=i, order=order) for i in range(n_features_out)]
    c = np.atleast_2d(c)
    
    # --
    # cheby_op
    
    r = np.zeros((num_nodes * n_features_out, n_signals))
    
    a = lmax / 2.
    
    twf_old = signal
    twf_cur = (L.dot(signal) - a * signal) / a
    
    tmpN = np.arange(num_nodes, dtype=int)
    for i in range(n_features_out):
        r[tmpN + num_nodes * i] = 0.5 * c[i, 0] * twf_old + c[i, 1] * twf_cur
    
    factor = 2 / a * (L - a * sparse.eye(num_nodes))
    for k in range(2, c.shape[1]):
        twf_new = factor.dot(twf_cur) - twf_old
        
        for i in range(n_features_out):
            r[tmpN + num_nodes * i] += c[i, k] * twf_new
            
        twf_old = twf_cur
        twf_cur = twf_new
    
    # --
    # return
    
    r = r.reshape((num_nodes, n_features_out, n_signals), order='F')
    r = r.transpose((1, 0, 2)) # taus x nodes x signals
    return r
