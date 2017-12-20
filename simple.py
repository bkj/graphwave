#!/usr/bin/env python


"""
    simple.py
"""


from __future__ import division, print_function

import numpy as np
from scipy import sparse

# --
# Helpers

def estimate_lmax(L):
    try:
        lmax = sparse.linalg.eigsh(L, k=1, tol=5e-3, ncv=min(L.shape[0], 10), return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1.01
        return lmax
    
    except sparse.linalg.ArpackNoConvergence:
        return 2

def compute_laplacian(W, dw):
    d = np.power(dw, -0.5)
    D = sparse.diags(np.ravel(d), 0).tocsc()
    return sparse.identity(W.shape[0]) - D * W * D

# >>

def characteristic_function(s, t=np.arange(0, 100, 2)):
    return (np.exp(np.complex(0, 1) * s) ** t.reshape(-1, 1)).mean(axis=1)


def featurize(heat_print):
    feats = []
    for i, sig in enumerate(heat_print):
        sig_feats = []
        for node_sig in sig:
            node_feats = characteristic_function(node_sig)
            node_feats = np.column_stack([node_feats.real, node_feats.imag]).reshape(-1)
            sig_feats.append(node_feats)
        
        feats.append(np.vstack(sig_feats))
        
    feats = np.hstack(feats)
    return feats


class SimpleHeat(object):
    
    def __init__(self, L, num_nodes, lmax, taus, normalize=False, **kwargs):
        assert isinstance(taus, list)
        
        self.L = L
        self.num_nodes = num_nodes
        self.lmax = lmax
        
        self._kernels = [lambda x, t=tau: np.exp(-t * x / lmax) for tau in taus]
        
    def _compute_cheby_coeff(self, i, order=30):
        N = order + 1
        a = self.lmax / 2.
        
        tmpN = np.arange(N)
        num  = np.cos(np.pi * (tmpN + 0.5) / N)
        
        c = np.zeros(N)
        for o in range(N):
            c[o] = 2. / N * np.dot(self._kernels[i](a * num + a), np.cos(np.pi * o * (tmpN + 0.5) / N))
            
        return c
    
    def filter(self, signal, order=30):
        assert signal.shape[0] == self.num_nodes
        n_signals = signal.shape[1]
        n_features_out = len(self._kernels)
        
        # --
        # compute_cheby_coeff
        
        c = [self._compute_cheby_coeff(order=order, i=i) for i in range(n_features_out)]
        c = np.atleast_2d(c)
        
        # --
        # cheby_op
        
        r = np.zeros((self.num_nodes * n_features_out, n_signals))
        
        a = self.lmax / 2.
        
        twf_old = signal
        twf_cur = (self.L.dot(signal) - a * signal) / a
        
        tmpN = np.arange(self.num_nodes, dtype=int)
        for i in range(n_features_out):
            r[tmpN + self.num_nodes * i] = 0.5 * c[i, 0] * twf_old + c[i, 1] * twf_cur
        
        factor = 2 / a * (self.L - a * sparse.eye(self.num_nodes))
        for k in range(2, c.shape[1]):
            twf_new = factor.dot(twf_cur) - twf_old
            
            for i in range(n_features_out):
                r[tmpN + self.num_nodes * i] += c[i, k] * twf_new
                
            twf_old = twf_cur
            twf_cur = twf_new
        
        # --
        # return
        
        r = r.reshape((self.num_nodes, n_features_out, n_signals), order='F')
        r = r.transpose((1, 0, 2))
        # ^^ taus x signals x nodes
        return r