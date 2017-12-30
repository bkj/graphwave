#!/usr/bin/env python


"""
    heat.py
"""


from __future__ import division, print_function

import cupy
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


# --
# Heat kernel

class Heat(object):
    
    def __init__(self, W, taus, lmax=None):
        assert isinstance(taus, list)
        
        self.num_nodes = W.shape[0]
        
        dw = np.asarray(W.sum(axis=1)).squeeze()
        self.L = compute_laplacian(W, dw)
        
        self.lmax = estimate_lmax(self.L) if lmax is None else lmax
        self.taus = taus
        
    def _compute_cheby_coeff(self, tau, order=30):
        N = order + 1
        a = self.lmax / 2.
        
        tmpN = np.arange(N)
        num  = np.cos(np.pi * (tmpN + 0.5) / N)
        
        c = np.zeros(N)
        for o in range(N):
            kernel = lambda x: np.exp(-tau * x / self.lmax)
            c[o] = 2. / N * np.dot(kernel(a * num + a), np.cos(np.pi * o * (tmpN + 0.5) / N))
            
        return c
    
    def filter(self, signal, order=30):
        assert signal.shape[0] == self.num_nodes
        n_signals = signal.shape[1]
        n_features_out = len(self.taus)
        
        # --
        # compute_cheby_coeff
        
        c = [self._compute_cheby_coeff(tau=tau, order=order) for tau in self.taus]
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


class CupyHeat(Heat):
    
    def __init__(self, W, taus, lmax=None):
        super(CupyHeat, self).__init__(W, taus, lmax=lmax)
        self.L = cupy.sparse.csr_matrix(self.L)
    
    def filter(self, signal, order=30):
        assert signal.shape[0] == self.num_nodes, 'signal.shape[0] != self.num_nodes'
        n_signals = signal.shape[1]
        n_features_out = len(self.taus)
        
        # --
        # compute_cheby_coeff
        
        c = [self._compute_cheby_coeff(tau=tau, order=order) for tau in self.taus]
        c = np.atleast_2d(c)
        
        # --
        # cheby_op
        r = cupy.zeros((self.num_nodes * n_features_out, n_signals))
        a = self.lmax / 2.
        
        signal_ = cupy.array(signal)
        twf_old = signal_
        twf_cur = (self.L.dot(signal_) - a * signal_) / a
        
        tmpN = np.arange(self.num_nodes, dtype=int)
        for i in range(n_features_out):
            r[tmpN + self.num_nodes * i] = 0.5 * c[i, 0] * twf_old + c[i, 1] * twf_cur
        
        factor = 2 / a * (self.L - a * cupy.sparse.eye(self.num_nodes))
        for k in range(2, c.shape[1]):
            twf_new = factor.dot(twf_cur) - twf_old
            
            for i in range(n_features_out):
                r[tmpN + self.num_nodes * i] += c[i, k] * twf_new
                
            twf_old = twf_cur
            twf_cur = twf_new
        
        # --
        # return
        
        r = r.get().reshape((self.num_nodes, n_features_out, n_signals), order='F')
        r = r.transpose((1, 0, 2))
        # ^^ taus x signals x nodes
        return r

