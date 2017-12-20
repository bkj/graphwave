#!/usr/bin/env python

"""
    helpers.py
"""

from __future__ import division, print_function

import numpy as np
from scipy import sparse

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


def characteristic_function(s, t=np.arange(0, 100, 2)):
    return (np.exp(np.complex(0, 1) * s) ** t.reshape(-1, 1)).mean(axis=1)


def featurize(heat_print):
    """ compute graphwave features """
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


def delayed_featurize(heat_print):
    """ compute graphwave features -- but don't average """
    feats = []
    for i, sig in enumerate(heat_print):
        sig_feats = []
        for node_sig in sig:
            node_feats = characteristic_function(node_sig)
            # node_feats = np.column_stack([node_feats.real, node_feats.imag]).reshape(-1)
            sig_feats.append(node_feats)
        
        feats.append(np.vstack(sig_feats))
        
    feats = np.hstack(feats)
    return feats