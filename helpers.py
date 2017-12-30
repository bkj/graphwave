#!/usr/bin/env python

"""
    helpers.py
"""

from __future__ import division, print_function

import numpy as np
from scipy import sparse
from joblib import Parallel, delayed


def characteristic_function(s, t=np.arange(0, 100, 2)):
    return (np.exp(np.complex(0, 1) * s) ** t.reshape(-1, 1)).mean(axis=1)


def featurize(heat_print, delayed=False):
    """ compute graphwave features """
    feats = []
    for i, sig in enumerate(heat_print):
        sig_feats = []
        for node_sig in sig:
            node_feats = characteristic_function(node_sig)
            if not delayed:
                node_feats = np.column_stack([node_feats.real, node_feats.imag]).reshape(-1)
            sig_feats.append(node_feats)
        
        feats.append(np.vstack(sig_feats))
        
    feats = np.hstack(feats)
    return feats


def par_graphwave(hk, n_chunks=10, **kwargs):
    """ parallel filter and featurize """
    assert hk.num_nodes % n_chunks == 0
    
    global _runner
    def _runner(chunk):
        return featurize(hk.filter(chunk), delayed=True)
    
    chunks = np.array_split(np.eye(hk.num_nodes), n_chunks, axis=1)
    
    print('par_graphwave -> starting jobs')
    jobs = [delayed(_runner)(chunk) for chunk in chunks]
    results = Parallel(**kwargs)(jobs)
    print('par_graphwave -> finishing jobs')
    
    tmp = np.array(results).mean(axis=0)
    pfeats = np.empty((tmp.shape[0], tmp.shape[1] * 2))
    pfeats[:,0::2] = tmp.real
    pfeats[:,1::2] = tmp.imag
    
    return pfeats

