#!/usr/bin/env python

"""
    helpers.py
"""


from __future__ import division, print_function

import numpy as np
from joblib import Parallel, delayed

def par_graphwave(hk, n_chunks=10, pct_queries=1.0, **kwargs):
    global _runner
    def _runner(chunk):
        return hk.featurize(chunk)
    
    if pct_queries == 1:
        queries = np.eye(hk.num_nodes)
    else:
        num_queries = int(pct_queries * hk.num_nodes)
        query_idx = np.sort(np.random.choice(hk.num_nodes, num_queries, replace=False))
        
        queries = np.zeros((hk.num_nodes, num_queries), dtype=int)
        for i, idx in enumerate(query_idx):
            queries[idx, i] = 1
    
    chunks = np.array_split(queries, n_chunks, axis=1)
    
    jobs = [delayed(_runner)(chunk) for chunk in chunks]
    results = Parallel(**kwargs)(jobs)
    
    return np.vstack(results)


