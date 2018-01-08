#!/usr/bin/env python

"""
    helpers.py
"""


from __future__ import division, print_function

import numpy as np
from joblib import Parallel, delayed

def par_graphwave(hk, n_chunks=10, **kwargs):
    """ parallel filter and featurize """
    # assert hk.num_nodes % n_chunks == 0
    
    global _runner
    def _runner(chunk):
        return hk.featurize(chunk)
    
    chunks = np.array_split(np.eye(hk.num_nodes), n_chunks, axis=1)
    
    print('par_graphwave -> starting jobs')
    jobs = [delayed(_runner)(chunk) for chunk in chunks]
    results = Parallel(**kwargs)(jobs)
    print('par_graphwave -> finishing jobs')
    
    return np.vstack(results)