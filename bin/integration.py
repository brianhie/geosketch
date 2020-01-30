from geosketch import gs, uniform, srs, kmeanspp
import numpy as np
from scanorama import transform
from scipy.sparse import csr_matrix, find
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from subprocess import Popen
import sys
from time import time

from utils import log, mkdir_p

def integrate_sketch(datasets_dimred, integration_fn, integration_fn_args={},
                     sampling_type='geosketch', N=10000):

    if sampling_type == 'geosketch':
        from geosketch import gs
        sampling_fn = gs
    else:
        from geosketch import uniform
        sampling_fn = uniform

    # Sketch each dataset.

    sketch_idxs = [
        sorted(set(sampling_fn(X, N, replace=False)))
        for X in datasets_dimred
    ]
    datasets_sketch = [ X[idx] for X, idx in zip(datasets_dimred, sketch_idxs) ]

    # Integrate the dataset sketches.

    datasets_int = integration_fn(datasets_sketch[:], **integration_fn_args)

    # Apply integrated coordinates back to full data.

    labels = []
    curr_label = 0
    for i, a in enumerate(datasets_sketch):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        curr_label += 1
    labels = np.array(labels, dtype=int)

    for i, (X_dimred, X_sketch) in enumerate(zip(datasets_dimred, datasets_sketch)):
        X_int = datasets_int[i]

        neigh = NearestNeighbors(n_neighbors=3).fit(X_dimred)
        _, neigh_idx = neigh.kneighbors(X_sketch)

        ds_idxs, ref_idxs = [], []
        for ref_idx in range(neigh_idx.shape[0]):
            for k_idx in range(neigh_idx.shape[1]):
                ds_idxs.append(neigh_idx[ref_idx, k_idx])
                ref_idxs.append(ref_idx)

        bias = transform(X_dimred, X_int, ds_idxs, ref_idxs, 15, batch_size=1000)

        datasets_int[i] = X_dimred + bias

    return datasets_int

def harmony(datasets_dimred):
    mkdir_p('data/harmony')

    n_samples = sum([ ds.shape[0] for ds in datasets_dimred ])

    embed_fname = 'data/harmony/embedding.txt'
    label_fname = 'data/harmony/labels.txt'

    np.savetxt(embed_fname, np.concatenate(datasets_dimred))

    labels = []
    curr_label = 0
    for i, a in enumerate(datasets_dimred):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        curr_label += 1
    labels = np.array(labels, dtype=int)

    np.savetxt(label_fname, labels)

    log('Integrating with harmony...')

    rcode = Popen('Rscript R/harmony.R {} {} > harmony.log 2>&1'
                  .format(embed_fname, label_fname), shell=True).wait()
    if rcode != 0:
        sys.stderr.write('ERROR: subprocess returned error code {}\n'
                         .format(rcode))
        exit(rcode)

    log('Done with harmony integration')

    integrated = np.loadtxt('data/harmony/integrated.txt')

    assert(n_samples == integrated.shape[0])

    datasets_return = []
    base = 0
    for ds in datasets_dimred:
        datasets_return.append(integrated[base:(base + ds.shape[0])])
        base += ds.shape[0]

    return datasets_return
