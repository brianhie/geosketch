from ample import gs, uniform, srs, kmeanspp
import numpy as np
from scanorama import batch_bias
from scipy.sparse import csr_matrix, find
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from subprocess import Popen
from time import time

from utils import log, mkdir_p

def integrate_sketch(datasets_dimred, integration_fn, integration_fn_args={},
                     sampling_fn=uniform, N=2000, n_iter=1):

    sketch_idxs = [ sampling_fn(X, N, replace=False)
                    for X in datasets_dimred ]
    datasets_sketch = [ X[idx] for X, idx in zip(datasets_dimred, sketch_idxs) ]

    for _ in range(n_iter):
        datasets_int = integration_fn(datasets_sketch[:], **integration_fn_args)

    labels = []
    curr_label = 0
    for i, a in enumerate(datasets_sketch):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        curr_label += 1
    labels = np.array(labels, dtype=int)

    for i, (X_dimred, X_sketch) in enumerate(zip(datasets_dimred, datasets_sketch)):
        #datasets_int[i] = batch_bias(X_dimred, X_sketch, datasets_int[i], batch_size=10000)
        
        neigh = NearestNeighbors(n_neighbors=3).fit(X_sketch)
        neigh_graph = neigh.kneighbors_graph(X_dimred, mode='distance')
        idx_i, idx_j, val = find(neigh_graph)
        neigh_graph = csr_matrix((1. / val, (idx_i, idx_j)))
        neigh_graph = normalize(neigh_graph, norm='l1', axis=1)
        
        #neigh_graph = normalize(rbf_kernel(X_dimred, X_sketch), norm='l1', axis=1)
                                
        datasets_int[i] = neigh_graph.dot(datasets_int[i])

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
