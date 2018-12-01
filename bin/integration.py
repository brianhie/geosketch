from ample import gs, uniform, srs, kmeanspp
import numpy as np
from scanorama import find_alignments, assemble
from subprocess import Popen
from time import time

from utils import log, mkdir_p

def integrate_sketch(datasets_dimred, integration_fn, integration_fn_args={},
                     sampling_fn=gs, N=2000, n_iter=1):

    sketch_idxs = [ sampling_fn(X, N, replace=False)
                    for X in datasets_dimred ]
    datasets_sketch = [ X[idx] for X, idx in zip(datasets_dimred, sketch_idxs) ]

    for _ in range(n_iter):
        datasets_sketch = integration_fn(datasets_sketch, **integration_fn_args)

    alignments, matches = find_alignments(datasets_sketch)
    
    for (i, j) in matches.keys():
        matches_mnn = matches[(i, j)]
        matches[(i, j)] = [
            (sketch_idxs[i][a], sketch_idxs[j][b]) for a, b in matches_mnn
        ]

    datasets_dimred = assemble(datasets_dimred,
                               alignments=alignments, matches=matches)

    return datasets_dimred

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
