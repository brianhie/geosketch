import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack

from process import load_names
from experiments import *
from utils import *

NAMESPACE = 'granuloma'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/granuloma/full',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)

    if not os.path.isfile('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE)):
        log('Dimension reduction with {}...'.format(METHOD))
        X_dimred = reduce_dimensionality(
            normalize(X), method=METHOD, dimred=DIMRED
        )
        log('Dimensionality = {}'.format(X_dimred.shape[1]))
        np.savetxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE))

    from geosketch import gs
    samp_idx = gs(X_dimred, 1000, replace=False)
    save_sketch(X, samp_idx, genes, NAMESPACE + '/1000')
    
    for scale in [ 0.05, 0.1, 0.2 ]:
        N = int(X.shape[0] * scale)
        samp_idx = gs(X_dimred, N, replace=False)
        save_sketch(X, samp_idx, genes, NAMESPACE + '/' + str(N))
