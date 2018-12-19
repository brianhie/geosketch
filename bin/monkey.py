import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack

from process import load_names
from experiments import *
from utils import *

NAMESPACE = 'monkey'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/monkey'
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

    from ample import gs

    idx1k = gs(X_dimred, 1000, replace=False)
    np.savetxt('data/monkey/sketch_idx_1000.txt', np.array(idx1k), fmt='%d')
    
    idx3k = gs(X_dimred, 3000, replace=False)
    np.savetxt('data/monkey/sketch_idx_3000.txt', np.array(idx3k), fmt='%d')

    idx5k = gs(X_dimred, 5000, replace=False)
    np.savetxt('data/monkey/sketch_idx_5000.txt', np.array(idx5k), fmt='%d')
    
