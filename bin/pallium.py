import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack

from process import load_names
from sketch import reduce_dimensionality, test
from utils import log

NAMESPACE = 'pallium'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/murine_atlases/neuron_1M/neuron_1M',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    
    log('Dimension reduction with {}...'.format(METHOD))
    X = vstack(datasets)
    X_dimred = reduce_dimensionality(X, method=METHOD, dimred=DIMRED)
    if METHOD == 'jl_sparse':
        X_dimred = X_dimred.toarray()
    log('Dimensionality = {}'.format(X_dimred.shape[1]))

    test(X_dimred, NAMESPACE, kmeans=False, visualize_orig=False)
    
    log('Done.')
