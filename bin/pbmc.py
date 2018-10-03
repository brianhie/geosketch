import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack

from process import load_names
from sketch import reduce_dimensionality, test
from utils import log

NAMESPACE = 'pbmc'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/pbmc/10x/68k_pbmc',
    'data/pbmc/10x/b_cells',
    'data/pbmc/10x/cd14_monocytes',
    'data/pbmc/10x/cd4_t_helper',
    'data/pbmc/10x/cd56_nk',
    'data/pbmc/10x/cytotoxic_t',
    'data/pbmc/10x/memory_t',
    'data/pbmc/10x/regulatory_t',
    'data/pbmc/pbmc_10X',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    
    log('Scanorama integration...')
    datasets_dimred, genes = process_data(datasets, genes)
    datasets_dimred = assemble(datasets_dimred)
    X_dimred = np.concatenate(datasets_dimred)
    #log('Dimension reduction with {}...'.format(METHOD))
    #X = vstack(datasets)
    #X_dimred = reduce_dimensionality(X, method=METHOD, dimred=DIMRED)
    #if METHOD == 'jl_sparse':
    #    X_dimred = X_dimred.toarray()
    #log('Dimensionality = {}'.format(X_dimred.shape[1]))

    test(X_dimred, NAMESPACE, perplexity=100,
         kmeans=False, visualize_orig=False)

    log('Done.')
