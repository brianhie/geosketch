import numpy as np
from scanorama import *
from scipy.sparse import vstack

from process import load_names
from sketch import reduce_dimensionality, test
from utils import log

np.random.seed(0)

NAMESPACE = 'ica'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/ica/ica_bone_marrow_h5',
    'data/ica/ica_cord_blood_h5',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)

    log('Scanorama integration...')
    datasets_dimred, genes = process_data(datasets, genes)
    datasets_dimred = assemble(datasets_dimred, knn=100, sigma=10)
    X_dimred = np.concatenate(datasets_dimred)
    #log('Dimension reduction with {}...'.format(METHOD))
    #X = vstack(datasets)
    #X_dimred = reduce_dimensionality(X, method=METHOD, dimred=DIMRED)
    #if METHOD == 'jl_sparse':
    #    X_dimred = X_dimred.toarray()
    #log('Dimensionality = {}'.format(X_dimred.shape[1]))

    test(X_dimred, 'ica', perplexity=1000)
    
    log('Done.')
