import numpy as np
from scanorama import *
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, LabelEncoder

from process import load_names
from sketch import srs, reduce_dimensionality
from utils import log

NAMESPACE = 'pbmc'
METHOD = 'svd'
DIMRED = 100
N = 10000

NAMESPACE += '_{}{}_N{}'.format(METHOD, DIMRED, N)

data_names = [
    'data/pbmc/10x/68k_pbmc',
    'data/pbmc/10x/b_cells',
    'data/pbmc/10x/cd14_monocytes',
    'data/pbmc/10x/cd4_t_helper',
    'data/pbmc/10x/cd56_nk',
    'data/pbmc/10x/cytotoxic_t',
    'data/pbmc/10x/memory_t',
    'data/pbmc/10x/regulatory_t',
    'data/pbmc/pbmc_kang',
    'data/pbmc/pbmc_10X',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    
    log('Scanorama integration...')
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)
    datasets_dimred = assemble(datasets_dimred)
    X_dimred = np.concatenate(datasets_dimred)

    cell_labels = (
        open('data/cell_labels/pbmc_cluster.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_

    log('Visualizing full...')
    visualize([ X_dimred ], cell_labels,
              NAMESPACE + '_original', cell_types,
              perplexity=100, n_iter=400,
              image_suffix='.png')
    
    log('SRS...')
    srs_idx = srs(X_dimred, N)

    log('Visualizing sampled...')
    visualize([ X_dimred[srs_idx, :] ], cell_labels[srs_idx],
              NAMESPACE + '_srs', cell_types,
              perplexity=150, n_iter=400, size=20,
              image_suffix='.png')

    log('Done.')
