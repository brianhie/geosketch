import numpy as np
from scanorama import *
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, LabelEncoder

from process import load_names
from sketch import srs, reduce_dimensionality
from utils import log

NAMESPACE = 'pancreas'
METHOD = 'svd'
DIMRED = 100
N = 1000

NAMESPACE += '_{}{}_N{}'.format(METHOD, DIMRED, N)

data_names = [
    'data/pancreas/pancreas_inDrop',
    'data/pancreas/pancreas_multi_celseq2_expression_matrix',
    'data/pancreas/pancreas_multi_celseq_expression_matrix',
    'data/pancreas/pancreas_multi_fluidigmc1_expression_matrix',
    'data/pancreas/pancreas_multi_smartseq2_expression_matrix',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    
    log('Scanorama integration...')
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)
    datasets_dimred = assemble(datasets_dimred)
    X_dimred = np.concatenate(datasets_dimred)
    
    cell_labels = (
        open('data/cell_labels/pancreas_cluster.txt')
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
              perplexity=50, n_iter=400, size=40,
              image_suffix='.png')

    log('Done.')
