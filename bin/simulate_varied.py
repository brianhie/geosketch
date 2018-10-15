import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder

from dropclust_experiments import experiment_dropclust
from experiments import *
from process import load_names
from utils import *

NAMESPACE = 'simulate_varied'
METHOD = 'svd'
DIMRED = 100

data_names = [ 'data/simulate/simulate_varied' ]

if __name__ == '__main__':
    if not os.path.isfile('data/dimred_{}.txt'.format(NAMESPACE)):
        datasets, genes_list, n_cells = load_names(data_names)
        datasets, genes = merge_datasets(datasets, genes_list)
        log('Dimension reduction with {}...'.format(METHOD))
        X = vstack(datasets)
        X_dimred = reduce_dimensionality(X, method=METHOD, dimred=DIMRED)
        if METHOD == 'jl_sparse':
            X_dimred = X_dimred.toarray()
        log('Dimensionality = {}'.format(X_dimred.shape[1]))
        np.savetxt('data/dimred_{}.txt'.format(NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred_{}.txt'.format(NAMESPACE))

    cell_labels = (
        open('data/cell_labels/simulate_varied_cluster.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)

    #experiment_efficiency_louvain(X_dimred, cell_labels)
    #
    #experiment_efficiency_kmeans(X_dimred, cell_labels)
    #
    experiment_gs(X_dimred, NAMESPACE, cell_labels=cell_labels,
                  kmeans=False, visualize_orig=False)
    #experiment_dropclust(X_dimred, 'data/' + NAMESPACE,
    #                     cell_labels=cell_labels,
    #                     perplexity=500)

    log('Done.')
