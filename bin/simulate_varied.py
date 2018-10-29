import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder, scale

from experiments import *
from process import load_names
from utils import *

NAMESPACE = 'simulate_varied'
METHOD = 'svd'
DIMRED = 100

data_names = [ 'data/simulate/simulate_varied' ]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)

    log('Dimension reduction with {}...'.format(METHOD))
    X_dimred = reduce_dimensionality(
        normalize(scale(X.toarray(), with_std=False)), method=METHOD, dimred=DIMRED
    )
    log('Dimensionality = {}'.format(X_dimred.shape[1]))
    np.savetxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
    #if not os.path.isfile('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE)):
    #    log('Dimension reduction with {}...'.format(METHOD))
    #    X_dimred = reduce_dimensionality(
    #        normalize(X), method=METHOD, dimred=DIMRED
    #    )
    #    log('Dimensionality = {}'.format(X_dimred.shape[1]))
    #    np.savetxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
    #else:
    #    X_dimred = np.loadtxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE))

    cell_labels = (
        open('data/cell_labels/simulate_varied_cluster.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)

    experiment_gs(X_dimred, NAMESPACE, cell_labels=cell_labels,
                  kmeans=False, visualize_orig=False)
    
    rare(X_dimred, NAMESPACE, cell_labels, le.transform(['Group4'])[0])
    
    balance(X_dimred, NAMESPACE, cell_labels)
    
    exit()
    
    experiment_uni(X_dimred, NAMESPACE, cell_labels=cell_labels,
                   kmeans=False, visualize_orig=False)
    
    name = 'data/{}'.format(NAMESPACE)
    if not os.path.isfile('{}/matrix.mtx'.format(name)):
        from save_mtx import save_mtx
        save_mtx(name, csr_matrix(X), [ str(i) for i in range(X.shape[1]) ])

    experiment_dropclust(X_dimred, name, cell_labels)

    experiment_efficiency_kmeans(X_dimred, cell_labels)

    experiment_efficiency_louvain(X_dimred, cell_labels)
    
    log('Done.')
