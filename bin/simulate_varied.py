import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack

from process import load_names
from experiments import *
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

    from sketch import srs, centroid_label
    srs_idx = srs(X_dimred, 2000)
    kmeans_k = 5
    km = KMeans(n_clusters=kmeans_k, n_jobs=10, verbose=0)
    km.fit(X_dimred[srs_idx, :])
    cell_labels = centroid_label(X_dimred, km.cluster_centers_,
                                 range(kmeans_k))
    cell_types = [ str(k) for k in range(kmeans_k) ]
    visualize(
        [ X_dimred ], cell_labels,
        NAMESPACE + '_cl', cell_types,
        perplexity=50, n_iter=400, image_suffix='.png'
    )

    #experiment_efficiency_kmeans(X_dimred, labels)
    experiment_srs(X_dimred, NAMESPACE, kmeans_k=50)
    
    log('Done.')
