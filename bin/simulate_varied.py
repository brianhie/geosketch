import datetime
from scanorama import *
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, LabelEncoder

from process import load_names
from sketch import srs, reduce_dimensionality
from utils import log

NAMESPACE = 'simulate_varied'
METHOD = 'svd'
DIMRED = 100
N = 10000

NAMESPACE += '_{}{}_N{}'.format(METHOD, DIMRED, N)

data_names = [ 'data/simulate/simulate_varied' ]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]

    log('Dimension reduction...')
    X = vstack(datasets)
    X_dimred = reduce_dimensionality(X, method=METHOD, dimred=DIMRED)
    if METHOD == 'jl_sparse':
        X_dimred = X_dimred.toarray()
    log('Dimensionality = {}'.format(X_dimred.shape[1]))

    #log('K-means...')
    #km = KMeans(n_clusters=10, n_jobs=10, verbose=0)
    #km.fit(X_dimred)
    #names = np.array([ str(x) for x in sorted(set(km.labels_)) ])
    #np.savetxt('data/cell_labels/{}.txt'.format(NAMESPACE),
    #           km.labels_)

    cell_labels = (
        open('data/cell_labels/simulate_varied.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_
    
    log('Visualizing original...')
    visualize([ X_dimred ], cell_labels, NAMESPACE + '_original',
              cell_types, perplexity=10, n_iter=500, image_suffix='.png',
              size=20)

    log('SRS...')
    srs_idx = srs(X_dimred, N)
    
    log('Visualizing sampled...')
    X_twoD = X_dimred[:, range(2)]
    visualize([ X_dimred[srs_idx, :] ], cell_labels[srs_idx],
              NAMESPACE + '_srs', cell_types,
              perplexity=10, n_iter=500, image_suffix='.png', size=40)

    log('Done.')
