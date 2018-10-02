import numpy as np
from scanorama import *
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, LabelEncoder

from process import load_names
from sketch import srs, reduce_dimensionality
from utils import log

NAMESPACE = 'mouse_brain'
METHOD = 'svd'
DIMRED = 100
N = 20000

NAMESPACE += '_{}{}_N{}'.format(METHOD, DIMRED, N)

data_names = [
    'data/mouse_brain/dropviz/Cerebellum_ALT',
    'data/mouse_brain/dropviz/Cortex_noRep5_FRONTALonly',
    'data/mouse_brain/dropviz/Cortex_noRep5_POSTERIORonly',
    'data/mouse_brain/dropviz/EntoPeduncular',
    'data/mouse_brain/dropviz/GlobusPallidus',
    'data/mouse_brain/dropviz/Hippocampus',
    'data/mouse_brain/dropviz/Striatum',
    'data/mouse_brain/dropviz/SubstantiaNigra',
    'data/mouse_brain/dropviz/Thalamus',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets = [ normalize(ds, axis=1) for ds in datasets ]

    log('Dimension reduction...')
    X = vstack(datasets)
    X_dimred = reduce_dimensionality(X, method=METHOD, dimred=DIMRED)
    if METHOD == 'jl_sparse':
        X_dimred = X_dimred.toarray()
    log('Dimensionality = {}'.format(X_dimred.shape[1]))

    log('K-means...')
    km = KMeans(n_clusters=10, n_jobs=10, verbose=0)
    km.fit(X_dimred)
    names = np.array([ str(x) for x in sorted(set(km.labels_)) ])
    np.savetxt('data/cell_labels/mouse_brain.txt', km.labels_)

    cell_labels = (
        open('data/cell_labels/mouse_brain.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_

    log('Visualizing full...')
    embedding = visualize([ X_dimred ], cell_labels,
                          NAMESPACE + '_original', cell_types,
                          perplexity=1000, n_iter=400,
                          image_suffix='.png')
    np.savetxt('data/embedding_mouse_brain.txt', embedding)
    
    Ns = [ 1000, 10000, 20000, 50000 ]

    for N in Ns:
        log('SRS {}...'.format(N))
        srs_idx = srs(X_dimred, N)

        log('Visualizing sampled...')
        visualize([ X_dimred[srs_idx, :] ], cell_labels[srs_idx],
                  NAMESPACE + '_srs{}'.format(N), cell_types,
                  perplexity=100, n_iter=400,
                  image_suffix='.png')

    log('Done.')
