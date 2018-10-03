from fbpca import pca
import numpy as np
import os
from scanorama import visualize
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.random_projection import SparseRandomProjection as JLSparse
import sys

from utils import log

DIMRED = 100

def srs(X, N, seed=None):
    if not seed is None:
        np.random.seed(seed)
    
    n_samples, n_features = X.shape

    srs_idx = []
    for i in range(N):
        Phi_i = np.random.normal(size=(n_features))
        Q_i = X.dot(Phi_i)
        Q_i[srs_idx] = 0
        k_argmax = np.argmax(np.absolute(Q_i))
        srs_idx.append(k_argmax)

    return srs_idx

def reduce_dimensionality(X, method='svd', dimred=DIMRED):
    if method == 'svd':
        k = min((dimred, X.shape[0], X.shape[1]))
        U, s, Vt = pca(X, k=k) # Automatically centers.
        return U[:, range(k)] * s[range(k)]
    elif method == 'jl_sparse':
        jls = JLSparse(n_components=dimred)
        return jls.fit_transform(X)
    else:
        sys.stderr.write('ERROR: Unknown method {}.'.format(svd))
        exit(1)

def test(X_dimred, name, kmeans=True, visualize_orig=True,
         downsample=True, n_downsample=100000, perplexity=500):

    # Assign cells to clusters.

    if kmeans or \
       not os.path.isfile('data/cell_labels/{}.txt'.format(name)):
        log('K-means...')
        km = KMeans(n_clusters=10, n_jobs=10, verbose=0)
        km.fit(X_dimred)
        np.savetxt('data/cell_labels/{}.txt'.format(name), km.labels_)
    
    cell_labels = (
        open('data/cell_labels/{}.txt'.format(name))
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    cell_types = le.classes_

    # Visualize original data.
    
    if visualize_orig:
         log('Visualizing original...')
     
         if downsample:
             log('Visualization will downsample to {}...'
                 .format(n_downsample))
             idx = np.random.choice(
                 X_dimred.shape[0], size=n_downsample, replace=False
             )
         else:
             idx = range(X_dimred.shape[0])
     
         embedding = visualize(
             [ X_dimred[idx, :] ], cell_labels[idx],
             name + '_orig{}'.format(len(idx)), cell_types,
             perplexity=perplexity, n_iter=400, image_suffix='.png'
         )
         np.savetxt('data/embedding_{}.txt'.format(name), embedding)

    # Downsample while preserving structure and visualize.

    Ns = [ 1000, 5000, 10000, 20000, 50000 ]

    for N in Ns:
        if N >= X_dimred.shape[0]:
            continue

        log('SRS {}...'.format(N))
        srs_idx = srs(X_dimred, N)
        log('Found {} entries'.format(len(set(srs_idx))))

        log('Visualizing sampled...')
        visualize([ X_dimred[srs_idx, :] ], cell_labels[srs_idx],
                  name + '_srs{}'.format(N), cell_types,
                  perplexity=(N/200), n_iter=500, size=max(int(20000/N), 1),
                  image_suffix='.png')
