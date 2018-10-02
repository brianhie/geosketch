from fbpca import pca
import numpy as np
from sklearn.random_projection import SparseRandomProjection as JLSparse
import sys

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
    
