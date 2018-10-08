from fbpca import pca
import datetime
from sklearn.random_projection import SparseRandomProjection as JLSparse
import sys

# Default parameters.
DIMRED = 100

def log(string):
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()

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
