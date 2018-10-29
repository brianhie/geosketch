import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

from experiments import *
from process import load_names
from utils import *

def plot(X, title, labels):
    plot_clusters(X, labels)
    plt.title(title)
    plt.savefig('{}.png'.format(title))

if __name__ == '__main__':
    NAMESPACE = 'tabula_ss2'
    from tabula_ss2 import data_names, load_cells, keep_valid

    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)
    cells = load_cells(data_names)
    valid_idx = keep_valid(cells)
    X = vstack(datasets)
    X = X[valid_idx, :]
#    datasets, genes_list, n_cells = load_names(data_names, norm=False)
#    datasets, genes = merge_datasets(datasets, genes_list)
#    X = vstack(datasets).toarray()

    cell_labels = (
        open('data/cell_labels/{}_cluster.txt'.format(NAMESPACE))
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)

    print(X.shape[0])
    print(len(cell_labels))

    # PCA, no centering.
    X_dimred = reduce_dimensionality(X, method='svd', dimred=2, raw=True)
    plot(X_dimred, 'pca', cell_labels)

    # Centering, PCA.
    X_dimred = reduce_dimensionality(X, method='svd', dimred=2)
    plot(X_dimred, 'center_pca', cell_labels)

    # Normalizing, centering, PCA.
    X_dimred = reduce_dimensionality(normalize(X), method='svd', dimred=2)
    plot(X_dimred, 'norm_center_pca', cell_labels)

    # Normalizing, log-transforming, centering, PCA.
    X_dimred = reduce_dimensionality(np.log1p(normalize(X)), method='svd', dimred=2)
    plot(X_dimred, 'norm_log_center_pca', cell_labels)
    
    # Normalizing, log-transforming, HVG, centering, PCA.
    X_dimred = reduce_dimensionality(np.log1p(normalize(X)), method='hvg', dimred=1000)
    X_dimred = reduce_dimensionality(X_dimred, method='svd', dimred=2, raw=True)
    plot(X_dimred, 'norm_log_hvg_center_pca', cell_labels)

    exit()
    # Centering, normalizing, PCA.
    X_dimred = reduce_dimensionality(normalize(scale(X, with_std=False)), method='svd', dimred=2)
    plot(X_dimred, 'center_norm_pca', cell_labels)

    # Normalizing, centering, PCA, normalizing.
    X_dimred = reduce_dimensionality(normalize(X), method='svd', dimred=2)
    X_dimred = normalize(X_dimred)
    plot(X_dimred, 'norm_center_pca_norm', cell_labels)
