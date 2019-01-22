from fbpca import pca
import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder, scale

from experiments import *
from process import load_names
from utils import *

NAMESPACE = '293t_jurkat'
METHOD = 'svd'
DIMRED = 100

data_names = [ 'data/293t_jurkat/jurkat_293t_99_1' ]

def plot(X, title, labels, bold=None):
    plot_clusters(X, labels)
    if bold:
        plot_clusters(X[bold], labels[bold], s=20)
    plt.title(title)
    plt.savefig('{}.png'.format(title))
    
if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)
    
    k = DIMRED
    U, s, Vt = pca(normalize(X), k=k)
    X_dimred = U[:, :k] * s[:k]

    labels = (
        open('data/cell_labels/jurkat_293t_99_1_clusters.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(labels)
    cell_labels = le.transform(labels)

    experiments(
        X_dimred, NAMESPACE,
        cell_labels=cell_labels,
        cell_exp_ratio=True,
        #spectral_nmi=True, louvain_ami=True,
        #rare=True,
        #rare_label=le.transform(['293t'])[0],
        #max_min_dist=True,
    )
    exit()

    from differential_entropies import differential_entropies
    differential_entropies(X_dimred, labels)
    
    plot_rare(X_dimred, cell_labels, le.transform(['293t'])[0], NAMESPACE)
