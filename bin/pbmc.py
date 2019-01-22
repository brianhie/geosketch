from fbpca import pca
import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder, scale

from experiments import *
from process import load_names
from utils import *

NAMESPACE = 'pbmc_68k'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/pbmc/68k'
]

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

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    if not os.path.isfile('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE)):
        log('Dimension reduction with {}...'.format(METHOD))
        X_dimred = reduce_dimensionality(
            normalize(X), method=METHOD, dimred=DIMRED
        )
        log('Dimensionality = {}'.format(X_dimred.shape[1]))
        np.savetxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE))

    labels = (
        open('data/cell_labels/{}_cluster.txt'.format(NAMESPACE))
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(labels)
    cell_labels = le.transform(labels)

    X = np.log1p(normalize(X))
    viz_genes = [ 'CD14' ]

    experiments(
        X_dimred, NAMESPACE,
        cell_labels=cell_labels,
        cell_exp_ratio=True,
        #spectral_nmi=True, louvain_ami=True,
        #rare=True,
        #rare_label=le.transform(['Dendritic'])[0],
        #max_min_dist=True
    )
    exit()
    
    plot_rare(X_dimred, cell_labels, le.transform(['Dendritic'])[0], NAMESPACE)
    
    experiment_gs(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        gene_names=viz_genes, genes=genes, gene_expr=X,
        N_only=20000, kmeans=False, visualize_orig=False
    )
    experiment_uni(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        gene_names=viz_genes, genes=genes, gene_expr=X,
        N_only=20000, kmeans=False, visualize_orig=False
    )
    experiment_srs(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        gene_names=viz_genes, genes=genes, gene_expr=X,
        N_only=20000, kmeans=False, visualize_orig=False
    )
    experiment_kmeanspp(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        gene_names=viz_genes, genes=genes, gene_expr=X,
        N_only=20000, kmeans=False, visualize_orig=False
    )
    
