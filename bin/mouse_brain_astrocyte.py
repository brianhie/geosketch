import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize, LabelEncoder

from experiments import *
from mouse_brain import keep_valid, data_names
from process import load_names
from utils import *

np.random.seed(0)

NAMESPACE = 'mouse_brain'
METHOD = 'svd'
DIMRED = 100


def auroc(X, genes, labels, focus, background=None):
    assert(len(genes) == X.shape[1])

    focus_idx = focus == labels
    if background is None:
        background_idx = range(X.shape[0])
    else:
        background_idx = background == labels
        
    for g, gene in enumerate(genes):
        x_gene = X[:, g]
        x_focus = x_gene[focus_idx]
        x_background = x_gene[background_idx]
        y_score = np.concatenate([ x_focus, x_background ])
        
        y_true = np.zeros(len(x_focus) + len(x_background))
        y_true[:len(x_focus)] = 1

        auroc = roc_auc_score(y_true, y_score)

        print('{}\t{}'.format(gene, auroc))

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    keep_valid(datasets)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)

    if not os.path.isfile('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE)):
        log('Dimension reduction with {}...'.format(METHOD))
        X_dimred = reduce_dimensionality(
            normalize(X), method=METHOD, dimred=DIMRED
        )
        log('Dimensionality = {}'.format(X_dimred.shape[1]))
        np.savetxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE))
        
    from ample import gs
    samp_idx = gs(X_dimred, 20000, replace=False)
    
    #from anndata import AnnData
    #import scanpy.api as sc
    #adata = AnnData(X=X_dimred[samp_idx, :])
    #sc.pp.neighbors(adata, use_rep='X')
    #sc.tl.louvain(adata, resolution=1.5, key_added='louvain')
    #
    #louv_labels = np.array(adata.obs['louvain'].tolist())
    #le = LabelEncoder().fit(louv_labels)
    #cell_labels = le.transform(louv_labels)
    #
    #np.savetxt('data/cell_labels/mouse_brain_louvain.txt', cell_labels)
    
    cell_labels = (
        open('data/cell_labels/mouse_brain_louvain.txt')
        .read().rstrip().split('\n')
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)

    astro = set([ 0, 3, 7, 31 ])
    oligo = set([ 2, 5, 22, 26, 27 ])
    focus = set([ 32 ])

    labels = []
    for cl in cell_labels:
        if cl in focus:
            labels.append(0)
        elif cl in astro or cl in oligo:
            labels.append(1)
        else:
            labels.append(2)
    labels = np.array(labels)

    X = np.log1p(normalize(X[samp_idx, :]))
    auroc(X.toarray(), genes, labels, 0, background=1)
    
    viz_genes = [
        'PLP1', 'MAL', 'PTGDS', 'MAG', 'CLDN11', 'APOD', 'FTH1',
        'ERMN', 'MBP', 'ENPP2', 'QDPR', 'MOBP', 'TRF',
        #'CST3', 'SPARCL1', 'PTN', 'CD81', 'APOE', 'ATP1A2', 'ITM2B'
    ]
    
    embedding = visualize(
        [ X_dimred[samp_idx, :] ], labels,
        NAMESPACE + '_astro{}'.format(len(samp_idx)),
        [ str(ct) for ct in sorted(set(labels)) ],
        gene_names=viz_genes, gene_expr=X, genes=genes,
        perplexity=100, n_iter=500, image_suffix='.png',
        viz_cluster=True
    )
