import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
import seaborn as sns
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

def astro_oligo_violin(X, genes, gene, labels, name):
    X = X.toarray()

    gidx = list(genes).index(gene)

    astro = X[labels == 'astro', gidx]
    oligo = X[labels == 'oligo', gidx]
    both = X[labels == 'both', gidx]

    plt.figure()
    sns.violinplot(data=[ astro, oligo, both ], scale='width', cut=0)
    sns.stripplot(data=[ astro, oligo, both ], jitter=True, color='black', size=1)
    plt.xticks([0, 1, 2], ['Astrocytes', 'Oligodendrocytes', 'Both'])
    plt.savefig('{}_violin_{}.svg'.format(name, gene))
        
def astro_oligo_joint(X, genes, gene1, gene2, labels, focus, name):
    X = X.toarray()

    gidx1 = list(genes).index(gene1)
    gidx2 = list(genes).index(gene2)

    idx = labels == focus

    x1 = X[(idx, gidx1)]
    x2 = X[(idx, gidx2)]

    plt.figure()
    sns.jointplot(
        x1, x2, kind='scatter', space=0, alpha=0.3
    ).plot_joint(sns.kdeplot, zorder=0, n_levels=10)
    plt.savefig('{}_joint_{}_{}_{}.png'.format(name, focus, gene1, gene2))
    
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
        
    from geosketch import gs, uniform
    samp_idx = gs(X_dimred, 20000, replace=False)
    #samp_idx = uniform(X_dimred, 20000, replace=False)
    
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

    astro = set([ 1, 4, 5, 7 ])
    oligo = set([ 8, 21, 23, 26, 27 ])
    focus = set([ 2 ])

    labels = []
    aob_labels = []
    for cl in cell_labels:
        if cl in focus:
            labels.append(2)
            aob_labels.append('both')
        elif cl in astro or cl in oligo:
            if cl in astro:
                if cl == 5:
                    aob_labels.append('bergman')
                    labels.append(3)
                else:
                    aob_labels.append('astro')
                    labels.append(0)
            else:
                aob_labels.append('oligo')
                labels.append(1)
        else:
            labels.append(4)
            aob_labels.append('none')
    labels = np.array(labels)
    aob_labels = np.array(aob_labels)

    X = np.log1p(normalize(X[samp_idx, :]))
    
    #auroc(X.toarray(), genes, labels, 0, background=1)
    #astro_oligo_joint(X, genes, 'GJA1', 'MBP', aob_labels, 'astro', NAMESPACE)
    #astro_oligo_joint(X, genes, 'GJA1', 'MBP', aob_labels, 'oligo', NAMESPACE)
    #astro_oligo_joint(X, genes, 'GJA1', 'MBP', aob_labels, 'both', NAMESPACE)
    #astro_oligo_joint(X, genes, 'GJA1', 'PLP1', aob_labels, 'astro', NAMESPACE)
    #astro_oligo_joint(X, genes, 'GJA1', 'PLP1', aob_labels, 'oligo', NAMESPACE)
    #astro_oligo_joint(X, genes, 'GJA1', 'PLP1', aob_labels, 'both', NAMESPACE)
    
    astro_oligo_violin(X, genes, 'GJA1', aob_labels, NAMESPACE)
    astro_oligo_violin(X, genes, 'MBP', aob_labels, NAMESPACE)
    astro_oligo_violin(X, genes, 'PLP1', aob_labels, NAMESPACE)

    viz_genes = [
        'SLC1A3', 'GJA1', 'MBP', 'PLP1', 'TRF',
        #'CST3', 'CPE', 'FTH1', 'APOE', 'MT1', 'NDRG2', 'TSPAN7',
        #'PLP1', 'MAL', 'PTGDS', 'CLDN11', 'APOD', 'QDPR', 'MAG', 'ERMN',
        #'PLP1', 'MAL', 'PTGDS', 'MAG', 'CLDN11', 'APOD', 'FTH1',
        #'ERMN', 'MBP', 'ENPP2', 'QDPR', 'MOBP', 'TRF',
        #'CST3', 'SPARCL1', 'PTN', 'CD81', 'APOE', 'ATP1A2', 'ITM2B'
    ]
    
    cell_labels = (
        open('data/cell_labels/mouse_brain_cluster.txt')
        .read().rstrip().split('\n')
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)
    
    embedding = visualize(
        [ X_dimred[samp_idx, :] ], labels,
        NAMESPACE + '_astro{}'.format(len(samp_idx)),
        [ str(ct) for ct in sorted(set(labels)) ],
        gene_names=viz_genes, gene_expr=X, genes=genes,
        perplexity=100, n_iter=500, image_suffix='.png',
        viz_cluster=True
    )

    #visualize_dropout(X, embedding, image_suffix='.png',
    #                  viz_prefix=NAMESPACE + '_dropout')
