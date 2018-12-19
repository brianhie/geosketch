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
    qc_idx = keep_valid(datasets)
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

    X = X[qc_idx]
    X_dimred = X_dimred[qc_idx]
    
    labels = np.array(
        open('data/cell_labels/mouse_brain_subcluster.txt')
        .read().rstrip().split('\n')
    )
    labels = labels[qc_idx]
    le = LabelEncoder().fit(labels)
    cell_labels = le.transform(labels)
    
    from geosketch import gs, uniform, srs, kmeanspp
    
    samp_fns = [ gs, uniform, srs, kmeanspp ]
    samp_names = [ 'gs', 'uni', 'srs', 'kmeanspp' ]

    for samp_fn, samp_name in zip(samp_fns, samp_names):

        print(samp_name)
        
        samp_idx = samp_fn(X_dimred, 20000, replace=False)
        
        report_cluster_counts(cell_labels[samp_idx])
        
        embedding = visualize(
            [ X_dimred[samp_idx, :] ], cell_labels[samp_idx],
            NAMESPACE + '_subcluster_{}{}'.format(samp_name, len(samp_idx)),
            [ str(ct) for ct in sorted(set(labels)) ],
            perplexity=100, n_iter=500, image_suffix='.png',
            #viz_cluster=True
        )
