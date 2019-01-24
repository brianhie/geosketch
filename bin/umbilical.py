from anndata import AnnData
from fbpca import pca
import numpy as np
from scanorama import *
import scanpy.api as sc
from scipy.sparse import vstack
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, scale

from process import load_names
from experiments import *
from utils import *

np.random.seed(0)

NAMESPACE = 'ica_macro'
METHOD = 'svd'
DIMRED = 30

data_names = [
    'data/ica/ica_cord_blood_h5',
]

def auroc(X, genes, labels, focus, background=None, cutoff=0.7):
    assert(len(genes) == X.shape[1])

    focus_idx = focus == labels
    if background is None:
        background_idx = focus != labels
    else:
        background_idx = background == labels

    y_true = np.zeros(sum(focus_idx) + sum(background_idx))
    y_true[:sum(focus_idx)] = 1

    data = []

    for g, gene in enumerate(genes):
        x_gene = X[:, g].toarray().flatten()
        x_focus = x_gene[focus_idx]
        x_background = x_gene[background_idx]
        y_score = np.concatenate([ x_focus, x_background ])
        
        auroc = roc_auc_score(y_true, y_score)

        data.append((auroc, gene))

    for auroc, gene in sorted(data):
        if auroc >= cutoff:
            print('{}\t{}'.format(gene, auroc))
        
def violin_jitter(X, genes, gene, labels, focus, background=None,
                  xlabels=None):
    gidx = list(genes).index(gene)

    focus_idx = focus == labels
    if background is None:
        background_idx = focus != labels
    else:
        background_idx = background == labels

    if xlabels is None:
        xlabels = [ 'Background', 'Focus' ]

    x_gene = X[:, gidx].toarray().flatten()
    x_focus = x_gene[focus_idx]
    x_background = x_gene[background_idx]
    
    plt.figure()
    sns.violinplot(data=[ x_focus, x_background ], scale='width', cut=0)
    sns.stripplot(data=[ x_focus, x_background ], jitter=True, color='black', size=1)
    plt.xticks([0, 1], xlabels)
    plt.savefig('{}_violin_{}.png'.format(NAMESPACE, gene))
    
if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)

    for name, dataset, genes in zip(data_names, datasets, genes_list):

        name = name.split('/')[-1]

        X = normalize(dataset)

        filter_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
                       if s >= 500 ]
        X = X[filter_idx]
        
        k = DIMRED
        U, s, Vt = pca(X, k=k)
        X_dimred = U[:, :k] * s[:k]

        viz_genes = [
            'CD74', 'JUNB', 'B2M',
            'CD14', 'CD68',
            #'PF4',
            #'HBB',
            #'CD19',
        ]

        from geosketch import gs, uniform
        samp_idx = gs(X_dimred, 20000, replace=False)

        adata = AnnData(X=X_dimred[samp_idx, :])
        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.louvain(adata, key_added='louvain')
        louv_labels = np.array(adata.obs['louvain'].tolist())

        le = LabelEncoder().fit(louv_labels)
        cell_labels = le.transform(louv_labels)

        X_samp = X[samp_idx].tocsc()
        
        #sc.tl.umap(adata)
        #umap_embedding = np.array(adata.obsm['X_umap'])
        #visualize(
        #    None, cell_labels, name + '_umap_louvain',
        #    [ str(ct) for ct in sorted(set(louv_labels)) ],
        #    gene_names=viz_genes, gene_expr=X_samp, genes=genes,
        #    embedding=umap_embedding, image_suffix='.png',
        #    viz_cluster=True
        #)

        cache_fname = 'data/embedding/umbilical_tsne.txt'
        if os.path.isfile(cache_fname):
            tsne_embedding = np.loadtxt(cache_fname)
            visualize(
                None, cell_labels, name + '_louvain',
                [ str(ct) for ct in sorted(set(louv_labels)) ],
                gene_names=viz_genes, gene_expr=X_samp, genes=genes,
                embedding=tsne_embedding, size=5,
                image_suffix='.png', #viz_cluster=True
            )
        else:
            tsne_embedding = visualize(
                [ X_dimred[samp_idx] ], cell_labels, name + '_louvain',
                [ str(ct) for ct in sorted(set(louv_labels)) ],
                gene_names=viz_genes, gene_expr=X_samp, genes=genes,
                size=5, image_suffix='.png', #viz_cluster=True
            )
            np.savetxt(cache_fname, tsne_embedding)

        #visualize_dropout(X_samp, umap_embedding, image_suffix='.png',
        #                  viz_prefix=name + '_umap_louvain_dropout')
        visualize_dropout(X_samp, tsne_embedding, image_suffix='.png',
                          viz_prefix=name + '_louvain_dropout')
        
        clusterA = set([ 20 ])
        clusterB = set([ 12 ])
        
        labels = []
        for cl in cell_labels:
            if cl in clusterA:
                labels.append(0)
            elif cl in clusterB:
                labels.append(1)
            else:
                labels.append(2)
        labels = np.array(labels)

        xlabels = [ 'Inflammatory macrophage', 'Macrophage' ]
        violin_jitter(X_samp, genes, 'CD14', labels, 0, 1, xlabels)
        violin_jitter(X_samp, genes, 'CD68', labels, 0, 1, xlabels)
        violin_jitter(X_samp, genes, 'CD74', labels, 0, 1, xlabels)
        violin_jitter(X_samp, genes, 'B2M', labels, 0, 1, xlabels)
        violin_jitter(X_samp, genes, 'JUNB', labels, 0, 1, xlabels)
        violin_jitter(X_samp, genes, 'HLA-DRA', labels, 0, 1, xlabels)
        
        auroc(X_samp, genes, np.array(labels), 0, background=1)
        
        #for label in sorted(set(louv_labels)):
        #    log('Label {}'.format(label))
        #    auroc(X_samp, genes, louv_labels, label)
        #    log('')

