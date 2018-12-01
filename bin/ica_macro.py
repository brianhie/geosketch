from anndata import AnnData
from fbpca import pca
import numpy as np
from scanorama import *
import scanpy.api as sc
from scipy.sparse import vstack
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
    'data/ica/ica_bone_marrow_h5',
]

def auroc(X, genes, labels, focus, background=None, cutoff=0.7):
    assert(len(genes) == X.shape[1])

    focus_idx = focus == labels
    if background is None:
        background_idx = focus != labels
    else:
        background_idx = background == labels

    y_true = np.zeros(X.shape[0])
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
        
if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)

    for name, dataset, genes in zip(data_names, datasets, genes_list):

        name = name.split('/')[-1]

        X = normalize(dataset)

        k = DIMRED
        U, s, Vt = pca(X, k=k)
        X_dimred = U[:, :k] * s[:k]

        viz_genes = [
            'CD14', 'CD68',
            'S100A8', 'S100A9', 'P4B', 'HBB',
            'CD4', 'CD19', 'CD34', 'CD56', 'CD8'
        ]

        from ample import gs
        samp_idx = gs(X_dimred, 20000, replace=False)

        adata = AnnData(X=X_dimred[samp_idx, :])
        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.louvain(adata, resolution=1., key_added='louvain')
        louv_labels = np.array(adata.obs['louvain'].tolist())

        le = LabelEncoder().fit(louv_labels)
        cell_labels = le.transform(louv_labels)

        X_samp = X[samp_idx].tocsc()
        
        for label in sorted(set(louv_labels)):
            log('Label {}'.format(label))
            auroc(X_samp, genes, louv_labels, label)
            log('')

        embedding = visualize(
            [ X_dimred[samp_idx] ], cell_labels,
            name + '_louvain',
            [ str(ct) for ct in sorted(set(louv_labels)) ],
            gene_names=viz_genes, gene_expr=X_samp, genes=genes,
            perplexity=100, n_iter=500, image_suffix='.png',
            viz_cluster=True
        )
        #experiment_gs(
        #    X_dimred, name, cell_labels=cell_labels,
        #    gene_names=viz_genes, genes=genes, gene_expr=X,
        #    N_only=20000, kmeans=False, visualize_orig=False
        #)
        #experiment_uni(
        #    X_dimred, name, cell_labels=cell_labels,
        #    gene_names=viz_genes, genes=genes, gene_expr=X,
        #    N_only=20000, kmeans=False, visualize_orig=False
        #)
    
        
