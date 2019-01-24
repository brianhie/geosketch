import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize, LabelEncoder

from experiments import *
from mouse_brain import keep_valid
from process import load_names
from utils import *

np.random.seed(0)

NAMESPACE = 'mouse_brain'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/mouse_brain/dropviz/Cerebellum_ALT',
    'data/mouse_brain/dropviz/Cortex_noRep5_FRONTALonly',
    'data/mouse_brain/dropviz/Cortex_noRep5_POSTERIORonly',
    'data/mouse_brain/dropviz/EntoPeduncular',
    'data/mouse_brain/dropviz/GlobusPallidus',
    'data/mouse_brain/dropviz/Hippocampus',
    'data/mouse_brain/dropviz/Striatum',
    'data/mouse_brain/dropviz/SubstantiaNigra',
    'data/mouse_brain/dropviz/Thalamus',
]

if __name__ == '__main__':
    #from process import process
    #process(data_names, min_trans=0)
    
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    qc_idx = keep_valid(datasets)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)
    X = X[qc_idx]

    if not os.path.isfile('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE)):
        log('Dimension reduction with {}...'.format(METHOD))
        X_dimred = reduce_dimensionality(
            normalize(X), method=METHOD, dimred=DIMRED
        )
        log('Dimensionality = {}'.format(X_dimred.shape[1]))
        np.savetxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE))
        
    labels = np.array(
        open('data/cell_labels/mouse_brain_cluster.txt')
        .read().rstrip().split('\n')
    )
    labels = labels[qc_idx]
    le = LabelEncoder().fit(labels)
    cell_names = sorted(set(labels))
    cell_labels = le.transform(labels)

    viz_genes = [
        'Nptxr', 'Calb1', 'Adora2a', 'Drd1', 'Nefm', 'C1ql2', 'Cck',
        'Rorb', 'Deptor', 'Gabra6',
        'Slc1a3', 'Gad1', 'Gad2', 'Slc17a6', 'Slc17a7', 'Th',
        'Pcp2', 'Sln', 'Lgi2'
    ]
    
    experiment_uni(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        viz_type='tsne', N_only=int(0.02 * X.shape[0]),
        kmeans=False, visualize_orig=False
    )
    experiment_gs(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        viz_type='tsne', N_only=int(0.02 * X.shape[0]),
        kmeans=False, visualize_orig=False
    )
    experiment_srs(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        viz_type='tsne', N_only=int(0.02 * X.shape[0]),
        kmeans=False, visualize_orig=False
    )
    experiment_kmeanspp(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        viz_type='tsne', N_only=int(0.02 * X.shape[0]),
        kmeans=False, visualize_orig=False
    )
    exit()
    experiment_uni(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        viz_type='umap', N_only=int(0.02 * X.shape[0]),
        gene_names=viz_genes, genes=genes, gene_expr=X,
        kmeans=False, visualize_orig=False
    )
    experiment_gs(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        viz_type='umap', N_only=int(0.02 * X.shape[0]),
        gene_names=viz_genes, genes=genes, gene_expr=X,
        kmeans=False, visualize_orig=False
    )
    experiment_srs(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        viz_type='umap', N_only=int(0.02 * X.shape[0]),
        kmeans=False, visualize_orig=False
    )
    experiment_kmeanspp(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        viz_type='umap', N_only=int(0.02 * X.shape[0]),
        kmeans=False, visualize_orig=False
    )
