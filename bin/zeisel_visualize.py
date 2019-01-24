from anndata import AnnData
import numpy as np
import os
from scanorama import *
import scanpy.api as sc
from scipy.sparse import vstack

from process import load_names
from experiments import *
from utils import *

NAMESPACE = 'zeisel'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/mouse_brain/zeisel/amygdala',
    'data/mouse_brain/zeisel/cerebellum',
    'data/mouse_brain/zeisel/cortex1',
    'data/mouse_brain/zeisel/cortex2',
    'data/mouse_brain/zeisel/cortex3',
    'data/mouse_brain/zeisel/enteric',
    'data/mouse_brain/zeisel/hippocampus',
    'data/mouse_brain/zeisel/hypothalamus',
    'data/mouse_brain/zeisel/medulla',
    'data/mouse_brain/zeisel/midbraindorsal',
    'data/mouse_brain/zeisel/midbrainventral',
    'data/mouse_brain/zeisel/olfactory',
    'data/mouse_brain/zeisel/pons',
    'data/mouse_brain/zeisel/spinalcord',
    'data/mouse_brain/zeisel/striatumdorsal',
    'data/mouse_brain/zeisel/striatumventral',
    'data/mouse_brain/zeisel/sympathetic',
    'data/mouse_brain/zeisel/thalamus',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)
    qc_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1)) if s >= 500 ]
    print('Found {} valid cells among all datasets'.format(len(qc_idx)))
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
        open('data/cell_labels/zeisel_cluster.txt')
        .read().rstrip().split('\n')
    )
    labels = labels[qc_idx]
    le = LabelEncoder().fit(labels)
    cell_labels = le.transform(labels)

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
    experiment_gs(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        viz_type='umap', N_only=int(0.02 * X.shape[0]),
        kmeans=False, visualize_orig=False
    )
    experiment_uni(
        X_dimred, NAMESPACE, cell_labels=cell_labels,
        viz_type='umap', N_only=int(0.02 * X.shape[0]),
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
