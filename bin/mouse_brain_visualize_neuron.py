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

    if os.path.isfile('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE)):
        X_dimred = np.loadtxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE))
    else:
        log('Dimension reduction with {}...'.format(METHOD))
        X_dimred = reduce_dimensionality(
            normalize(X), method=METHOD, dimred=DIMRED
        )
        log('Dimensionality = {}'.format(X_dimred.shape[1]))
        np.savetxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
        
    labels = np.array(
        open('data/cell_labels/mouse_brain_cluster.txt')
        .read().rstrip().split('\n')
    )
    labels = labels[qc_idx]
    le = LabelEncoder().fit(labels)
    cell_names = sorted(set(labels))
    cell_labels = le.transform(labels)

    neuron_idx = cell_labels == le.transform(['Neuron'])[0]
    
    if os.path.isfile('data/cell_labels/mouse_brain_neuron_louv.txt'):
        louv_labels = np.array([
            int(ll) for ll in
            open('data/cell_labels/mouse_brain_neuron_louv.txt')
            .read().rstrip().split('\n')
        ])
    else:
        adata = AnnData(X=X_dimred[neuron_idx])
        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.louvain(adata, resolution=0.5, key_added='louvain')
        louv_labels = np.array(adata.obs['louvain'].tolist())
        le_louv = LabelEncoder().fit(louv_labels)
        louv_labels = le_louv.transform(louv_labels)
        with open('data/cell_labels/mouse_brain_neuron_louv.txt', 'w') as of:
            for ll in louv_labels:
                of.write(str(int(ll)) + '\n')

    cell_labels[neuron_idx] = louv_labels + len(set(cell_labels))

    for viz_type in [ 'umap' ]: #, 'tsne' ]:
        experiment_gs(
            X_dimred, NAMESPACE, cell_labels=cell_labels,
            viz_type=viz_type, N_only=20000, kmeans=False, visualize_orig=False
        )
        experiment_uni(
            X_dimred, NAMESPACE, cell_labels=cell_labels,
            viz_type=viz_type, N_only=20000, kmeans=False, visualize_orig=False
        )
        experiment_srs(
            X_dimred, NAMESPACE, cell_labels=cell_labels,
            viz_type=viz_type, N_only=20000, kmeans=False, visualize_orig=False
        )
        experiment_kmeanspp(
            X_dimred, NAMESPACE, cell_labels=cell_labels,
            viz_type=viz_type, N_only=20000, kmeans=False, visualize_orig=False
        )

    from umbilical import auroc
    X_csc = X[neuron_idx].tocsc()
    for ll in sorted(set(louv_labels)):
        print('Label {}'.format(ll))
        auroc(X_csc, genes, louv_labels, ll)
        print('')
