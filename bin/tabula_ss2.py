import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, LabelEncoder

from experiments import *
from process import load_names
from utils import *

np.random.seed(0)

NAMESPACE = 'tabula_ss2'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/murine_atlases/tabula_ss2/Aorta',
    'data/murine_atlases/tabula_ss2/Bladder',
    'data/murine_atlases/tabula_ss2/Brain_Myeloid',
    'data/murine_atlases/tabula_ss2/Brain_Non-Myeloid',
    'data/murine_atlases/tabula_ss2/Diaphragm',
    'data/murine_atlases/tabula_ss2/Fat',
    'data/murine_atlases/tabula_ss2/Heart',
    'data/murine_atlases/tabula_ss2/Kidney',
    'data/murine_atlases/tabula_ss2/Large_Intestine',
    'data/murine_atlases/tabula_ss2/Limb_Muscle',
    'data/murine_atlases/tabula_ss2/Liver',
    'data/murine_atlases/tabula_ss2/Lung',
    'data/murine_atlases/tabula_ss2/Mammary_Gland',
    'data/murine_atlases/tabula_ss2/Marrow',
    'data/murine_atlases/tabula_ss2/Pancreas',
    'data/murine_atlases/tabula_ss2/Skin',
    'data/murine_atlases/tabula_ss2/Spleen',
    'data/murine_atlases/tabula_ss2/Thymus',
    'data/murine_atlases/tabula_ss2/Tongue',
    'data/murine_atlases/tabula_ss2/Trachea',
]

def load_cells(data_names):
    cells = []
    for dname in data_names:
        with open('{}.txt'.format(dname)) as f:
            cells += f.readline().rstrip().split()[1:]
    return cells

def keep_valid(cells):
    cell_to_type = {}
    cell_to_meta = {}
    with open('data/murine_atlases/tabula_ss2/annotations.csv') as f:
        for line in f:
            fields = line.split(',')
            cell_to_type[fields[2]] = fields[4] if fields[4] else 'Other'
            cell_to_meta[fields[2]] = '\t'.join(fields)

    cell_labels = []
    meta = []
    valid_idx = []
    for idx, cell in enumerate(cells):
        if cell in cell_to_type:
            cell_labels.append(cell_to_type[cell])
            meta.append(cell_to_meta[cell])
            valid_idx.append(idx)

    with open('data/cell_labels/tabula_ss2_cluster.txt', 'w') as of:
        for cl in cell_labels:
            of.write(cl + '\n')

    return valid_idx, meta

if __name__ == '__main__':
    #from process import process
    #process(data_names, min_trans=0)

    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)
    cells = load_cells(data_names)
    valid_idx, metas = keep_valid(cells)
    X = vstack(datasets)
    X = X[valid_idx, :]

    if not os.path.isfile('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE)):
        log('Dimension reduction with {}...'.format(METHOD))
        X_dimred = reduce_dimensionality(
            normalize(X), method=METHOD, dimred=DIMRED
        )
        log('Dimensionality = {}'.format(X_dimred.shape[1]))
        np.savetxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE))

    viz_genes = [
    ]

    cell_labels = (
        open('data/cell_labels/tabula_ss2_cluster.txt')
        .read().rstrip().split('\n')
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)

    experiments(
        X_dimred, NAMESPACE,
        entropy=True, cell_labels=cell_labels,
        max_min_dist=True
    )

    exit()

    balance(X_dimred, NAMESPACE, cell_labels, weights=s[:k])

    experiment_gs(X_dimred, NAMESPACE, cell_labels=cell_labels,
                  kmeans=False, visualize_orig=False)

    experiment_uni(X_dimred, NAMESPACE, cell_labels=cell_labels,
                   kmeans=False, visualize_orig=False)

    name = 'data/{}'.format(NAMESPACE)
    if not os.path.isfile('{}/matrix.mtx'.format(name)):
        from save_mtx import save_mtx
        save_mtx(name, csr_matrix(X), [ str(i) for i in range(X.shape[1]) ])

    experiment_dropclust(X_dimred, name, cell_labels)

    experiment_efficiency_kmeans(X_dimred, cell_labels)

    experiment_efficiency_louvain(X_dimred, cell_labels)

    log('Done.')
