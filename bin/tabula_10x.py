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

NAMESPACE = 'tabula_10x'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/murine_atlases/tabula_10x/Bladder-10X_P4_3',
    'data/murine_atlases/tabula_10x/Bladder-10X_P4_4',
    'data/murine_atlases/tabula_10x/Bladder-10X_P7_7',
    'data/murine_atlases/tabula_10x/Heart-10X_P7_4',
    'data/murine_atlases/tabula_10x/Kidney-10X_P4_5',
    'data/murine_atlases/tabula_10x/Kidney-10X_P4_6',
    'data/murine_atlases/tabula_10x/Kidney-10X_P7_5',
    'data/murine_atlases/tabula_10x/Liver-10X_P4_2',
    'data/murine_atlases/tabula_10x/Liver-10X_P7_0',
    'data/murine_atlases/tabula_10x/Liver-10X_P7_1',
    'data/murine_atlases/tabula_10x/Lung-10X_P7_8',
    'data/murine_atlases/tabula_10x/Lung-10X_P7_9',
    'data/murine_atlases/tabula_10x/Lung-10X_P8_12',
    'data/murine_atlases/tabula_10x/Lung-10X_P8_13',
    'data/murine_atlases/tabula_10x/Mammary-10X_P7_12',
    'data/murine_atlases/tabula_10x/Mammary-10X_P7_13',
    'data/murine_atlases/tabula_10x/Marrow-10X_P7_2',
    'data/murine_atlases/tabula_10x/Marrow-10X_P7_3',
    'data/murine_atlases/tabula_10x/Muscle-10X_P7_14',
    'data/murine_atlases/tabula_10x/Muscle-10X_P7_15',
    'data/murine_atlases/tabula_10x/Spleen-10X_P4_7',
    'data/murine_atlases/tabula_10x/Spleen-10X_P7_6',
    'data/murine_atlases/tabula_10x/Thymus-10X_P7_11',
    'data/murine_atlases/tabula_10x/Tongue-10X_P4_0',
    'data/murine_atlases/tabula_10x/Tongue-10X_P4_1',
    'data/murine_atlases/tabula_10x/Tongue-10X_P7_10',
    'data/murine_atlases/tabula_10x/Trachea-10X_P8_14',
    'data/murine_atlases/tabula_10x/Trachea-10X_P8_15',
]

def load_cells(data_names):
    cells = []
    for dname in data_names:
        prefix = dname.split('/')[-1].split('-')[-1]
        with open('{}/barcodes.tsv'.format(dname)) as f:
            cells += [
                prefix + '_' + c[:-2] for c in f.read().rstrip().split()
            ]
    return cells

def keep_valid(cells):
    cell_to_type = {}
    cell_to_meta = {}
    with open('data/murine_atlases/tabula_10x/annotations.csv') as f:
        for line in f:
            fields = line.split(',')
            cell_to_type[fields[0]] = fields[2]
            cell_to_meta[fields[0]] = '\t'.join(fields)

    cell_labels = []
    meta = []
    valid_idx = []
    for idx, cell in enumerate(cells):
        if cell in cell_to_type:
            cell_labels.append(cell_to_type[cell])
            meta.append(cell_to_meta[cell])
            valid_idx.append(idx)

    with open('data/cell_labels/tabula_10x_cluster.txt', 'w') as of:
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
        open('data/cell_labels/tabula_10x_cluster.txt')
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
                  kmeans=False, visualize_orig=False, weights=s[:k])

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
