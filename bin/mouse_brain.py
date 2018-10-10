import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, LabelEncoder

from process import load_names
from experiments import *
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

def keep_valid(datasets):
    n_valid = 0
    for i in range(len(datasets)):
        valid_idx = []
        with open('{}/meta.txt'.format(data_names[i])) as f:
            n_lines = 0
            for j, line in enumerate(f):
                fields = line.rstrip().split()
                if fields[1] != 'NA':
                    valid_idx.append(j)
                n_lines += 1
        assert(n_lines == datasets[i].shape[0])
        datasets[i] = datasets[i][valid_idx, :]
        print('{} has {} valid cells'
              .format(data_names[i], len(valid_idx)))
        n_valid += len(valid_idx)
    print('Found {} valid cells among all datasets'.format(n_valid))

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names)
    keep_valid(datasets)
    datasets, genes = merge_datasets(datasets, genes_list)

    if not os.path.isfile('data/dimred_{}.txt'.format(NAMESPACE)):
        datasets_dimred, genes = process_data(datasets, genes)
        if not 'raw' in NAMESPACE:
            log('Scanorama integration...')
            datasets_dimred = assemble(datasets_dimred, sigma=50,
                                       batch_size=25000)
        X_dimred = np.concatenate(datasets_dimred)
        np.savetxt('data/dimred_{}.txt'.format(NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred_{}.txt'.format(NAMESPACE))
        
    viz_genes = [
        'Gja1', 'Flt1', 'Gabra6', 'Syt1', 'Gabrb2', 'Gabra1',
        'Meg3', 'Mbp', 'Rgs5', 'Pcp2', 'Dcn', 'Pvalb', 'Nnat',
        'C1qb', 'Acta2', 'Syt6', 'Lhx1', 'Sox4', 'Tshz2', 'Cplx3',
        'Shisa8', 'Fibcd1', 'Drd1', 'Otof', 'Chat', 'Th', 'Rora',
        'Synpr', 'Cacng4', 'Ttr', 'Gpr37', 'C1ql3', 'Fezf2',
    ]

    cell_labels = (
        open('data/cell_labels/mouse_brain_cluster.txt')
        .read().rstrip().split('\n')
    )
    le = LabelEncoder().fit(cell_labels)
    cell_labels = le.transform(cell_labels)

    experiment_efficiency_louvain(X_dimred, cell_labels)

    experiment_efficiency_kmeans(X_dimred, cell_labels)

    experiment_srs(X_dimred, NAMESPACE, cell_labels=cell_labels,
                   perplexity=1200, gene_names=viz_genes,
                   genes=genes, gene_expr=vstack(datasets),
                   kmeans=False)

    log('Done.')
