import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize, LabelEncoder

from geosketch import *
from experiments import *
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

def keep_valid(datasets):
    n_valid = 0
    qc_idx = []
    for i in range(len(datasets)):
        
        valid_idx = []
        with open('{}/meta.txt'.format(data_names[i])) as f:
            
            n_lines = 0
            for j, line in enumerate(f):
                
                fields = line.rstrip().split()
                if fields[1] != 'NA':
                    valid_idx.append(j)
                    if fields[3] != 'doublet' and \
                       fields[3] != 'outlier':
                        qc_idx.append(n_valid)
                    n_valid += 1
                n_lines += 1

        assert(n_lines == datasets[i].shape[0])
        assert(len(qc_idx) <= n_valid)

        datasets[i] = datasets[i][valid_idx, :]
        print('{} has {} valid cells'
              .format(data_names[i], len(valid_idx)))

    print('Found {} cells among all datasets'.format(n_valid))
    print('Found {} valid cells among all datasets'.format(len(qc_idx)))
    
    return qc_idx
    
if __name__ == '__main__':
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
        
    #X_dimred = X_dimred[qc_idx]
    
    viz_genes = [
        'Nptxr', 'Calb1', 'Adora2a', 'Drd1', 'Nefm', 'C1ql2', 'Cck',
        'Rorb', 'Deptor', 'Gabra6',
        'Slc1a3', 'Gad1', 'Gad2', 'Slc17a6', 'Slc17a7', 'Th',
        'Pcp2', 'Sln', 'Lgi2'
        #'Gja1', 'Flt1', 'Gabra6', 'Syt1', 'Gabrb2', 'Gabra1',
        #'Meg3', 'Mbp', 'Rgs5', 'Pcp2', 'Dcn', 'Pvalb', 'Nnat',
        #'C1qb', 'Acta2', 'Syt6', 'Lhx1', 'Sox4', 'Tshz2', 'Cplx3',
        #'Shisa8', 'Fibcd1', 'Drd1', 'Otof', 'Chat', 'Th', 'Rora',
        #'Synpr', 'Cacng4', 'Ttr', 'Gpr37', 'C1ql3', 'Fezf2',
    ]

    labels = np.array(
        open('data/cell_labels/mouse_brain_cluster.txt')
        .read().rstrip().split('\n')
    )
    labels = labels[qc_idx]
    le = LabelEncoder().fit(labels)
    cell_names = sorted(set(labels))
    cell_labels = le.transform(labels)

    experiments(
        X_dimred, NAMESPACE, n_seeds=4,
        #cell_labels=cell_labels,
        #cell_exp_ratio=True,
        #louvain_ami=True,
        #rare=True,
        #rare_label=le.transform(['Macrophage'])[0],
        max_min_dist=True,
    )
    exit()
    
    report_cluster_counts(labels)
    
    from differential_entropies import differential_entropies
    differential_entropies(X_dimred, labels)
    
    plot_rare(X_dimred, cell_labels, le.transform(['Macrophage'])[0],
              NAMESPACE, n_seeds=4)
    
    from geosketch import gs
    samp_idx = gs(X_dimred, 1000, replace=False)
    save_sketch(X, samp_idx, genes, NAMESPACE + '1000')
    
    for scale in [ 10, 25, 100 ]:
        N = int(X.shape[0] / scale)
        samp_idx = gs(X_dimred, N, replace=False)
        save_sketch(X, samp_idx, genes, NAMESPACE + str(N))
    

