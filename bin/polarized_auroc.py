import numpy as np
import seaborn as sns
from scanorama import *
from scipy.sparse import vstack
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize, LabelEncoder
import sys
from process import load_names

NAMESPACE = 'polarized'

data_names = [
    'data/macrophage/gmcsf_day6_1',
    'data/macrophage/gmcsf_day6_2',
    'data/macrophage/mcsf_day6_1',
    'data/macrophage/mcsf_day6_2',
]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, log1p=True)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred, genes = process_data(datasets, genes)
    
    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    labels[labels == 1] = 0
    labels[labels == 3] = 2
    labels[labels == 2] = 1

    X = np.concatenate([ X_i.toarray() for X_i in datasets ])
    X = normalize(X)

    for gidx, gene in enumerate(genes):
        case1 = X[labels == 0, gidx]
        case2 = X[labels == 1, gidx]

        fields = [ gene ]
        fields += ttest_ind(case1, case2, equal_var=False)
        fields += (np.mean(case1), np.mean(case2))

        fields.append(roc_auc_score(labels, X[:, gidx]))

        print('\t'.join([ str(f) for f in fields ]))

        
