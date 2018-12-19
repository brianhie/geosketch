import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder, scale

from experiments import *
from process import load_names
from utils import *

NAMESPACE = 'hematopoeisis'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/pseudotime/GSE72857_umitab',
]

def write_table(dataset, genes, name):
    prefix = name.split('/')[-1]
    with open(name + '_table.txt', 'w') as f:
        header = '\t'.join([ prefix + str(i) for i in range(dataset.shape[0]) ])
        f.write(header + '\n')

        for i in range(dataset.shape[1]):
            line = '\t'.join([ str(int(j)) for j in dataset[:, i] ])
            f.write(genes[i] + '\t' + line + '\n')
            
def keep_valid():
    with open('data/pseudotime/GSE72857_umitab.txt') as f:
        all_cells = f.readline().rstrip().split()[1:]

    with open('data/pseudotime/meta.txt') as f:
        cell_to_type = {}
        for line in f:
            fields = line.rstrip().split()
            if len(fields) == 0:
                continue
            cell_to_type[fields[0]] = fields[1]

    valid_idx = []
    cell_names = []
    for c_idx, cell in enumerate(all_cells):
        if cell in cell_to_type:
            valid_idx.append(c_idx)
            cell_names.append(cell_to_type[cell])
            
    return valid_idx, cell_names
    
if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = datasets, genes_list[0]
    valid_idx, cell_names = keep_valid()
    print(len(valid_idx))
    X = vstack(datasets)[valid_idx]

    if not os.path.isfile('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE)):
        log('Dimension reduction with {}...'.format(METHOD))
        X_dimred = reduce_dimensionality(
            normalize(X), method=METHOD, dimred=DIMRED
        )
        log('Dimensionality = {}'.format(X_dimred.shape[1]))
        np.savetxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE))

    print(sorted(set(cell_names)))
        
    le = LabelEncoder().fit(cell_names)
    cell_labels = le.transform(cell_names)
    
    report_cluster_counts(cell_labels)
    
    write_table(X.toarray(), genes, 'data/pseudotime/' + NAMESPACE)

    with open('data/pseudotime/monocle_meta.txt', 'w') as of:
        of.write('Label\n')
        for idx in range(X.shape[0]):
            of.write('hematopoeisis{}\t{}'.format(idx, cell_names[idx]))
            
    from geosketch import *

    gs_idx = louvain1(X_dimred, 300, replace=True)
    write_table(X[gs_idx, :].toarray(), genes, 'data/pseudotime/' + NAMESPACE + '_gs')
    report_cluster_counts(cell_labels[gs_idx])

    with open('data/pseudotime/monocle_meta_gs.txt', 'w') as of:
        of.write('Label\n')
        i = 0
        for idx in range(X.shape[0]):
            if idx not in gs_idx:
                continue
            of.write('hematopoeisis{}\t{}'.format(i, cell_names[idx]))
            i += 1
    
    uni_idx = uniform(X_dimred, 300, replace=False)
    write_table(X[uni_idx, :].toarray(), genes, 'data/pseudotime/' + NAMESPACE + '_uni')
    report_cluster_counts(cell_labels[uni_idx])
    
    with open('data/pseudotime/monocle_meta_uni.txt', 'w') as of:
        of.write('Label\n')
        i = 0
        for idx in range(X.shape[0]):
            if idx not in uni_idx:
                continue
            of.write('hematopoeisis{}\t{}'.format(i, cell_names[idx]))
            i += 1

    with open('data/pseudotime/monocle_meta_genes.txt', 'w') as of:
        of.write('gene_short_name\n')
        for gene in genes:
            of.write('{}\t{}\n'.format(gene, gene))
