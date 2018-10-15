import numpy as np
import os
from scanorama import visualize
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from subprocess import Popen
import sys

from experiments import *
from sketch import *
from utils import *

def dropclust_preprocess(name):
    os.chdir('../dropClust')

    rcode = Popen('Rscript dropClust_preprocess.R {} >> py.log'
                  .format(name), shell=True).wait()

    if rcode != 0:
        sys.stderr.write('ERROR: subprocess returned error code {}\n'
                         .format(rcode))
        exit(rcode)

    os.chdir('../ample')

def dropclust_sample(name, N):
    os.chdir('../dropClust')

    rcode = Popen('Rscript dropClust_sample.R {} {} >> py.log'
                  .format(name, N), shell=True).wait()

    if rcode != 0:
        sys.stderr.write('ERROR: subprocess returned error code {}\n'
                         .format(rcode))
        exit(rcode)

    with open('{}_dropclust{}.txt'.format(name, N)) as f:
        dropclust_idx = [
            int(idx) for idx in
            f.read().rstrip().split()
        ]

    os.chdir('../ample')

    return dropclust_idx

def experiment_dropclust(X_dimred, name, cell_labels=None,
                         kmeans=True,
                         gene_names=None, gene_expr=None, genes=None,
                         perplexity=500, kmeans_k=10):

    # Assign cells to clusters.

    if cell_labels is None and \
       (kmeans or \
        not os.path.isfile('data/cell_labels/{}.txt'.format(name))):
        log('K-means...')
        km = KMeans(n_clusters=kmeans_k, n_jobs=10, verbose=0)
        km.fit(X_dimred)
        np.savetxt('data/cell_labels/{}.txt'.format(name), km.labels_)

    if cell_labels is None:
        cell_labels = (
            open('data/cell_labels/{}.txt'.format(name))
            .read().rstrip().split()
        )
        le = LabelEncoder().fit(cell_labels)
        cell_labels = le.transform(cell_labels)
        cell_types = le.classes_
    else:
        cell_types = [ str(ct) for ct in sorted(set(cell_labels)) ]

    dropclust_preprocess(name)
        
    Ns = [ 1000, 5000, 10000, 20000, 50000 ]

    for N in Ns:
        log('dropClust {}...'.format(N))
        dropclust_idx = dropclust_sample(name, N)
        log('Found {} entries'.format(len(set(dropclust_idx))))

        log('Visualizing sampled...')

        if not gene_names is None and \
           not gene_expr is None and \
           not genes is None:
            expr = gene_expr[dropclust_idx, :]
        else:
            expr = None

        visualize([ X_dimred[dropclust_idx, :] ], cell_labels[dropclust_idx],
                  name + '_dropclust{}'.format(N), cell_types,
                  gene_names=gene_names, gene_expr=expr, genes=genes,
                  perplexity=max(N/200, 50), n_iter=500,
                  size=max(int(30000/N), 1), image_suffix='.png')

        report_cluster_counts(cell_labels[dropclust_idx])
