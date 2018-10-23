import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

namespaces = sys.argv[1:]

def plot_stat(dists, stat_fn):
    sample_sizes = None
    for dist_idx, dist in enumerate(dists):
        if sample_sizes is None:
            sample_sizes = sorted(dist.keys())
        else:
            assert(sorted(dist.keys()) == sample_sizes)

    plt.figure()
    for dist in dists:
        stats = []
        for sample_size in sample_sizes:
            clusters = sorted(dist[sample_size].keys())
            cluster_hist = np.zeros(max(clusters) + 1)
            for c in range(max(clusters) + 1):
                if c in clusters:
                    cluster_hist[c] = dist[sample_size][c]
            cluster_hist /= np.sum(cluster_hist)
            stats.append(stat_fn(cluster_hist * 100))
        plt.plot(sample_sizes, stats)
        plt.scatter(sample_sizes, stats)

def parse_log(fnames):
    dists = []
    for fname in fnames:
         with open(fname) as f:
             ssize_to_counts = {}
             n_sampled = None
             cluster_to_count = {}
             for line in f:
                 if '|' in line and ('Sampling' in line or 'dropClust' in line):
                     old_sampled = n_sampled
                     n_sampled = int(line.rstrip().split()[-1].rstrip('.'))
                     # Save cluster to count dict.
                     if old_sampled is not None:
                         ssize_to_counts[old_sampled] = cluster_to_count
                         
                         # New sampling strategy.
                         if old_sampled > n_sampled:
                             dists.append(ssize_to_counts)
                             ssize_to_counts = {}
                         
                     # Reset.
                     cluster_to_count = {}
                     
                 if 'has' in line and 'cells' in line and \
                    line.startswith('Cluster'):
                     fields = line.rstrip().split()
                     cluster_to_count[int(fields[1])] = int(fields[3])
     
             assert(n_sampled is not None)
             assert(n_sampled not in ssize_to_counts)
             ssize_to_counts[n_sampled] = cluster_to_count
             dists.append(ssize_to_counts)
        
    sample_sizes = None
    for dist_idx, dist in enumerate(dists):
        if sample_sizes is None:
            sample_sizes = sorted(dist.keys())
        else:
            assert(sorted(dist.keys()) == sample_sizes)

    for sample_size in sample_sizes:
        cluster_hists = []
        for dist in dists:
            clusters = sorted(dist[sample_size].keys())
            cluster_hist = np.zeros(max(clusters) + 1)
            for c in range(max(clusters) + 1):
                if c in clusters:
                    cluster_hist[c] = dist[sample_size][c]
            cluster_hist /= np.sum(cluster_hist)
            cluster_hists.append(cluster_hist)
            print('Median proportion = {}'
                  #.format(np.sum(cluster_hist != 0)))
                  .format(np.median(cluster_hist) * 100))
        print('Expected proportion = {}'
              .format(1. / (max(clusters) + 1) * 100))
        print('')


    plot_stat(dists, np.median)
    plt.title('Median')
    
    plot_stat(dists, np.std)
    plt.title('Standard deviation')
    
    plot_stat(dists, lambda x: np.sum(x != 0))
    plt.title('Number of clusters')
    
    plt.show()
            
    
if __name__ == '__main__':
    parse_log([ 'target/logs/{}.log'.format(namespace)
                for namespace in namespaces ])
