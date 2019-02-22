import numpy as np
import sys
from scanorama import plt
plt.rcParams.update({'font.size': 20})
import scipy.stats as ss

from utils import mkdir_p

def parse_stats(fname):
    with open(fname) as f:
        samp_fns = {}
        for line in f:
            fields = line.rstrip().split('\t')
    
            if fields[0] == 'name':
                stat_names = fields[4:]
                continue

            if not ('_namespace', None) in samp_fns:
                samp_fns[('_namespace', None)] = fields[0]
            
            samp_fn = fields[1]
            if samp_fn == 'gs_gap_N':
                samp_fn = 'a_' + samp_fn
            if samp_fn == 'uniform':
                samp_fn = 'b_' + samp_fn
            
            replace = fields[2] == 'True'
            if (samp_fn, replace) not in samp_fns:
                samp_fns[(samp_fn, replace)] = {}
    
            N = float(fields[3])
            if N not in samp_fns[(samp_fn, replace)]:
                samp_fns[(samp_fn, replace)][N] = []
    
            stat = { stat_names[i]: fields[4 + i]
                     for i in range(len(stat_names)) }
                
            samp_fns[(samp_fn, replace)][N].append(stat)
            
    return samp_fns

def plot_stats(stat, samp_fns=None, fname=None, dtype=float,
               only_fns=None, only_replace=None, max_N=None):
    if samp_fns is None:
        assert(fname is not None)
        samp_fns = parse_stats(fname)
        
    colors = [
        #'#377eb8', '#ff7f00', '#f781bf',
        #'#4daf4a', '#ff0000', '#a65628', '#984ea3',
        #'#999999', '#e41a1c', '#dede00',
        #'#ffe119', '#e6194b', '#ffbea3',
        #'#911eb4', '#46f0f0', '#f032e6',
        #'#d2f53c', '#008080', '#e6beff',
        #'#aa6e28', '#800000', '#aaffc3',
        #'#808000', '#ffd8b1', '#000080',
        #'#808080', '#fabebe', '#a3f4ff'

         '#ff7f00', #'#f781bf', '#4c4c4c', 
        
        #'#377eb8', '#ff7f00', #'#4daf4a',
        #'#984ea3',
        #'#f781bf', '#4c4c4c', '#a65628', '#984ea3',
        #'#999999', '#e41a1c', '#dede00',
        '#ffe119', '#e6194b', '#377eb8','#ffbea3',
        '#911eb4', '#46f0f0', '#f032e6',
        '#d2f53c', '#008080', '#e6beff',
        '#aa6e28', '#800000', '#aaffc3',
        '#808000', '#ffd8b1', '#000080',
        '#808080', '#fabebe', '#a3f4ff'
    ]

    plt.figure()

    c_idx = 0
    
    for s_idx, (samp_fn, replace) in enumerate(
            sorted(samp_fns, key=lambda x: '{}_{}'.format(*x))[::-1]):

        if samp_fn.startswith('_'):
            continue
        if only_fns is not None and samp_fn not in only_fns:
            continue
        if only_replace is not None and replace != only_replace:
            continue

        Ns = []
        means = []
        sems = []
        for N in samp_fns[(samp_fn, replace)]:
            if max_N is not None and N > max_N:
                continue
            stat_vals = [ dtype(stat_dict[stat])
                          for stat_dict in samp_fns[(samp_fn, replace)][N]
                          if stat in stat_dict ]
            if len(stat_vals) == 0:
                continue
            Ns.append(N)
            means.append(np.mean(stat_vals))
            sems.append(ss.sem(stat_vals))
            
        sort_idx = np.argsort(Ns)
        Ns = np.array(Ns)[sort_idx]
        means = np.array(means)[sort_idx]
        sems = np.array(sems)[sort_idx]

        label = '{}_{}'.format(samp_fn, replace)

        plt.plot(Ns, means, color=colors[c_idx], label=label, linewidth=4.)
        plt.scatter(Ns, means, color=colors[c_idx])
        plt.fill_between(Ns, means - sems, means + sems, alpha=0.3,
                         color=colors[c_idx])

        c_idx = (c_idx + 1) % len(colors)

    namespace = samp_fns[('_namespace', None)]
    title = '{}_{}'.format(namespace, stat)
    if only_replace is not None:
        title += '_replace{}'.format(only_replace)
        
    plt.title(title)
    plt.xlabel('Sample size')
    plt.ylabel(stat)
    plt.yscale('log')
    plt.legend()
    mkdir_p('target/stats_plots')
    plt.savefig('target/stats_plots/{}.svg'.format(title))
    
if __name__ == '__main__':
    only_fns = set([
        #'gs_gap_k',
        'a_gs_gap_N',
        #'b_uniform',
        #'gs_grid',
        #'gs_gap',
        #'srs',
        #'pc_pick',
        'louvain1',
        'louvain3',
        #'kmeans++',
        'kmeans++',
    ])
    
    plot_stats(sys.argv[1], fname=sys.argv[2],
               only_replace=len(sys.argv) > 3, only_fns=only_fns)
