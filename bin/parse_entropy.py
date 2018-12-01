import fileinput
from scanorama import plt
plt.rcParams.update({'font.size': 22})
import sys

methods = [
    'geosketch',
    'harmony',
    'scanorama',
    'uncorrected'
]
#methods = [
#    'geosketch',
#    'uniform',
#    'srs',
#    'kmeanspp',
#]
data = {}

curr_method = None
ks = []
entropies = []

f = open(sys.argv[1])

for line in fileinput.input():
    line = line.rstrip()
    fields = line.split()

    if len(fields) == 1:
        if len(ks) > 0:
            data[curr_method] = entropies
            ks = []
            entropies = []
        curr_method = line
        continue
    
    k = int(fields[2].rstrip(','))
    if k > 15:
        continue
    
    ks.append(k)
    entropies.append(float(fields[-1]))
    
data[curr_method] = entropies

plt.figure()

for method in methods:
    if method == 'geosketch':
        label = 'Scanorama + GeoSketch'
    else:
        label = method.capitalize()
        #label = 'Scanorama + ' + method.capitalize()
    plt.plot(ks, data[method], label=label)
    plt.scatter(ks, data[method])

plt.legend()
plt.xlabel('k-means, number of clusters')
plt.ylabel('Normalized Shannon entropy')
plt.ylim([0, 1.05])
plt.savefig('entropies.svg')
