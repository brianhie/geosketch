from scanorama import plt
plt.rcParams.update({'font.size': 25})
import seaborn as sns

sizes = [
    4185,
    68579,
    465281,
    665858,
]

times = [
    13.6908118724823,
    50.770941495895386,
    483.3363349437714,
    617.4005923271179,
]

plt.figure()
plt.plot([4185, 665858], [30, 630], '--')
plt.scatter(sizes, times)
plt.xticks(sizes, rotation=30)
plt.xlabel('Data set size')
plt.ylabel('Time (seconds)')
plt.savefig('svd.svg')
