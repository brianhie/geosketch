# Geometric sketching

## Overview

`geosketch` is a Python package that implements the geometric sketching algorithm described by Brian Hie, Hyunghoon Cho, Benjamin DeMeo, Bryan Bryson, and Bonnie Berger in ["Geometric sketching compactly summarizes the single-cell transcriptomic landscape", Cell Systems (2019)](https://www.cell.com/cell-systems/fulltext/S2405-4712\(19\)30152-8). This repository contains an example implementation of the algorithm as well as scripts necessary for reproducing the experiments in the paper.

## Installation

You should be able to install from PyPI:
```
pip install geosketch
```

## API example usage

**Parameter documentation for the geometric sketching `gs()` function is in the source code at the top of [`geosketch/sketch.py`](geosketch/sketch.py).**

**For an example of usage of `geosketch` in R using the [`reticulate`](https://rstudio.github.io/reticulate/) library, see  [`example.R`](example.R). WARNING: The indices returned by `geosketch` are 0-indexed, but R uses 1-indexing, so the `one_indexed` parameter should be set to `TRUE` when called from R.**

Here is example usage of `geosketch` in Python. First, put your data set into a matrix:
```
X = [ sparse or dense matrix, samples in rows, features in columns ]
```

Then, compute the top PCs:
```Python
# Compute PCs.
from fbpca import pca
U, s, Vt = pca(X, k=100) # E.g., 100 PCs.
X_dimred = U[:, :100] * s[:100]
```

Now, you are ready to sketch!
```Python
# Sketch.
from geosketch import gs
N = 20000 # Number of samples to obtain from the data set.
sketch_index = gs(X_dimred, N, replace=False)

X_sketch = X_dimred[sketch_index]
```

## Examples

### Data set download

All of the data used in our study can be downloaded from http://cb.csail.mit.edu/cb/geosketch/data.tar.gz. Download and unpack this data with the command:

```
wget http://cb.csail.mit.edu/cb/geosketch/data.tar.gz
tar xvf data.tar.gz
```

### Visualizing sketches of a mouse brain data set

We can visualize a large data set of cells from different regions of the mouse brain collected by [Saunders et al. (2018)](http://dropviz.org/).

To visualize the sketches obtained by geometric sketching and other baseline algorithms, download the data using the commands above and then run:
```
python bin/mouse_brain_visualize.py
```
This will output PNG files to the top level directory visualizing different sketches produced by different algorithms, including geometric sketching.

## Algorithm implementation

For those interested, the algorithm implementation is available in the file [`geosketch/sketch.py`](geosketch/sketch.py).

## Questions

For questions, please use the [GitHub Discussions](https://github.com/brianhie/geosketch/discussions) forum. For bugs or other problems, please file an [issue](https://github.com/brianhie/geosketch/issues).
