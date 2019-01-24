# Geometric sketching

## Overview

`geosketch` is a Python package that implements the geometric sketching algorithm described by Brian Hie, Hyunghoon Cho, Benjaming DeMeo, Bryan Bryson, and Bonnie Berger (2019). This repository contains an example implementation of the algorithm as well as scripts necessary for reproducing the experiments in the paper.

## Installation

You should be able to install from PyPI:
```
pip install geosketch
```

## API example usage

**Parameter documentation for the geometric sketching `gs()` function is in the source code at the top of [`geosketch/sketch.py`](geosketch/sketch.py).**

Here is example usage of `geosketch` in Python. First, put your data set into a matrix:
```
X = [ sparse or dense matrix, samples in rows, features in columns ]
```

Then, compute the top PCs:
```
# Compute PCs.
from fbpca import pca
U, s, Vt = pca(X, k=100) # E.g., 100 PCs.
X_dimred = U[:, :100] * s[:100]
```

Now, you are ready to sketch!
```
# Sketch.
from geosketch import gs
N = 20000 # Number of samples to obtain from the data set.
sketch_index = gs(X_dimred, N, replace=False)

X_sketch = X_dimred[sketch_index]
```

## Examples

### Data set download

All of the data used in our study can be downloaded from http://geosketch.csail.mit.edu/data.tar.gz. Download and unpack this data with the command:

```
wget http://geosketch.csail.mit.edu/data.tar.gz
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

For questions about the pipeline and code, contact brianhie@mit.edu and hhcho@mit.edu. We will do our best to provide support, address any issues, and keep improving this software. And do not hesitate to submit a pull request and contribute!
