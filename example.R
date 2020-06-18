# WARNING: Sketch indices returned by geosketch are 0-indexed but R
# uses 1-indexing; indices should be shifted by 1 by setting
# `one_indexed = TRUE` for any downstream analysis.

library(reticulate)
library(rsvd)

geosketch <- import('geosketch')

# Generate some random data.
X <- replicate(20, rnorm(1000))

# Get top PCs from randomized SVD.
s <- rsvd(X, k=10)
X.pcs <- s$u %*% diag(s$d)

# Sketch 10% of data.
sketch.size <- as.integer(100)
sketch.indices <- geosketch$gs(X.pcs, sketch.size, one_indexed = TRUE)
print(sketch.indices)
