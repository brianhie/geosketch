library(Matrix)
library(Seurat)
library(cowplot)
library(dplyr)

args = commandArgs(trailingOnly=TRUE)

pdf(paste(c(args[1], ".pdf"), collapse=""))

# Read data.
X.data <- Read10X(data.dir = args[1])
X <- CreateSeuratObject(raw.data = X.data)

# Normalize.
X <- NormalizeData(object = X, normalization.method = "LogNormalize",
                   scale.factor = 10000)

# Highly variable genes.
X <- FindVariableGenes(object = X, mean.function = ExpMean,
                       dispersion.function = LogVMR, 
                       do.plot = FALSE)
hv.genes <- head(rownames(X@hvg.info), 1000)

# Regress out mitochondrial genes.
mito.genes <- grep(pattern = "^mt-", x = rownames(x = X@data), value = TRUE)
percent.mito <- (Matrix::colSums(X@raw.data[mito.genes, ]) /
                 Matrix::colSums(X@raw.data))
X <- AddMetaData(object = X, metadata = percent.mito, col.name = "percent.mito")
X <- ScaleData(object = X, genes.use = hv.genes, display.progress = FALSE, 
               vars.to.regress = "percent.mito", do.par = TRUE, num.cores = 4)

# PCA.
X <- RunPCA(object = X, pc.genes = hv.genes, pcs.compute = 100,
            do.print = FALSE, pcs.print = 1:5, genes.print = 5)

# Clustering.
X <- FindClusters(object = X, reduction.type = "pca", dims.use = 1:100,
                  resolution = 3., n.start = 10, nn.eps = 0.5,
                  print.output = FALSE)
write.table(X@ident, file = paste(c(args[1], "_labels.txt"), collapse=""),
            quote = FALSE, sep = "\t")

# Visualization.
#X <- RunTSNE(object = X, reduction.use = "pca", dims.use = 1:100,
#             nthreads = 4, max_iter = 2000, do.fast = TRUE)
#X <- RunUMAP(object = X, reduction.use = "pca", dims.use = 1:100,
#             min_dist = 0.75)
#p1 <- (DimPlot(object = X, reduction.use = "tsne", no.legend = TRUE,
#               do.return = TRUE, vector.friendly = TRUE) +
#       ggtitle("t-SNE") + theme(plot.title = element_text(hjust = 0.5)))
#p2 <- (DimPlot(object = X, reduction.use = "umap", no.legend = TRUE,
#               do.return = TRUE, vector.friendly = TRUE) +
#       ggtitle("UMAP") + theme(plot.title = element_text(hjust = 0.5)))
#plot_grid(p1, p2)
#
## Marker genes.
X.markers <- FindAllMarkers(object = X, only.pos = TRUE, min.pct = 0.25, 
                            thresh.use = 0.25)
X.markers %>% group_by(cluster) %>% top_n(10, avg_logFC) %>% print(n = Inf)
