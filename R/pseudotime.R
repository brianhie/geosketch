rm(list = ls())  # Clear the environment
options(warn=-1) # Turn off warning message globally
library(monocle) # Load Monocle

cds <- readRDS(gzcon(url("http://trapnell-lab.gs.washington.edu/public_share/valid_subset_GSE72857_cds2.RDS")))

# Update the old CDS object to be compatible with Monocle 3
cds <- updateCDS(cds)

pData(cds)$cell_type2 <- plyr::revalue(as.character(pData(cds)$cluster),
                                       c("1" = 'Erythrocyte',
                                         "2" = 'Erythrocyte',
                                         "3" = 'Erythrocyte',
                                         "4" = 'Erythrocyte',
                                         "5" = 'Erythrocyte',
                                         "6" = 'Erythrocyte',
                                         "7" = 'Multipotent progenitors',
                                         "8" = 'Megakaryocytes',
                                         "9" = 'GMP',
                                         "10" = 'GMP',
                                         "11" = 'Dendritic cells',
                                         "12" = 'Basophils',
                                         "13" = 'Basophils',
                                         "14" = 'Monocytes',
                                         "15" = 'Monocytes',
                                         "16" = 'Neutrophils',
                                         "17" = 'Neutrophils',
                                         "18" = 'Eosinophls',
                                         "19" = 'lymphoid'))

cell_type_color <- c("Basophils" = "#E088B8",
                     "Dendritic cells" = "#46C7EF",
                     "Eosinophls" = "#EFAD1E",
                     "Erythrocyte" = "#8CB3DF",
                     "Monocytes" = "#53C0AD",
                     "Multipotent progenitors" = "#4EB859",
                     "GMP" = "#D097C4",
                     "Megakaryocytes" = "#ACC436",
                     "Neutrophils" = "#F5918A",
                     'NA' = '#000080')

# Pass TRUE if you want to see progress output on some of Monocle 3's operations
DelayedArray:::set_verbose_block_processing(TRUE)

# Passing a higher value will make some computations faster but use more memory. Adjust with caution!
options(DelayedArray.block.size=1000e6)

cds <- estimateSizeFactors(cds)
cds <- estimateDispersions(cds)

cds <- preprocessCDS(cds, num_dim = 20)

cds <- reduceDimension(cds, reduction_method = 'UMAP')

cds <- partitionCells(cds)

cds <- learnGraph(cds,  RGE_method = 'SimplePPT')

plot_cell_trajectory(cds,
                     color_by = "cell_type2") +
                     scale_color_manual(values = cell_type_color)

