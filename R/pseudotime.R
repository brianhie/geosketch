suppressMessages(library(monocle))

pdf('mono_macro_gs.pdf')

expr.matrix <- as.matrix(read.table("data/pseudotime/mono_macro_gs_table.txt"))
sample.sheet <- read.table("data/pseudotime/mono_macro_meta_gs.txt")
gene.annotation <- read.delim("data/pseudotime/mono_macro_genes.txt")

pd <- new("AnnotatedDataFrame", data = sample.sheet)
fd <- new("AnnotatedDataFrame", data = gene.annotation)
obj <- newCellDataSet(expr.matrix, phenoData = pd, featureData = fd,
                      expressionFamily=tobit())

orderingGenes <- scan("data/pseudotime/mono_macro_diffexpr.txt",
                      what = typeof(""), sep = "\n")
obj <- setOrderingFilter(obj, orderingGenes)

obj <- reduceDimension(obj, max_components = 2, method = 'DDRTree')

obj <- orderCells(obj)

print(pData(obj)$Pseudotime)

plot_cell_trajectory(obj, color_by = "Label")
