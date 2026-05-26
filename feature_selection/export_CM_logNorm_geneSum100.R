library(Seurat)
library(Matrix)

## Load the seurat object 
load("data/mergeTAC24_outsource_CM_subset_ML.RData")
CM <- mergeTAC24_outsource_CM_subset_ML
rm(mergeTAC24_outsource_CM_subset_ML)

## Extract log-normalized RNA expression
DefaultAssay(CM) <- "RNA"
expr_mat <- GetAssayData(
  object = CM,
  assay  = "RNA",
  slot   = "data"
)

## Gene-level filtering -- Retain genes with sufficient total expression across all cells
gene_totals <- Matrix::rowSums(expr_mat)
min_gene_total <- 100
expr_filt <- expr_mat[gene_totals > min_gene_total, ] 

## Extract the MLP predicted label from the Seurat object 
labels <- CM@meta.data$mlp_pred_name
names(labels) <- colnames(CM)

## Export files

# Expression matrix (genes × cells, Matrix Market format)
Matrix::writeMM(
  obj  = expr_filt,
  file = "data/CM_RNA.mtx"
)

# Gene names
write.table(
  rownames(expr_filt),
  file      = "data/genes.txt",
  quote     = FALSE,
  row.names = FALSE,
  col.names = FALSE
)

# Cell barcodes
write.table(
  colnames(expr_filt),
  file      = "data/CM_cells.txt",
  quote     = FALSE,
  row.names = FALSE,
  col.names = FALSE
)

# Cell Labels 
write.table(
  labels,
  file      = "data/CM_labels.txt",
  sep       = "\t",
  quote     = FALSE,
  row.names = TRUE,
  col.names = FALSE
)
