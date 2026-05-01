suppressPackageStartupMessages({
  library(Seurat)
  library(arrow)
  library(dplyr)
  library(ggplot2)
  library(Matrix)
})

# ------------------------------------------------------------------------------
# Configuration (edit for your dataset; keep CLASS_LABELS aligned with sc_pipeline.py)
# ------------------------------------------------------------------------------

# --- Input ---
INPUT_PATH <- "CM_Human_HF_SCP1303_seurat_obj.rds"
INPUT_TYPE <- "seurat"

# --- Output paths (Seurat input: base Parquet name and results stem) ---
OUTPUT_PARQUET    <- "SCP1303_CM.parquet"
RESULTS_BASE_NAME <- "SCP1303_CM"

# --- Seurat meta.data mapping (INPUT_TYPE == "seurat") ---
META_DISEASE_COL <- "disease_state"
META_SAMPLE_COL  <- "sample_id"

# --- Class labels (order must match CLASS_MAP indices in sc_pipeline.py) ---
CLASS_LABELS <- c(
  "normal",
  "hypertrophic cardiomyopathy",
  "dilated cardiomyopathy"
)

# --- Seurat processing ---
RUN_NORMALIZATION_PIPELINE <- TRUE
N_VARIABLE_FEATURES       <- 2000L
SEURAT_CLUSTER_DIMS       <- 1:6
SEURAT_CLUSTER_RESOLUTION <- 0.1
ELBOW_NDIMS               <- 50L
ELBOW_PLOT_PATH           <- "elbow_plot.png"

# --- k-means (PCA space; for cluster-to-phenotype comparison) ---
KMEANS_CENTERS <- 3L
KMEANS_SEED    <- 42L
KMEANS_NSTART  <- 25L

# --- Python invocation: PYTHON_EXE SC_PIPELINE_PY <parquet> [--output-dir DIR] ---
PYTHON_EXE     <- "/home/diren/.local/share/r-miniconda/envs/r-reticulate/bin/python3.8"
SC_PIPELINE_PY <- "sc_pipeline.py"
OUTPUT_DIR     <- "."

# --- Output artifacts ---
OUTPUT_SEURAT_RDS    <- "integrated_cardiomyocytes_with_metadata.rds"
FEATURE_PLOT_KMEANS  <- "feature_plot_kmeans.png"
FEATURE_PLOT_DISEASE <- "feature_plot_disease.png"

# --- Manual k-means -> phenotype labels (character cluster id from kmeans()) ---
# R uses 1-based cluster indices; keys must be "1", "2", ... as character.
# Align values with biological labels used in your manuscript.
cluster_to_disease <- c(
  "1" = "Donor",
  "2" = "DCM"
)

set.seed(KMEANS_SEED)

# ==============================================================================
# Section 1: Data ingestion and Parquet export
# ==============================================================================

message("Section 1: data ingestion and Parquet export")

have_seurat <- FALSE
cell_order  <- NULL
so          <- NULL

if (INPUT_TYPE == "parquet") {
  OUTPUT_PARQUET    <- INPUT_PATH
  RESULTS_BASE_NAME <- sub("\\.(parquet|csv)$", "", basename(INPUT_PATH))
  message(
    "Parquet mode: OUTPUT_PARQUET = ", OUTPUT_PARQUET,
    ", RESULTS_BASE_NAME = ", RESULTS_BASE_NAME
  )
} else if (INPUT_TYPE == "seurat") {
  have_seurat <- TRUE
  message("Loading Seurat object: ", INPUT_PATH)

  ext <- tolower(sub(".*\\.", "", INPUT_PATH))
  if (ext == "rds") {
    so <- tryCatch(
      readRDS(INPUT_PATH),
      error = function(e) {
        stop(
          "Failed to read RDS '", INPUT_PATH, "': ", conditionMessage(e),
          call. = FALSE
        )
      }
    )
  } else if (ext == "robj") {
    robj_env <- new.env(parent = emptyenv())
    tryCatch(
      load(INPUT_PATH, envir = robj_env),
      error = function(e) {
        stop(
          "Failed to load R object '", INPUT_PATH, "': ", conditionMessage(e),
          call. = FALSE
        )
      }
    )
    if (exists("obj", envir = robj_env, inherits = FALSE)) {
      so <- robj_env$obj
    } else if (exists("so", envir = robj_env, inherits = FALSE)) {
      so <- robj_env$so
    } else {
      stop("Expected object 'obj' or 'so' in .Robj file: ", INPUT_PATH, call. = FALSE)
    }
    rm(robj_env)
    gc()
  } else {
    stop("Seurat input must be .rds or .Robj. Got: ", INPUT_PATH, call. = FALSE)
  }

  gc()
  cell_order <- colnames(so)

  if (RUN_NORMALIZATION_PIPELINE) {
    so <- NormalizeData(so, verbose = FALSE)
    so <- FindVariableFeatures(
      so,
      selection.method = "vst",
      nfeatures        = N_VARIABLE_FEATURES,
      verbose          = FALSE
    )
    so <- ScaleData(so, verbose = FALSE)
    so <- RunPCA(so, features = VariableFeatures(object = so), verbose = FALSE)
  }

  meta <- so@meta.data
  meta$cell_id <- rownames(meta)

  if (!"disease_state" %in% colnames(meta)) {
    if (META_DISEASE_COL %in% colnames(meta)) {
      meta$disease_state <- meta[[META_DISEASE_COL]]
    } else {
      stop(
        "Seurat meta.data must contain 'disease_state' or META_DISEASE_COL.",
        call. = FALSE
      )
    }
  }
  if (!"sample_id" %in% colnames(meta)) {
    if (META_SAMPLE_COL %in% colnames(meta)) {
      meta$sample_id <- meta[[META_SAMPLE_COL]]
    } else {
      stop(
        "Seurat meta.data must contain 'sample_id' or META_SAMPLE_COL.",
        call. = FALSE
      )
    }
  }

  stopifnot(
    "cell_id column missing"        = "cell_id" %in% colnames(meta),
    "disease_state column missing"  = "disease_state" %in% colnames(meta),
    "sample_id column missing"      = "sample_id" %in% colnames(meta)
  )

  so@meta.data <- meta

  message("Exporting expression and metadata to Parquet (long, sparse-safe)")
  expr     <- GetAssayData(so, layer = "data")
  meta_sub <- meta[cell_order, c("cell_id", "disease_state", "sample_id"), drop = FALSE]

  parquet_base <- sub("\\.parquet$", "", OUTPUT_PARQUET)
  meta_path <- paste0(parquet_base, "_meta.parquet")
  expr_path <- paste0(parquet_base, "_expr.parquet")
  write_parquet(meta_sub, meta_path)

  if (inherits(expr, "dgCMatrix")) {
    s <- summary(expr)
    expr_long <- data.frame(
      cell_id = colnames(expr)[s$j],
      gene    = rownames(expr)[s$i],
      value   = s$x,
      stringsAsFactors = FALSE
    )
  } else {
    expr <- as(expr, "CsparseMatrix")
    s <- summary(expr)
    expr_long <- data.frame(
      cell_id = colnames(expr)[s$j],
      gene    = rownames(expr)[s$i],
      value   = s$x,
      stringsAsFactors = FALSE
    )
  }
  write_parquet(expr_long, expr_path)
  message("Wrote ", meta_path, " and ", expr_path)
} else {
  stop("INPUT_TYPE must be 'seurat' or 'parquet'. Got: ", INPUT_TYPE, call. = FALSE)
}

# ==============================================================================
# Section 2: Python machine-learning pipeline
# ==============================================================================

message("Section 2: invoking Python pipeline")

py_args <- c(path.expand(SC_PIPELINE_PY), path.expand(OUTPUT_PARQUET))
if (!identical(OUTPUT_DIR, ".")) {
  py_args <- c(py_args, "--output-dir", path.expand(OUTPUT_DIR))
}
ret <- system2(PYTHON_EXE, py_args, stdout = TRUE, stderr = TRUE)
if (!is.null(attr(ret, "status")) && attr(ret, "status") != 0) {
  stop(
    "Python pipeline failed (exit ", attr(ret, "status"), "). ",
    "Verify PYTHON_EXE and SC_PIPELINE_PY.",
    call. = FALSE
  )
}
message("Python pipeline completed.")

RESULTS_CSV     <- file.path(OUTPUT_DIR, paste0(RESULTS_BASE_NAME, "_results.csv"))
RESULTS_PARQUET <- file.path(OUTPUT_DIR, paste0(RESULTS_BASE_NAME, "_results.parquet"))

# ==============================================================================
# Section 2b: Construct Seurat from Parquet when no object was supplied
# ==============================================================================

if (!have_seurat) {
  message("Section 2b: building Seurat from Parquet")

  parquet_base <- sub("\\.parquet$", "", OUTPUT_PARQUET)
  meta_path    <- paste0(parquet_base, "_meta.parquet")
  expr_path    <- paste0(parquet_base, "_expr.parquet")
  meta_cols    <- c("cell_id", "disease_state", "sample_id")

  if (file.exists(meta_path) && file.exists(expr_path)) {
    message("Reading split Parquet (metadata + long expression)")
    meta_df   <- read_parquet(meta_path)
    expr_long <- read_parquet(expr_path)
    stopifnot(
      "metadata columns missing from meta Parquet" =
        all(meta_cols %in% colnames(meta_df)),
      "expression long table missing required columns" =
        all(c("cell_id", "gene", "value") %in% colnames(expr_long))
    )
    cells <- meta_df$cell_id
    genes <- sort(unique(expr_long$gene))
    cell_to_j <- match(expr_long$cell_id, cells)
    gene_to_i <- match(expr_long$gene, genes)
    valid <- !is.na(cell_to_j) & !is.na(gene_to_i)
    counts_mat <- sparseMatrix(
      i = gene_to_i[valid],
      j = cell_to_j[valid],
      x = expr_long$value[valid],
      dims = c(length(genes), length(cells)),
      dimnames = list(genes, cells)
    )
    rownames(meta_df) <- meta_df$cell_id
    so <- CreateSeuratObject(counts = counts_mat, meta.data = meta_df)
  } else {
    message("Reading legacy wide Parquet")
    df <- read_parquet(OUTPUT_PARQUET)
    stopifnot(
      "metadata columns missing from wide Parquet" =
        all(meta_cols %in% colnames(df))
    )
    gene_cols <- setdiff(colnames(df), meta_cols)
    expr <- as.matrix(df[, gene_cols, drop = FALSE])
    rownames(expr) <- df$cell_id
    counts_mat <- t(expr)
    colnames(counts_mat) <- df$cell_id
    meta <- df[, meta_cols, drop = FALSE]
    rownames(meta) <- meta$cell_id
    so <- CreateSeuratObject(counts = counts_mat, meta.data = meta)
  }

  if (RUN_NORMALIZATION_PIPELINE) {
    so <- NormalizeData(so, verbose = FALSE)
    so <- FindVariableFeatures(
      so,
      selection.method = "vst",
      nfeatures        = N_VARIABLE_FEATURES,
      verbose          = FALSE
    )
    so <- ScaleData(so, verbose = FALSE)
    so <- RunPCA(so, features = VariableFeatures(object = so), verbose = FALSE)
  }
  cell_order  <- colnames(so)
  have_seurat <- TRUE
  message("Seurat object constructed from Parquet.")
}

# ==============================================================================
# Section 3: Clustering, k-means, and diagnostic dimension plots
# ==============================================================================

message("Section 3: clustering and dimension reduction plots")

message("Elbow plot (PCA standard deviations)")
elbow_plot <- ElbowPlot(so, ndims = ELBOW_NDIMS) +
  ggtitle("Elbow plot: PCs for FindNeighbors")
ggsave(ELBOW_PLOT_PATH, plot = elbow_plot, width = 6, height = 4, dpi = 150)
message("Saved ", ELBOW_PLOT_PATH)

pca_embeddings <- Embeddings(so, reduction = "pca")[, SEURAT_CLUSTER_DIMS, drop = FALSE]
set.seed(KMEANS_SEED)
kmeans_res <- kmeans(
  pca_embeddings,
  centers = KMEANS_CENTERS,
  nstart  = KMEANS_NSTART
)
kmeans_col <- paste0("kmeans_", KMEANS_CENTERS)
so@meta.data[[kmeans_col]] <- as.factor(kmeans_res$cluster)

so <- FindNeighbors(so, reduction = "pca", dims = SEURAT_CLUSTER_DIMS, verbose = FALSE)
so <- FindClusters(so, resolution = SEURAT_CLUSTER_RESOLUTION, verbose = FALSE)
message("Seurat cluster sizes:")
print(table(so@meta.data$seurat_clusters))
message("k-means cluster sizes:")
print(table(so@meta.data[[kmeans_col]]))

reduction_2d <- if ("umap" %in% names(so@reductions)) "umap" else "pca"

p_kmeans  <- DimPlot(so, reduction = reduction_2d, group.by = kmeans_col) +
  ggtitle("k-means cluster")
p_disease <- DimPlot(so, reduction = reduction_2d, group.by = "disease_state") +
  ggtitle("Disease state")
ggsave(FEATURE_PLOT_KMEANS,  plot = p_kmeans,  width = 6, height = 5, dpi = 150)
ggsave(FEATURE_PLOT_DISEASE, plot = p_disease, width = 6, height = 5, dpi = 150)
message("Saved ", FEATURE_PLOT_KMEANS, " and ", FEATURE_PLOT_DISEASE)
message(
  "Review the plots and update `cluster_to_disease` in Configuration if needed, ",
  "then re-run the script."
)

# ==============================================================================
# Section 4: Attach model outputs and k-means-derived labels
# ==============================================================================

message("Section 4: attaching results to Seurat")

if (file.exists(RESULTS_PARQUET)) {
  res <- read_parquet(RESULTS_PARQUET)
} else if (file.exists(RESULTS_CSV)) {
  res <- read.csv(RESULTS_CSV, stringsAsFactors = FALSE)
} else {
  stop(
    "Results not found. Expected ", RESULTS_PARQUET, " or ", RESULTS_CSV, ".",
    call. = FALSE
  )
}

so@meta.data$cluster_disease <- cluster_to_disease[as.character(so@meta.data[[kmeans_col]])]

meta_res <- left_join(
  data.frame(cell_id = cell_order, stringsAsFactors = FALSE),
  res,
  by = "cell_id"
)
cols_to_add <- setdiff(colnames(meta_res), "cell_id")
for (col in cols_to_add) {
  so[[col]] <- meta_res[[col]]
}

so$y_true_label   <- CLASS_LABELS[so$y_true + 1L]
so$xgb_pred_label <- CLASS_LABELS[so$xgb_pred + 1L]
so$nn_pred_label  <- CLASS_LABELS[so$nn_pred + 1L]
so$rf_pred_label  <- CLASS_LABELS[so$rf_pred + 1L]

message("Attached columns: ", paste(cols_to_add, collapse = ", "))
message("Attached decoded labels: y_true_label, xgb_pred_label, nn_pred_label, rf_pred_label")
message("Attached cluster_disease from k-means mapping.")

saveRDS(so, OUTPUT_SEURAT_RDS)
message("Finished. Seurat object written to ", OUTPUT_SEURAT_RDS)
