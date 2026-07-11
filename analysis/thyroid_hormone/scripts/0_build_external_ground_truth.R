################################################################################
# Script: 0_build_external_ground_truth.R
#
# Purpose:
#   Build the external thyroid-hormone-response ground truth (Wu et al.,
#   E-MTAB-14810) used to validate CASCADE-Explainer gene rankings: loads raw
#   Low-TH / High-TH 10x matrices, QCs and clusters the cells, annotates broad
#   cell types via a literature marker panel, then runs per-cell-type
#   HighTH-vs-LowTH differential expression.
#
# Output:
#   - data/gt/paper_marker_cluster_annotation_summary.csv
#   - data/gt/seurat_metadata_with_paper_marker_annotation.csv
#   - data/gt/Task1_external_ground_truth_EMTAB14810.csv (consumed by 05_build_ground_truth_objects.R)
#
# Usage:
#   source("scripts/0_build_external_ground_truth.R")
################################################################################


#######################################
# LOAD CONFIG
#######################################

source(here::here("scripts", "00_config.R"))


#######################################
# LOAD PACKAGES
#######################################

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})


#######################################
# STEP 1: LOAD RAW 10X MATRICES
#######################################

raw_dir <- here::here("data", "raw", "E-MTAB-14810")

low_counts <- readMM(file.path(raw_dir, "Lib1_LibLowTH_matrix.mtx.gz"))
low_genes  <- read.delim(file.path(raw_dir, "Lib1_LibLowTH_features.tsv.gz"), header = FALSE)
low_cells  <- read.delim(file.path(raw_dir, "Lib1_LibLowTH_barcodes.tsv.gz"), header = FALSE)

high_counts <- readMM(file.path(raw_dir, "Lib2_LibHighTH_matrix.mtx.gz"))
high_genes  <- read.delim(file.path(raw_dir, "Lib2_LibHighTH_features.tsv.gz"), header = FALSE)
high_cells  <- read.delim(file.path(raw_dir, "Lib2_LibHighTH_barcodes.tsv.gz"), header = FALSE)

rownames(low_counts)  <- low_genes$V1
rownames(high_counts) <- high_genes$V1
colnames(low_counts)  <- paste0("LowTH_", low_cells$V1)
colnames(high_counts) <- paste0("HighTH_", high_cells$V1)

stopifnot(all(rownames(low_counts) == rownames(high_counts)))

counts <- cbind(low_counts, high_counts)

meta <- data.frame(
  cell = colnames(counts),
  condition = c(rep("LowTH", ncol(low_counts)), rep("HighTH", ncol(high_counts)))
)
rownames(meta) <- meta$cell

gene_annot <- data.frame(
  gene_id = low_genes$V1,
  gene_symbol = low_genes$V2,
  stringsAsFactors = FALSE
)


#######################################
# STEP 2: QC
#######################################

seu <- CreateSeuratObject(counts = counts, meta.data = meta)

mt_ids <- gene_annot$gene_id[grepl("^mt-", gene_annot$gene_symbol, ignore.case = TRUE)]
seu[["percent.mt"]] <- PercentageFeatureSet(seu, features = mt_ids)

VlnPlot(seu, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

seu <- subset(
  seu,
  subset =
    nFeature_RNA > 500 &
    nFeature_RNA < 5500 &
    nCount_RNA > 1000 &
    nCount_RNA < 22000 &
    percent.mt < 2
)
VlnPlot(seu, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
FeatureScatter(seu, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")


#######################################
# STEP 3: NORMALIZATION, PCA, CLUSTERING
#######################################

seu <- NormalizeData(seu)
seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 3000)
seu <- ScaleData(seu)
seu <- RunPCA(seu, npcs = 50)
ElbowPlot(seu, ndims = 50)

seu <- FindNeighbors(seu, dims = 1:20)
seu <- FindClusters(seu, resolution = 0.3)
seu <- RunUMAP(seu, dims = 1:20)

DimPlot(seu, reduction = "umap", group.by = "condition")
DimPlot(seu, reduction = "umap", label = TRUE)

cluster_markers <- FindAllMarkers(
  seu,
  only.pos = TRUE,
  min.pct = 0.1,
  logfc.threshold = 0.25
)


#######################################
# STEP 4: PAPER-MARKER-BASED CLUSTER ANNOTATION
#
# Assigns broad cell-type labels to clusters using:
#   1) module scores from literature-derived marker sets
#   2) overlap of each cluster's top markers with the same marker sets
#######################################

get_ids_from_symbols <- function(symbols, annot_df) {
  annot_df$gene_id[match(symbols, annot_df$gene_symbol)]
}

# Broad cell-type marker sets covering cortex, hypothalamus, and pituitary
# (E-MTAB-14810 spans a broader region than CASCADE's own cortex-only M2
# training data; this reference set is intentionally broader).
paper_sets <- list(
  GABA_neuron = c("Gad2", "Lef1", "Vipr2", "Syt10", "Sst", "Npy", "Avp"),
  glut_neuron = c(
    "Il1rapl2", "Fezf1", "Lmx1a", "Onecut2", "Myo5b",
    "Tbx3", "Pomc", "Tafa4", "Ppp1r17", "Slc17a7", "Tac2"
  ),
  astrocyte = c("Slc1a3"),
  oligodendrocyte = c("Mag"),
  OPC = c("Pdgfra"),
  microglia = c("Aif1"),
  tanycyte = c("Rax", "Col23a1"),
  pars_tuberalis = c("Tshb")
)

paper_set_ids <- lapply(paper_sets, get_ids_from_symbols, annot_df = gene_annot)
paper_set_ids <- lapply(paper_set_ids, function(x) unique(x[!is.na(x)]))

for (nm in names(paper_set_ids)) {
  if (length(paper_set_ids[[nm]]) > 0) {
    seu <- AddModuleScore(object = seu, features = list(paper_set_ids[[nm]]), name = paste0(nm, "_score"))
  }
}

score_cols <- grep("_score1$", colnames(seu@meta.data), value = TRUE)

cluster_scores <- seu@meta.data %>%
  mutate(cluster = as.character(seu$seurat_clusters)) %>%
  group_by(cluster) %>%
  summarise(across(all_of(score_cols), mean, na.rm = TRUE), .groups = "drop")

cluster_markers_annot <- cluster_markers %>%
  left_join(gene_annot %>% distinct(gene_id, gene_symbol), by = c("gene" = "gene_id"))

top_markers_by_cluster <- cluster_markers_annot %>%
  group_by(cluster) %>%
  slice_max(order_by = avg_log2FC, n = 50) %>%
  summarise(top_symbols = list(unique(na.omit(gene_symbol))), .groups = "drop")

overlap_df <- lapply(seq_len(nrow(top_markers_by_cluster)), function(i) {
  cl <- as.character(top_markers_by_cluster$cluster[i])
  genes <- top_markers_by_cluster$top_symbols[[i]]
  overlaps <- sapply(names(paper_sets), function(ct) sum(genes %in% paper_sets[[ct]]))
  data.frame(cluster = cl, paper_set = names(overlaps), overlap = as.numeric(overlaps), stringsAsFactors = FALSE)
}) %>%
  bind_rows()

score_mat <- as.data.frame(cluster_scores)
rownames(score_mat) <- score_mat$cluster
score_mat$cluster <- NULL

score_winner <- apply(score_mat, 1, function(x) names(x)[which.max(x)])
score_winner <- gsub("_score1$", "", score_winner)
score_margin <- apply(score_mat, 1, function(x) {
  xs <- sort(x, decreasing = TRUE)
  xs[1] - xs[2]
})

score_assign <- data.frame(
  cluster = rownames(score_mat),
  score_label = score_winner,
  score_margin = score_margin,
  stringsAsFactors = FALSE
)

overlap_winner <- overlap_df %>%
  group_by(cluster) %>%
  slice_max(order_by = overlap, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(overlap_label = ifelse(overlap == 0, "unknown", paper_set), overlap_n = overlap) %>%
  select(cluster, overlap_label, overlap_n)

# If score-based and overlap-based labels agree, accept the shared label.
# If overlap is at least weakly informative (>=2 markers) and the score margin
# is small, trust overlap over score; if both signals are weak, flag ambiguous.
cluster_assign <- score_assign %>%
  left_join(overlap_winner, by = "cluster") %>%
  mutate(
    final_label = case_when(
      score_label == overlap_label ~ score_label,
      overlap_n >= 2 & score_margin >= 0.05 ~ score_label,
      overlap_n >= 2 & score_margin < 0.05 ~ overlap_label,
      score_margin < 0.05 & overlap_n <= 1 ~ "ambiguous",
      TRUE ~ score_label
    )
  )

cluster_to_celltype <- setNames(cluster_assign$final_label, cluster_assign$cluster)
seu$cell_type_marker <- unname(cluster_to_celltype[as.character(seu$seurat_clusters)])

table(seu$seurat_clusters, seu$cell_type_marker)
table(seu$cell_type_marker, seu$condition)

DimPlot(seu, reduction = "umap", group.by = "cell_type_marker", label = TRUE, repel = TRUE)

score_long <- cluster_scores %>%
  tidyr::pivot_longer(cols = -cluster, names_to = "score_name", values_to = "mean_score") %>%
  mutate(score_name = gsub("_score1$", "", score_name))

ggplot(score_long, aes(x = score_name, y = factor(cluster), fill = mean_score)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(x = "Marker set", y = "Cluster", fill = "Mean module score") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


#######################################
# STEP 5: PER-CELL-TYPE DIFFERENTIAL EXPRESSION (HighTH vs LowTH)
#######################################

write.csv(cluster_assign, file = here::here("data", "gt", "paper_marker_cluster_annotation_summary.csv"), row.names = FALSE)
write.csv(seu@meta.data, file = here::here("data", "gt", "seurat_metadata_with_paper_marker_annotation.csv"), row.names = TRUE)

seu_gt <- subset(seu, subset = cell_type_marker != "ambiguous")
Idents(seu_gt) <- seu_gt$condition

cell_types <- unique(seu_gt$cell_type_marker)
task1_truth_list <- list()

for (ct in cell_types) {
  seu_ct <- subset(seu_gt, subset = cell_type_marker == ct)

  res <- FindMarkers(seu_ct, ident.1 = "HighTH", ident.2 = "LowTH", logfc.threshold = 0, min.pct = 0)
  res$gene_id <- rownames(res)
  res <- merge(res, gene_annot, by = "gene_id", all.x = TRUE)
  res$cell_type <- ct
  res$abs_score <- abs(res$avg_log2FC)

  task1_truth_list[[ct]] <- res
}

task1_truth_all <- dplyr::bind_rows(task1_truth_list)

write.csv(
  task1_truth_all,
  here::here("data", "gt", "Task1_external_ground_truth_EMTAB14810.csv"),
  row.names = FALSE
)

message("Saved external thyroid-response ground truth.")
