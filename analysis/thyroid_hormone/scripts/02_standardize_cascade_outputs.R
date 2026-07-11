################################################################################
# Script: 02_standardize_cascade_outputs.R
#
# Purpose:
#   Read and standardize the four CASCADE explainability output files into one
#   harmonized long-format table.
#
# Output:
#   - results/rds/cascade_outputs_standardized.rds
#   - results/tables/cascade_outputs_summary.csv
#   - results/tables/cascade_cell_index_audit.csv
################################################################################


#######################################
# LOAD CONFIG
#######################################

source(here::here("scripts", "00_config.R"))


#######################################
# LOAD PACKAGES
#######################################

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(readr)
  library(stringr)
  library(tibble)
})


#######################################
# HELPER FUNCTIONS
#######################################

#------------------------------------------------------------------------------
# Function: make_cell_key
#
# Purpose:
#   Build a synthetic cell key from batch_idx and sample_in_batch.
#
# Args:
#   batch_idx: vector
#   sample_in_batch: vector
#
# Returns:
#   Character vector
#------------------------------------------------------------------------------
make_cell_key <- function(batch_idx, sample_in_batch) {
  paste0("batch_", batch_idx, "__cell_", sample_in_batch)
}


#------------------------------------------------------------------------------
# Function: standardize_one_cascade_file
#
# Purpose:
#   Read and standardize one CASCADE explainability output file.
#
# Args:
#   dataset_name: character
#   task: character
#   perturbation_strategy: character
#   path: character
#
# Returns:
#   List with:
#     - data: standardized long data.frame
#     - summary: one-row summary table
#     - cell_audit: one-row audit table
#------------------------------------------------------------------------------
standardize_one_cascade_file <- function(dataset_name,
                                         task,
                                         perturbation_strategy,
                                         path) {
  
  message("Standardizing CASCADE file: ", dataset_name)
  
  df <- data.table::fread(
    input = path,
    data.table = FALSE,
    fill = TRUE,
    quote = "\"",
    encoding = "UTF-8"
  )
  
  required_cols <- c(
    "task", "task_type", "batch_idx", "sample_in_batch",
    "cell_type", "rank", "gene", "gene_id",
    "importance", "importance_softmax"
  )
  
  missing_cols <- setdiff(required_cols, colnames(df))
  if (length(missing_cols) > 0) {
    stop(
      "Missing required columns in ", dataset_name, ": ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  df_std <- df %>%
    transmute(
      dataset_name = dataset_name,
      source = "cascade",
      analysis_task = task,
      perturbation_strategy = perturbation_strategy,
      original_task = .data$task,
      task_type = .data$task_type,
      batch_idx = as.integer(.data$batch_idx),
      sample_in_batch = as.integer(.data$sample_in_batch),
      cell_key = make_cell_key(.data$batch_idx, .data$sample_in_batch),
      cell_type = as.character(.data$cell_type),
      gene_symbol = as.character(.data$gene),
      gene_id = as.character(.data$gene_id),
      gene = as.character(.data$gene),   # keep a generic gene column for joins later
      rank = as.integer(.data$rank),
      importance = as.numeric(.data$importance),
      importance_softmax = as.numeric(.data$importance_softmax)
    ) %>%
    arrange(.data$batch_idx, .data$sample_in_batch, .data$rank)
  
  # STEP: Create a stable per-file cell index based on unique cell order
  unique_cells <- df_std %>%
    distinct(.data$batch_idx, .data$sample_in_batch, .data$cell_key, .data$cell_type) %>%
    arrange(.data$batch_idx, .data$sample_in_batch) %>%
    mutate(prediction_row_index = row_number())
  
  df_std <- df_std %>%
    left_join(
      unique_cells,
      by = c("batch_idx", "sample_in_batch", "cell_key", "cell_type")
    )
  
  # Summary table
  summary_tbl <- tibble(
    dataset_name = dataset_name,
    analysis_task = task,
    perturbation_strategy = perturbation_strategy,
    n_rows = nrow(df_std),
    n_unique_cells = n_distinct(df_std$cell_key),
    n_unique_genes = n_distinct(df_std$gene),
    rank_min = min(df_std$rank, na.rm = TRUE),
    rank_max = max(df_std$rank, na.rm = TRUE),
    importance_min = min(df_std$importance, na.rm = TRUE),
    importance_max = max(df_std$importance, na.rm = TRUE)
  )
  
  # Audit expected ranks per cell
  cell_rank_counts <- df_std %>%
    count(.data$cell_key, name = "n_ranks_for_cell")
  
  cell_audit_tbl <- tibble(
    dataset_name = dataset_name,
    analysis_task = task,
    perturbation_strategy = perturbation_strategy,
    n_unique_cells = n_distinct(df_std$cell_key),
    min_ranks_per_cell = min(cell_rank_counts$n_ranks_for_cell, na.rm = TRUE),
    median_ranks_per_cell = median(cell_rank_counts$n_ranks_for_cell, na.rm = TRUE),
    max_ranks_per_cell = max(cell_rank_counts$n_ranks_for_cell, na.rm = TRUE),
    any_duplicate_gene_within_cell = any(
      df_std %>%
        count(.data$cell_key, .data$gene) %>%
        filter(.data$n > 1) %>%
        nrow() > 0
    )
  )
  
  list(
    data = df_std,
    summary = summary_tbl,
    cell_audit = cell_audit_tbl
  )
}


#######################################
# RUN STANDARDIZATION
#######################################

cascade_results <- vector("list", nrow(CASCADE_FILES))

for (i in seq_len(nrow(CASCADE_FILES))) {
  cascade_results[[i]] <- standardize_one_cascade_file(
    dataset_name = CASCADE_FILES$dataset_name[i],
    task = CASCADE_FILES$task[i],
    perturbation_strategy = CASCADE_FILES$perturbation_strategy[i],
    path = CASCADE_FILES$path[i]
  )
}

cascade_outputs_standardized <- bind_rows(lapply(cascade_results, `[[`, "data"))
cascade_outputs_summary <- bind_rows(lapply(cascade_results, `[[`, "summary"))
cascade_cell_index_audit <- bind_rows(lapply(cascade_results, `[[`, "cell_audit"))


#######################################
# SAVE OUTPUTS
#######################################

saveRDS(
  cascade_outputs_standardized,
  file = here::here("results", "rds", "cascade_outputs_standardized.rds")
)

readr::write_csv(
  cascade_outputs_summary,
  file = here::here("results", "tables", "cascade_outputs_summary.csv")
)

readr::write_csv(
  cascade_cell_index_audit,
  file = here::here("results", "tables", "cascade_cell_index_audit.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Saved standardized CASCADE outputs.")
print(cascade_outputs_summary)
print(cascade_cell_index_audit)
