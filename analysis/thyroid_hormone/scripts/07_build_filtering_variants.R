################################################################################
# Script: 07_build_filtering_variants.R
#
# Purpose:
#   Build analysis variants based on:
#     1. all cells
#     2. importance-filtered cells
#
# Filtering rule:
#   Keep a cell if its top-ranked gene importance is:
#     - > 0.02
#       OR
#     - > 85th percentile of top-gene importance within the same output file
#
# Outputs:
#   - results/rds/cascade_outputs_with_filtering_flags.rds
#   - results/rds/cascade_outputs_filtered_variants.rds
#   - results/tables/filtering_summary.csv
#   - results/tables/filtering_thresholds.csv
################################################################################


#######################################
# LOAD CONFIG
#######################################

source(here::here("scripts", "00_config.R"))


#######################################
# LOAD PACKAGES
#######################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tibble)
})


#######################################
# LOAD INPUT OBJECTS
#######################################

cascade_outputs_standardized <- readRDS(
  here::here("results", "rds", "cascade_outputs_standardized.rds")
)


#######################################
# STEP 1: COMPUTE TOP-GENE IMPORTANCE PER CELL
#######################################

cell_top_gene_importance <- cascade_outputs_standardized %>%
  group_by(
    .data$dataset_name,
    .data$analysis_task,
    .data$perturbation_strategy,
    .data$cell_key,
    .data$batch_idx,
    .data$sample_in_batch,
    .data$cell_type
  ) %>%
  slice_min(order_by = .data$rank, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(
    dataset_name,
    analysis_task,
    perturbation_strategy,
    cell_key,
    batch_idx,
    sample_in_batch,
    cell_type,
    top_gene = gene,
    top_gene_rank = rank,
    top_gene_importance = importance,
    top_gene_importance_softmax = importance_softmax
  )


#######################################
# STEP 2: FILE-SPECIFIC QUANTILE THRESHOLDS
#######################################

filtering_thresholds <- cell_top_gene_importance %>%
  group_by(.data$dataset_name, .data$analysis_task, .data$perturbation_strategy) %>%
  summarise(
    n_cells = n(),
    absolute_threshold = ANALYSIS_PARAMS$importance_threshold_absolute,
    quantile_threshold = stats::quantile(
      .data$top_gene_importance,
      probs = ANALYSIS_PARAMS$importance_threshold_quantile,
      na.rm = TRUE
    ),
    min_top_importance = min(.data$top_gene_importance, na.rm = TRUE),
    median_top_importance = median(.data$top_gene_importance, na.rm = TRUE),
    max_top_importance = max(.data$top_gene_importance, na.rm = TRUE),
    .groups = "drop"
  )


#######################################
# STEP 3: FLAG CELLS PASSING IMPORTANCE FILTER
#######################################

cell_filter_flags <- cell_top_gene_importance %>%
  left_join(
    filtering_thresholds,
    by = c("dataset_name", "analysis_task", "perturbation_strategy")
  ) %>%
  mutate(
    pass_importance_abs = .data$top_gene_importance > .data$absolute_threshold,
    pass_importance_quant = .data$top_gene_importance > .data$quantile_threshold,
    pass_importance_filter = .data$pass_importance_abs | .data$pass_importance_quant
  )


#######################################
# STEP 4: ATTACH FILTER FLAGS TO FULL CASCADE OUTPUT
#######################################

cascade_outputs_with_filtering_flags <- cascade_outputs_standardized %>%
  left_join(
    cell_filter_flags %>%
      select(
        .data$dataset_name,
        .data$analysis_task,
        .data$perturbation_strategy,
        .data$cell_key,
        .data$top_gene,
        .data$top_gene_importance,
        .data$absolute_threshold,
        .data$quantile_threshold,
        .data$pass_importance_abs,
        .data$pass_importance_quant,
        .data$pass_importance_filter
      ),
    by = c("dataset_name", "analysis_task", "perturbation_strategy", "cell_key")
  )


#######################################
# STEP 5: CREATE ANALYSIS VARIANTS
#######################################

cascade_outputs_filtered_variants <- bind_rows(
  
  cascade_outputs_with_filtering_flags %>%
    mutate(filter_mode = "all_cells"),
  
  cascade_outputs_with_filtering_flags %>%
    filter(.data$pass_importance_filter) %>%
    mutate(filter_mode = "importance_filtered")
)


#######################################
# STEP 6: SUMMARIZE FILTERING
#######################################

filtering_summary <- cell_filter_flags %>%
  group_by(.data$dataset_name, .data$analysis_task, .data$perturbation_strategy) %>%
  summarise(
    n_cells_total = n(),
    n_cells_pass_abs = sum(.data$pass_importance_abs, na.rm = TRUE),
    n_cells_pass_quant = sum(.data$pass_importance_quant, na.rm = TRUE),
    n_cells_pass_filter = sum(.data$pass_importance_filter, na.rm = TRUE),
    prop_cells_pass_filter = mean(.data$pass_importance_filter, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(.data$analysis_task, .data$perturbation_strategy)


#######################################
# SAVE OUTPUTS
#######################################

saveRDS(
  cascade_outputs_with_filtering_flags,
  file = here::here("results", "rds", "cascade_outputs_with_filtering_flags.rds")
)

saveRDS(
  cascade_outputs_filtered_variants,
  file = here::here("results", "rds", "cascade_outputs_filtered_variants.rds")
)

readr::write_csv(
  filtering_summary,
  file = here::here("results", "tables", "filtering_summary.csv")
)

readr::write_csv(
  filtering_thresholds,
  file = here::here("results", "tables", "filtering_thresholds.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Saved filtering variants.")
print(filtering_thresholds)
print(filtering_summary)
