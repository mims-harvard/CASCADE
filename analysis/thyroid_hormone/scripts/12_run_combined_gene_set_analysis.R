################################################################################
# Script: 12_run_combined_gene_set_analysis.R
#
# Purpose:
#   Build shared and task-specific gene sets using the selected aggregation +
#   filter configuration, within and across perturbation strategies.
#
# Outputs:
#   - results/tables/combined_gene_set_membership.csv
#   - results/tables/combined_gene_set_summary.csv
#   - results/tables/combined_gene_set_enrichment.csv
################################################################################


#######################################
# LOAD CONFIG
#######################################

source(here::here("scripts", "00_config.R"))
source(here::here("scripts", "gt_benchmark_helpers.R"))


#######################################
# LOAD PACKAGES
#######################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tibble)
  library(tidyr)
})


#######################################
# LOAD INPUT OBJECTS
#######################################

cascade_main_selected_aggregation <- readRDS(
  here::here("results", "rds", "cascade_main_selected_aggregation.rds")
)

gt_thyroid_response_standardized <- readRDS(
  here::here("results", "rds", "gt_thyroid_response_standardized.rds")
)

gt_mmc2_long <- readRDS(
  here::here("results", "rds", "gt_mmc2_long_mapped_final.rds")
)


#######################################
# STEP 1: DEFINE HIGH-IMPORTANCE SETS
#######################################

TOP_N_MAIN <- 200

main_gene_sets <- cascade_main_selected_aggregation %>%
  dplyr::mutate(in_top_main = .data$aggregated_rank <= TOP_N_MAIN) %>%
  dplyr::select(
    .data$dataset_name,
    .data$analysis_task,
    .data$perturbation_strategy,
    .data$gene,
    .data$aggregated_rank,
    .data$aggregated_score,
    .data$in_top_main
  )


#######################################
# STEP 2: WITHIN-PERTURBATION COMBINED SETS
#
# Question:
#   Within each perturbation strategy, which genes are shared between
#   treatment and THR, and which are task-specific?
#######################################

within_strategy_membership <- main_gene_sets %>%
  dplyr::filter(.data$in_top_main) %>%
  dplyr::select(.data$analysis_task, .data$perturbation_strategy, .data$gene) %>%
  dplyr::distinct() %>%
  tidyr::pivot_wider(
    names_from = .data$analysis_task,
    values_from = .data$analysis_task,
    values_fn = length,
    values_fill = 0
  ) %>%
  dplyr::mutate(
    treatment = if ("treatment" %in% colnames(.)) .data$treatment else 0L,
    THR = if ("THR" %in% colnames(.)) .data$THR else 0L,
    in_treatment = .data$treatment > 0,
    in_thr = .data$THR > 0,
    gene_set_label = dplyr::case_when(
      .data$in_treatment & .data$in_thr ~ "shared_treatment_and_thr",
      .data$in_treatment & !.data$in_thr ~ "treatment_only",
      !.data$in_treatment & .data$in_thr ~ "thr_only",
      TRUE ~ "neither"
    )
  ) %>%
  dplyr::mutate(
    comparison_type = "within_strategy",
    comparison_group = .data$perturbation_strategy
  ) %>%
  dplyr::select(
    .data$comparison_type,
    .data$comparison_group,
    .data$gene,
    .data$gene_set_label
  )


#######################################
# STEP 3: ACROSS-STRATEGY COMBINED SETS
#
# Question:
#   Within each task, which genes are shared across perturbation strategies,
#   and which are strategy-specific?
#######################################

across_strategy_membership <- main_gene_sets %>%
  dplyr::filter(.data$in_top_main) %>%
  dplyr::select(.data$analysis_task, .data$perturbation_strategy, .data$gene) %>%
  dplyr::distinct() %>%
  tidyr::pivot_wider(
    names_from = .data$perturbation_strategy,
    values_from = .data$perturbation_strategy,
    values_fn = length,
    values_fill = 0
  ) %>%
  dplyr::mutate(
    cell_type = if ("cell_type" %in% colnames(.)) .data$cell_type else 0L,
    treatment = if ("treatment" %in% colnames(.)) .data$treatment else 0L,
    in_cell_type = .data$cell_type > 0,
    in_treatment_strategy = .data$treatment > 0,
    gene_set_label = dplyr::case_when(
      .data$in_cell_type & .data$in_treatment_strategy ~ "shared_across_strategies",
      .data$in_cell_type & !.data$in_treatment_strategy ~ "cell_type_only",
      !.data$in_cell_type & .data$in_treatment_strategy ~ "treatment_strategy_only",
      TRUE ~ "neither"
    )
  ) %>%
  dplyr::mutate(
    comparison_type = "across_strategy",
    comparison_group = .data$analysis_task
  ) %>%
  dplyr::select(
    .data$comparison_type,
    .data$comparison_group,
    .data$gene,
    .data$gene_set_label
  )


#######################################
# STEP 4: COMBINE BOTH MEMBERSHIP TABLES
#######################################

combined_gene_set_membership <- dplyr::bind_rows(
  within_strategy_membership,
  across_strategy_membership
)


#######################################
# STEP 5: SUMMARIES
#######################################

combined_gene_set_summary <- combined_gene_set_membership %>%
  dplyr::group_by(.data$comparison_type, .data$comparison_group, .data$gene_set_label) %>%
  dplyr::summarise(
    n_genes = dplyr::n_distinct(.data$gene),
    .groups = "drop"
  ) %>%
  dplyr::arrange(.data$comparison_type, .data$comparison_group, .data$gene_set_label)


#######################################
# STEP 6: SIMPLE GT ENRICHMENT FOR COMBINED SETS
#######################################

thyroid_gt_main <- gt_thyroid_response_standardized %>%
  dplyr::group_by(.data$gene) %>%
  dplyr::summarise(
    gt_positive = any(.data$sig_main, na.rm = TRUE),
    .groups = "drop"
  )

mmc2_selected <- gt_mmc2_long %>%
  dplyr::select(.data$gene, .data$gt_name, .data$binary_label_main)


#######################################
# STEP 6A: THYROID ENRICHMENT
#######################################

combined_thyroid_enrichment <- dplyr::bind_rows(
  lapply(seq_len(nrow(combined_gene_set_summary)), function(i) {

    row_i <- combined_gene_set_summary[i, ]

    tmp <- combined_gene_set_membership %>%
      dplyr::filter(
        .data$comparison_type == row_i$comparison_type,
        .data$comparison_group == row_i$comparison_group
      ) %>%
      dplyr::left_join(thyroid_gt_main, by = "gene") %>%
      dplyr::mutate(
        set_flag = .data$gene_set_label == row_i$gene_set_label
      )

    # Only need overlap/odds_ratio/fisher_p here; drop the extra precision/
    # recall/jaccard columns safe_fisher_set_vs_gt also returns.
    safe_fisher_set_vs_gt(tmp$set_flag, tmp$gt_positive) %>%
      dplyr::select(.data$odds_ratio, .data$fisher_p, .data$overlap) %>%
      dplyr::mutate(
        comparison_type = row_i$comparison_type,
        comparison_group = row_i$comparison_group,
        set_name = row_i$gene_set_label,
        gt_name = "thyroid_response_pooled_main"
      )
  })
)


#######################################
# STEP 6B: MMC2 ENRICHMENT
#######################################

combined_mmc2_enrichment <- dplyr::bind_rows(
  lapply(seq_len(nrow(combined_gene_set_summary)), function(i) {

    row_i <- combined_gene_set_summary[i, ]

    dplyr::bind_rows(
      lapply(unique(mmc2_selected$gt_name), function(gt_nm) {

        tmp <- combined_gene_set_membership %>%
          dplyr::filter(
            .data$comparison_type == row_i$comparison_type,
            .data$comparison_group == row_i$comparison_group
          ) %>%
          dplyr::left_join(
            mmc2_selected %>% dplyr::filter(.data$gt_name == gt_nm),
            by = "gene"
          ) %>%
          dplyr::mutate(
            set_flag = .data$gene_set_label == row_i$gene_set_label
          )

        safe_fisher_set_vs_gt(tmp$set_flag, tmp$binary_label_main) %>%
          dplyr::select(.data$odds_ratio, .data$fisher_p, .data$overlap) %>%
          dplyr::mutate(
            comparison_type = row_i$comparison_type,
            comparison_group = row_i$comparison_group,
            set_name = row_i$gene_set_label,
            gt_name = gt_nm
          )
      })
    )
  })
)


combined_gene_set_enrichment <- dplyr::bind_rows(
  combined_thyroid_enrichment,
  combined_mmc2_enrichment
) %>%
  dplyr::arrange(.data$comparison_type, .data$comparison_group, .data$set_name, .data$gt_name)


#######################################
# SAVE OUTPUTS
#######################################

readr::write_csv(
  combined_gene_set_membership,
  file = here::here("results", "tables", "combined_gene_set_membership.csv")
)

readr::write_csv(
  combined_gene_set_summary,
  file = here::here("results", "tables", "combined_gene_set_summary.csv")
)

readr::write_csv(
  combined_gene_set_enrichment,
  file = here::here("results", "tables", "combined_gene_set_enrichment.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Saved combined gene-set analysis.")
print(combined_gene_set_summary)
print(utils::head(combined_gene_set_enrichment, 20))
