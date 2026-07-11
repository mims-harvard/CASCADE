################################################################################
# Script: 11_run_cascade_gene_level_analysis.R
#
# Purpose:
#   Run the main CASCADE gene-level validation analysis using the selected
#   aggregation + filter combination.
#
# Outputs:
#   - results/rds/cascade_main_selected_aggregation.rds
#   - results/tables/cascade_main_selected_config.csv
#   - results/tables/cascade_gene_level_results.csv
#   - results/tables/cascade_gene_level_summary.csv
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
  library(pROC)
})


#######################################
# LOAD INPUT OBJECTS
#######################################

aggregation_results_all_methods <- readRDS(
  here::here("results", "rds", "aggregation_results_all_methods.rds")
)

gt_thyroid_response_standardized <- readRDS(
  here::here("results", "rds", "gt_thyroid_response_standardized.rds")
)

gt_mmc2_long <- readRDS(
  here::here("results", "rds", "gt_mmc2_long_mapped_final.rds")
)

best_choice <- read.csv(
  here::here("results", "tables", "aggregation_filter_best_choice.csv"),
  stringsAsFactors = FALSE
)


#######################################
# STEP 1: FREEZE SELECTED CONFIGURATION
#######################################

selected_method <- best_choice$aggregation_method[1]
selected_filter <- best_choice$filter_mode[1]

cascade_main_selected_aggregation <- aggregation_results_all_methods %>%
  dplyr::filter(
    .data$aggregation_method == selected_method,
    .data$filter_mode == selected_filter
  )

cascade_main_selected_config <- tibble::tibble(
  selected_aggregation_method = selected_method,
  selected_filter_mode = selected_filter
)


#######################################
# STEP 2: BUILD GT OBJECTS
#######################################

thyroid_gt_binary <- build_thyroid_gt_binary_sets(gt_thyroid_response_standardized)


#######################################
# STEP 3: RUN GENE-LEVEL ANALYSIS
#######################################

selected_groups <- cascade_main_selected_aggregation %>%
  dplyr::group_by(.data$dataset_name, .data$analysis_task, .data$perturbation_strategy) %>%
  dplyr::group_split()

cascade_gene_level_results <- dplyr::bind_rows(
  lapply(selected_groups, function(agg_df) {

    meta <- agg_df %>%
      dplyr::distinct(dataset_name, analysis_task, perturbation_strategy, aggregation_method, filter_mode)

    # Thyroid pooled binary GTs
    thyroid_binary_results <- dplyr::bind_rows(
      lapply(unique(thyroid_gt_binary$gt_name), function(gt_nm) {

        gt_sub <- thyroid_gt_binary %>%
          dplyr::filter(.data$gt_name == gt_nm)

        merged <- agg_df %>%
          dplyr::left_join(gt_sub, by = "gene")

        auc_value <- safe_auc(
          labels = merged$binary_label_main,
          scores = merged$aggregated_score
        )

        fisher_tbl <- dplyr::bind_rows(
          lapply(
            ANALYSIS_PARAMS$overlap_top_n,
            function(n) safe_fisher_from_topn(
              labels = merged$binary_label_main,
              ranks = merged$aggregated_rank,
              top_n = n
            )
          )
        )

        fisher_tbl %>%
          dplyr::mutate(
            result_family = "thyroid_binary",
            gt_name = gt_nm,
            gt_type = "binary",
            auc = auc_value,
            spearman_rho = NA_real_
          )
      })
    )

    # MMC2 GTs
    mmc2_results <- dplyr::bind_rows(
      lapply(unique(gt_mmc2_long$gt_name), function(gt_nm) {

        gt_sub <- gt_mmc2_long %>%
          dplyr::filter(.data$gt_name == gt_nm)

        merged <- agg_df %>%
          dplyr::left_join(gt_sub, by = "gene")

        auc_value <- safe_auc(
          labels = merged$binary_label_main,
          scores = merged$aggregated_score
        )

        spearman_value <- if (unique(gt_sub$gt_type)[1] == "continuous") {
          safe_spearman(
            x = merged$aggregated_score,
            y = merged$abs_score
          )
        } else {
          NA_real_
        }

        fisher_tbl <- dplyr::bind_rows(
          lapply(
            ANALYSIS_PARAMS$overlap_top_n,
            function(n) safe_fisher_from_topn(
              labels = merged$binary_label_main,
              ranks = merged$aggregated_rank,
              top_n = n
            )
          )
        )

        fisher_tbl %>%
          dplyr::mutate(
            result_family = "mmc2",
            gt_name = gt_nm,
            gt_type = unique(gt_sub$gt_type)[1],
            auc = auc_value,
            spearman_rho = spearman_value
          )
      })
    )

    dplyr::bind_rows(thyroid_binary_results, mmc2_results) %>%
      dplyr::mutate(
        dataset_name = meta$dataset_name,
        analysis_task = meta$analysis_task,
        perturbation_strategy = meta$perturbation_strategy,
        aggregation_method = meta$aggregation_method,
        filter_mode = meta$filter_mode
      ) %>%
      dplyr::select(
        dataset_name,
        analysis_task,
        perturbation_strategy,
        aggregation_method,
        filter_mode,
        result_family,
        gt_name,
        gt_type,
        top_n,
        auc,
        spearman_rho,
        odds_ratio,
        fisher_p,
        overlap,
        n_top,
        n_positive
      )
  })
)


#######################################
# STEP 4: SUMMARISE RESULTS
#######################################

cascade_gene_level_summary <- cascade_gene_level_results %>%
  dplyr::group_by(
    .data$dataset_name,
    .data$analysis_task,
    .data$perturbation_strategy,
    .data$result_family,
    .data$gt_name,
    .data$gt_type
  ) %>%
  dplyr::summarise(
    auc = max(.data$auc, na.rm = TRUE),
    spearman_rho = max(.data$spearman_rho, na.rm = TRUE),
    max_odds_ratio = max(.data$odds_ratio, na.rm = TRUE),
    min_fisher_p = min(.data$fisher_p, na.rm = TRUE),
    best_overlap = max(.data$overlap, na.rm = TRUE),
    .groups = "drop"
  )


#######################################
# SAVE OUTPUTS
#######################################

saveRDS(
  cascade_main_selected_aggregation,
  file = here::here("results", "rds", "cascade_main_selected_aggregation.rds")
)

readr::write_csv(
  cascade_main_selected_config,
  file = here::here("results", "tables", "cascade_main_selected_config.csv")
)

readr::write_csv(
  cascade_gene_level_results,
  file = here::here("results", "tables", "cascade_gene_level_results.csv")
)

readr::write_csv(
  cascade_gene_level_summary,
  file = here::here("results", "tables", "cascade_gene_level_summary.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Saved CASCADE main gene-level analysis.")
print(cascade_main_selected_config)
print(utils::head(cascade_gene_level_summary, 20))
