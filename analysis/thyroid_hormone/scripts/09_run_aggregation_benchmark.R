################################################################################
# Script: 09_run_aggregation_benchmark.R
#
# Purpose:
#   Run all aggregation methods across all CASCADE outputs and benchmark them
#   against gene-level external ground truths.
#
# Outputs:
#   - results/rds/aggregation_results_all_methods.rds
#   - results/rds/aggregation_benchmark_results.rds
#   - results/tables/aggregation_result_summary.csv
#   - results/tables/aggregation_benchmark_results.csv
################################################################################


#######################################
# LOAD CONFIG
#######################################

select <- dplyr::select
mutate <- dplyr::mutate
arrange <- dplyr::arrange
source(here::here("scripts", "00_config.R"))
source(here::here("scripts", "08_aggregation_functions.R"))
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

cascade_outputs_filtered_variants <- readRDS(
  here::here("results", "rds", "cascade_outputs_filtered_variants.rds")
)

gt_thyroid_response_standardized <- readRDS(
  here::here("results", "rds", "gt_thyroid_response_standardized.rds")
)

gt_mmc2_long <- readRDS(
  here::here("results", "rds", "gt_mmc2_long_mapped_final.rds")
)


#######################################
# METHODS TO TEST
#######################################

AGGREGATION_METHODS <- c(
  "combmnz",
  "borda",
  "mrr",
  "topk_10",
  "topk_25",
  "topk_50",
  "topk_100",
  "topk_200",
  "geom_mean_rank",
  "weighted_borda"
)


#######################################
# HELPER: PER-GT BENCHMARK
#######################################

benchmark_one_gt <- function(agg_df, gt_df, gt_name, label_col) {

  merged <- agg_df %>%
    left_join(gt_df, by = "gene")

  auc_value <- safe_auc(
    labels = merged[[label_col]],
    scores = if ("aggregated_score" %in% colnames(merged)) merged$aggregated_score else NA_real_
  )

  fisher_tbl <- bind_rows(
    lapply(
      ANALYSIS_PARAMS$overlap_top_n,
      function(n) safe_fisher_from_topn(
        labels = merged[[label_col]],
        ranks = merged$aggregated_rank,
        top_n = n
      )
    )
  )

  fisher_tbl %>%
    mutate(
      auc = auc_value,
      gt_name = gt_name,
      label_definition = label_col
    )
}


#######################################
# STEP 1: RUN ALL AGGREGATIONS
#######################################

grouped_inputs <- cascade_outputs_filtered_variants %>%
  group_by(.data$dataset_name, .data$analysis_task, .data$perturbation_strategy, .data$filter_mode) %>%
  group_split()

aggregation_results_all_methods <- dplyr::bind_rows(
  lapply(grouped_inputs, function(df_part) {

    meta <- df_part %>%
      dplyr::distinct(dataset_name, analysis_task, perturbation_strategy, filter_mode)

    dplyr::bind_rows(
      lapply(AGGREGATION_METHODS, function(m) {
        run_aggregation_method(df_part, m) %>%
          dplyr::mutate(
            dataset_name = meta$dataset_name,
            analysis_task = meta$analysis_task,
            perturbation_strategy = meta$perturbation_strategy,
            filter_mode = meta$filter_mode
          ) %>%
          dplyr::select(
            dataset_name,
            analysis_task,
            perturbation_strategy,
            filter_mode,
            aggregation_method,
            dplyr::everything()
          )
      })
    )
  })
)


#######################################
# STEP 2: BENCHMARK AGAINST GTS
#######################################

thyroid_gt_binary <- build_thyroid_gt_binary_sets(gt_thyroid_response_standardized)

aggregation_groups <- aggregation_results_all_methods %>%
  group_by(.data$dataset_name, .data$analysis_task, .data$perturbation_strategy,
           .data$filter_mode, .data$aggregation_method) %>%
  group_split()

aggregation_benchmark_results <- bind_rows(
  lapply(aggregation_groups, function(agg_df) {

    meta <- agg_df %>%
      distinct(.data$dataset_name, .data$analysis_task, .data$perturbation_strategy,
               .data$filter_mode, .data$aggregation_method)

    # Thyroid pooled GTs
    thyroid_results <- bind_rows(
      benchmark_one_gt(
        agg_df = agg_df,
        gt_df = thyroid_gt_binary %>% filter(.data$gt_name == "thyroid_response_pooled_main"),
        gt_name = "thyroid_response_pooled_main",
        label_col = "binary_label_main"
      ),
      benchmark_one_gt(
        agg_df = agg_df,
        gt_df = thyroid_gt_binary %>% filter(.data$gt_name == "thyroid_response_pooled_top_abs_score"),
        gt_name = "thyroid_response_pooled_top_abs_score",
        label_col = "binary_label_main"
      )
    )

    # MMC2 GTs
    mmc2_results <- bind_rows(
      lapply(unique(gt_mmc2_long$gt_name), function(gt_nm) {

        gt_sub <- gt_mmc2_long %>%
          filter(.data$gt_name == gt_nm)

        benchmark_one_gt(
          agg_df = agg_df,
          gt_df = gt_sub,
          gt_name = gt_nm,
          label_col = "binary_label_main"
        )
      })
    )

    bind_rows(thyroid_results, mmc2_results) %>%
      mutate(
        dataset_name = meta$dataset_name,
        analysis_task = meta$analysis_task,
        perturbation_strategy = meta$perturbation_strategy,
        filter_mode = meta$filter_mode,
        aggregation_method = meta$aggregation_method
      ) %>%
      select(
        .data$dataset_name, .data$analysis_task, .data$perturbation_strategy,
        .data$filter_mode, .data$aggregation_method, dplyr::everything()
      )
  })
)


#######################################
# STEP 3: SUMMARIES
#######################################

aggregation_result_summary <- aggregation_results_all_methods %>%
  group_by(.data$dataset_name, .data$analysis_task, .data$perturbation_strategy,
           .data$filter_mode, .data$aggregation_method) %>%
  summarise(
    n_genes = n_distinct(.data$gene),
    max_score = max(.data$aggregated_score, na.rm = TRUE),
    min_score = min(.data$aggregated_score, na.rm = TRUE),
    .groups = "drop"
  )


#######################################
# SAVE OUTPUTS
#######################################

saveRDS(
  aggregation_results_all_methods,
  file = here::here("results", "rds", "aggregation_results_all_methods.rds")
)

saveRDS(
  aggregation_benchmark_results,
  file = here::here("results", "rds", "aggregation_benchmark_results.rds")
)

readr::write_csv(
  aggregation_result_summary,
  file = here::here("results", "tables", "aggregation_result_summary.csv")
)

readr::write_csv(
  aggregation_benchmark_results,
  file = here::here("results", "tables", "aggregation_benchmark_results.csv")
)

message("Saved aggregation benchmark results.")
print(aggregation_result_summary)
