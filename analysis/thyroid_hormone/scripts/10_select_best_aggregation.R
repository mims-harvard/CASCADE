################################################################################
# Script: 10_select_best_aggregation.R
#
# Purpose:
#   Select the best aggregation × filtering combination using benchmark results.
#
# Selection rule:
#   1. Primary: median AUROC across GTs
#   2. Secondary: mean AUROC across GTs
#   3. Tertiary: median log(odds ratio) across top-N enrichments
#
# Outputs:
#   - results/tables/aggregation_filter_combo_ranking.csv
#   - results/tables/aggregation_filter_best_choice.csv
#   - results/tables/aggregation_method_ranking.csv
#   - results/tables/filter_mode_ranking.csv
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
})


#######################################
# LOAD INPUT
#######################################

aggregation_benchmark_results <- readRDS(
  here::here("results", "rds", "aggregation_benchmark_results.rds")
)


#######################################
# STEP 1: CLEAN METRICS
#######################################

benchmark_clean <- aggregation_benchmark_results %>%
  dplyr::mutate(
    log_odds_ratio = log(.data$odds_ratio),
    log_odds_ratio = dplyr::if_else(
      is.infinite(.data$log_odds_ratio),
      NA_real_,
      .data$log_odds_ratio
    )
  )


#######################################
# STEP 2: RANK AGGREGATION × FILTER COMBINATIONS
#######################################

aggregation_filter_combo_ranking <- benchmark_clean %>%
  dplyr::group_by(.data$aggregation_method, .data$filter_mode) %>%
  dplyr::summarise(
    median_auc = median(.data$auc, na.rm = TRUE),
    mean_auc = mean(.data$auc, na.rm = TRUE),
    median_log_odds_ratio = median(.data$log_odds_ratio, na.rm = TRUE),
    mean_log_odds_ratio = mean(.data$log_odds_ratio, na.rm = TRUE),
    n_benchmarks = dplyr::n(),
    n_nonmissing_auc = sum(!is.na(.data$auc)),
    n_nonmissing_or = sum(!is.na(.data$odds_ratio)),
    .groups = "drop"
  ) %>%
  dplyr::arrange(
    dplyr::desc(.data$median_auc),
    dplyr::desc(.data$mean_auc),
    dplyr::desc(.data$median_log_odds_ratio),
    dplyr::desc(.data$mean_log_odds_ratio)
  )

aggregation_filter_best_choice <- aggregation_filter_combo_ranking %>%
  dplyr::slice_head(n = 1)


#######################################
# STEP 3: OPTIONAL MARGINAL RANKINGS
#######################################

aggregation_method_ranking <- benchmark_clean %>%
  dplyr::group_by(.data$aggregation_method) %>%
  dplyr::summarise(
    median_auc = median(.data$auc, na.rm = TRUE),
    mean_auc = mean(.data$auc, na.rm = TRUE),
    median_log_odds_ratio = median(.data$log_odds_ratio, na.rm = TRUE),
    mean_log_odds_ratio = mean(.data$log_odds_ratio, na.rm = TRUE),
    n_benchmarks = dplyr::n(),
    .groups = "drop"
  ) %>%
  dplyr::arrange(
    dplyr::desc(.data$median_auc),
    dplyr::desc(.data$mean_auc),
    dplyr::desc(.data$median_log_odds_ratio)
  )

filter_mode_ranking <- benchmark_clean %>%
  dplyr::group_by(.data$filter_mode) %>%
  dplyr::summarise(
    median_auc = median(.data$auc, na.rm = TRUE),
    mean_auc = mean(.data$auc, na.rm = TRUE),
    median_log_odds_ratio = median(.data$log_odds_ratio, na.rm = TRUE),
    mean_log_odds_ratio = mean(.data$log_odds_ratio, na.rm = TRUE),
    n_benchmarks = dplyr::n(),
    .groups = "drop"
  ) %>%
  dplyr::arrange(
    dplyr::desc(.data$median_auc),
    dplyr::desc(.data$mean_auc),
    dplyr::desc(.data$median_log_odds_ratio)
  )


#######################################
# STEP 4: SAVE OUTPUTS
#######################################

readr::write_csv(
  aggregation_filter_combo_ranking,
  file = here::here("results", "tables", "aggregation_filter_combo_ranking.csv")
)

readr::write_csv(
  aggregation_filter_best_choice,
  file = here::here("results", "tables", "aggregation_filter_best_choice.csv")
)

readr::write_csv(
  aggregation_method_ranking,
  file = here::here("results", "tables", "aggregation_method_ranking.csv")
)

readr::write_csv(
  filter_mode_ranking,
  file = here::here("results", "tables", "filter_mode_ranking.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Best aggregation × filter combination:")
print(aggregation_filter_best_choice)

message("Top aggregation × filter combinations:")
print(utils::head(aggregation_filter_combo_ranking, 10))

message("Marginal aggregation ranking:")
print(aggregation_method_ranking)

message("Marginal filter ranking:")
print(filter_mode_ranking)
