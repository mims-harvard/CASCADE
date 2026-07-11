################################################################################
# Script: 16_run_cascade_vs_de_hitset_comparison.R
#
# Purpose:
#   Compare CASCADE top-gene sets against significant DE hit sets using the
#   same mapped gene universe, across ALL CASCADE aggregation × filter combos
#   and multiple CASCADE top-N thresholds.
#
# Main question:
#   Is CASCADE better than simply extracting significant DE genes?
#
# Main comparison:
#   - CASCADE: top-N genes (N = 25, 50, 100, 200, 500)
#   - DE: significant DE hit set after universe restriction
#
# Metrics:
#   - Fisher exact odds ratio / p-value
#   - overlap count
#   - precision
#   - recall
#   - Jaccard index
#
# Outputs:
#   - results/tables/cascade_vs_de_hitset_results.csv
#   - results/tables/cascade_vs_de_hitset_summary.csv
#   - results/tables/cascade_vs_de_hitset_winner_table.csv
#   - results/tables/cascade_vs_de_hitset_combo_summary.csv
#   - results/tables/cascade_vs_de_hitset_gt_summary.csv
#   - results/tables/cascade_vs_de_hitset_topn_summary.csv
#   - results/tables/de_hitset_summary.csv
#   - results/tables/cascade_hitset_universe_completion_audit.csv
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
})


#######################################
# LOAD INPUT OBJECTS
#######################################

aggregation_results_all_methods <- readRDS(
  here::here("results", "rds", "aggregation_results_all_methods.rds")
)

de_baselines_restricted_to_cascade_universe <- readRDS(
  here::here("results", "rds", "de_baselines_restricted_to_cascade_universe.rds")
)

cascade_gene_universes <- readRDS(
  here::here("results", "rds", "cascade_gene_universes.rds")
)

gt_thyroid_response_standardized <- readRDS(
  here::here("results", "rds", "gt_thyroid_response_standardized.rds")
)

gt_mmc2_long <- readRDS(
  here::here("results", "rds", "gt_mmc2_long_mapped_final.rds")
)


#######################################
# PARAMETERS
#######################################

DE_P_ADJ_THRESH <- 0.05
DE_ABS_LOGFC_THRESH <- 0.5

CASCADE_TOP_N_VALUES <- c(25, 50, 100, 200, 500)


#######################################
# STEP 1: BUILD GT OBJECTS
#######################################

thyroid_gt_binary <- build_thyroid_gt_binary_sets(gt_thyroid_response_standardized)

all_gt_binary <- dplyr::bind_rows(
  thyroid_gt_binary %>%
    dplyr::mutate(result_family = "thyroid_binary"),
  gt_mmc2_long %>%
    dplyr::select(.data$gene, .data$gt_name, .data$gt_type, .data$binary_label_main) %>%
    dplyr::mutate(result_family = "mmc2")
)


#######################################
# STEP 2: TASK/STRATEGY UNIVERSES
#######################################

cascade_task_strategy_universe <- cascade_gene_universes %>%
  dplyr::filter(.data$analysis_task %in% c("treatment", "THR")) %>%
  dplyr::distinct(.data$analysis_task, .data$perturbation_strategy, .data$gene)


#######################################
# STEP 3: COMPLETE ALL CASCADE COMBOS
#######################################

cascade_completed_list <- lapply(
  aggregation_results_all_methods %>%
    dplyr::group_by(
      .data$aggregation_method,
      .data$filter_mode,
      .data$analysis_task,
      .data$perturbation_strategy
    ) %>%
    dplyr::group_split(),
  function(df_part) complete_cascade_combo_to_universe(df_part, cascade_task_strategy_universe)
)

cascade_completed_all_combos <- dplyr::bind_rows(
  lapply(cascade_completed_list, `[[`, "data")
)

cascade_hitset_universe_completion_audit <- dplyr::bind_rows(
  lapply(cascade_completed_list, `[[`, "audit")
)


#######################################
# STEP 4: BUILD DE HIT SETS
#######################################

de_hit_sets <- de_baselines_restricted_to_cascade_universe %>%
  dplyr::mutate(
    perturbation_strategy = .data$comparison_universe,
    is_de_hit = .data$p_adj < DE_P_ADJ_THRESH & abs(.data$log2fc) > DE_ABS_LOGFC_THRESH
  ) %>%
  dplyr::group_by(
    .data$analysis_task,
    .data$perturbation_strategy,
    .data$baseline_type,
    .data$scope
  ) %>%
  dplyr::group_split() %>%
  lapply(function(df_part) {

    meta <- df_part %>%
      dplyr::distinct(
        .data$analysis_task,
        .data$perturbation_strategy,
        .data$baseline_type,
        .data$scope
      )

    full_universe <- cascade_task_strategy_universe %>%
      dplyr::filter(
        .data$analysis_task == meta$analysis_task,
        .data$perturbation_strategy == meta$perturbation_strategy
      ) %>%
      dplyr::distinct(.data$gene)

    full_universe %>%
      dplyr::left_join(
        df_part %>%
          dplyr::select(.data$gene, .data$is_de_hit),
        by = "gene"
      ) %>%
      dplyr::mutate(
        is_de_hit = dplyr::if_else(is.na(.data$is_de_hit), FALSE, .data$is_de_hit),
        analysis_task = meta$analysis_task,
        perturbation_strategy = meta$perturbation_strategy,
        baseline_type = meta$baseline_type,
        scope = meta$scope,
        source_label = dplyr::case_when(
          meta$baseline_type == "pooled" ~ paste0("DE_pooled_", meta$analysis_task),
          TRUE ~ paste0("DE_scoped_", meta$scope, "_", meta$analysis_task)
        )
      ) %>%
      dplyr::select(
        .data$analysis_task,
        .data$perturbation_strategy,
        .data$baseline_type,
        .data$scope,
        .data$source_label,
        .data$gene,
        .data$is_de_hit
      )
  }) %>%
  dplyr::bind_rows()

de_hitset_summary <- de_hit_sets %>%
  dplyr::group_by(
    .data$analysis_task,
    .data$perturbation_strategy,
    .data$baseline_type,
    .data$scope,
    .data$source_label
  ) %>%
  dplyr::summarise(
    n_genes_in_universe = dplyr::n(),
    n_de_hits = sum(.data$is_de_hit, na.rm = TRUE),
    prop_de_hits = mean(.data$is_de_hit, na.rm = TRUE),
    .groups = "drop"
  )


#######################################
# STEP 5: CASCADE TOP-N SETS VS GT
#######################################

cascade_hitset_results <- dplyr::bind_rows(
  lapply(
    cascade_completed_all_combos %>%
      dplyr::group_by(
        .data$aggregation_method,
        .data$filter_mode,
        .data$analysis_task,
        .data$perturbation_strategy
      ) %>%
      dplyr::group_split(),
    function(df_part) {

      meta <- df_part %>%
        dplyr::distinct(
          .data$aggregation_method,
          .data$filter_mode,
          .data$analysis_task,
          .data$perturbation_strategy
        )

      dplyr::bind_rows(
        lapply(CASCADE_TOP_N_VALUES, function(top_n_val) {

          cascade_set <- df_part %>%
            dplyr::mutate(in_cascade_set = .data$aggregated_rank <= top_n_val) %>%
            dplyr::select(.data$gene, .data$in_cascade_set)

          dplyr::bind_rows(
            lapply(unique(all_gt_binary$gt_name), function(gt_nm) {

              gt_sub <- all_gt_binary %>%
                dplyr::filter(.data$gt_name == gt_nm)

              merged <- cascade_set %>%
                dplyr::left_join(gt_sub, by = "gene")

              safe_fisher_set_vs_gt(
                set_flag = merged$in_cascade_set,
                gt_flag = merged$binary_label_main
              ) %>%
                dplyr::mutate(
                  source_method = "CASCADE",
                  source_label = paste0("CASCADE_", meta$aggregation_method, "_", meta$filter_mode),
                  aggregation_method = meta$aggregation_method,
                  filter_mode = meta$filter_mode,
                  analysis_task = meta$analysis_task,
                  perturbation_strategy = meta$perturbation_strategy,
                  baseline_type = NA_character_,
                  scope = NA_character_,
                  set_definition = paste0("top_", top_n_val),
                  cascade_top_n = top_n_val,
                  result_family = unique(gt_sub$result_family)[1],
                  gt_name = gt_nm,
                  gt_type = unique(gt_sub$gt_type)[1]
                )
            })
          )
        })
      )
    }
  )
)


#######################################
# STEP 6: DE HIT SETS VS GT
#######################################

de_hitset_results <- dplyr::bind_rows(
  lapply(
    de_hit_sets %>%
      dplyr::group_by(
        .data$analysis_task,
        .data$perturbation_strategy,
        .data$baseline_type,
        .data$scope,
        .data$source_label
      ) %>%
      dplyr::group_split(),
    function(df_part) {

      meta <- df_part %>%
        dplyr::distinct(
          .data$analysis_task,
          .data$perturbation_strategy,
          .data$baseline_type,
          .data$scope,
          .data$source_label
        )

      dplyr::bind_rows(
        lapply(unique(all_gt_binary$gt_name), function(gt_nm) {

          gt_sub <- all_gt_binary %>%
            dplyr::filter(.data$gt_name == gt_nm)

          merged <- df_part %>%
            dplyr::left_join(gt_sub, by = "gene")

          safe_fisher_set_vs_gt(
            set_flag = merged$is_de_hit,
            gt_flag = merged$binary_label_main
          ) %>%
            dplyr::mutate(
              source_method = "DE",
              source_label = meta$source_label,
              aggregation_method = NA_character_,
              filter_mode = NA_character_,
              analysis_task = meta$analysis_task,
              perturbation_strategy = meta$perturbation_strategy,
              baseline_type = meta$baseline_type,
              scope = meta$scope,
              set_definition = paste0(
                "de_hits_padj_", DE_P_ADJ_THRESH,
                "_abslogfc_", DE_ABS_LOGFC_THRESH
              ),
              cascade_top_n = NA_real_,
              result_family = unique(gt_sub$result_family)[1],
              gt_name = gt_nm,
              gt_type = unique(gt_sub$gt_type)[1]
            )
        })
      )
    }
  )
)


#######################################
# STEP 7: COMBINE RESULTS
#######################################

cascade_vs_de_hitset_results <- dplyr::bind_rows(
  cascade_hitset_results,
  de_hitset_results
) %>%
  dplyr::select(
    .data$source_method,
    .data$source_label,
    .data$aggregation_method,
    .data$filter_mode,
    .data$analysis_task,
    .data$perturbation_strategy,
    .data$baseline_type,
    .data$scope,
    .data$set_definition,
    .data$cascade_top_n,
    .data$result_family,
    .data$gt_name,
    .data$gt_type,
    .data$odds_ratio,
    .data$fisher_p,
    .data$overlap,
    .data$set_size,
    .data$gt_size,
    .data$precision,
    .data$recall,
    .data$jaccard
  )


#######################################
# STEP 8: SUMMARY
#######################################

cascade_vs_de_hitset_summary <- cascade_vs_de_hitset_results


#######################################
# STEP 9: WINNER TABLE
#######################################

cascade_rows <- cascade_vs_de_hitset_results %>%
  dplyr::filter(.data$source_method == "CASCADE")

pooled_de_rows <- cascade_vs_de_hitset_results %>%
  dplyr::filter(.data$source_method == "DE", .data$baseline_type == "pooled") %>%
  dplyr::select(
    .data$analysis_task,
    .data$perturbation_strategy,
    .data$result_family,
    .data$gt_name,
    .data$gt_type,
    pooled_de_label = .data$source_label,
    pooled_de_odds_ratio = .data$odds_ratio,
    pooled_de_fisher_p = .data$fisher_p,
    pooled_de_overlap = .data$overlap,
    pooled_de_set_size = .data$set_size,
    pooled_de_precision = .data$precision,
    pooled_de_recall = .data$recall,
    pooled_de_jaccard = .data$jaccard
  )

best_scoped_de_rows <- cascade_vs_de_hitset_results %>%
  dplyr::filter(.data$source_method == "DE", .data$baseline_type != "pooled") %>%
  dplyr::group_by(
    .data$analysis_task,
    .data$perturbation_strategy,
    .data$result_family,
    .data$gt_name,
    .data$gt_type
  ) %>%
  dplyr::arrange(
    dplyr::desc(.data$odds_ratio),
    dplyr::desc(.data$recall),
    dplyr::desc(.data$precision),
    .by_group = TRUE
  ) %>%
  dplyr::slice_head(n = 1) %>%
  dplyr::ungroup() %>%
  dplyr::select(
    .data$analysis_task,
    .data$perturbation_strategy,
    .data$result_family,
    .data$gt_name,
    .data$gt_type,
    best_scoped_de_label = .data$source_label,
    best_scoped_de_odds_ratio = .data$odds_ratio,
    best_scoped_de_fisher_p = .data$fisher_p,
    best_scoped_de_overlap = .data$overlap,
    best_scoped_de_set_size = .data$set_size,
    best_scoped_de_precision = .data$precision,
    best_scoped_de_recall = .data$recall,
    best_scoped_de_jaccard = .data$jaccard
  )

cascade_vs_de_hitset_winner_table <- cascade_rows %>%
  dplyr::left_join(
    pooled_de_rows,
    by = c(
      "analysis_task",
      "perturbation_strategy",
      "result_family",
      "gt_name",
      "gt_type"
    )
  ) %>%
  dplyr::left_join(
    best_scoped_de_rows,
    by = c(
      "analysis_task",
      "perturbation_strategy",
      "result_family",
      "gt_name",
      "gt_type"
    )
  ) %>%
  dplyr::mutate(
    winner_vs_pooled_de_or = dplyr::case_when(
      is.na(.data$pooled_de_odds_ratio) ~ "CASCADE",
      is.na(.data$odds_ratio) ~ "pooled_DE",
      .data$odds_ratio > .data$pooled_de_odds_ratio ~ "CASCADE",
      .data$odds_ratio < .data$pooled_de_odds_ratio ~ "pooled_DE",
      TRUE ~ "tie"
    ),
    winner_vs_pooled_de_recall = dplyr::case_when(
      is.na(.data$pooled_de_recall) ~ "CASCADE",
      is.na(.data$recall) ~ "pooled_DE",
      .data$recall > .data$pooled_de_recall ~ "CASCADE",
      .data$recall < .data$pooled_de_recall ~ "pooled_DE",
      TRUE ~ "tie"
    ),
    winner_vs_best_scoped_de_or = dplyr::case_when(
      is.na(.data$best_scoped_de_odds_ratio) ~ "CASCADE",
      is.na(.data$odds_ratio) ~ "best_scoped_DE",
      .data$odds_ratio > .data$best_scoped_de_odds_ratio ~ "CASCADE",
      .data$odds_ratio < .data$best_scoped_de_odds_ratio ~ "best_scoped_DE",
      TRUE ~ "tie"
    ),
    winner_vs_best_scoped_de_recall = dplyr::case_when(
      is.na(.data$best_scoped_de_recall) ~ "CASCADE",
      is.na(.data$recall) ~ "best_scoped_DE",
      .data$recall > .data$best_scoped_de_recall ~ "CASCADE",
      .data$recall < .data$best_scoped_de_recall ~ "best_scoped_DE",
      TRUE ~ "tie"
    )
  )


#######################################
# STEP 10: COMBO SUMMARY
#######################################

cascade_vs_de_hitset_combo_summary <- cascade_vs_de_hitset_winner_table %>%
  dplyr::group_by(.data$aggregation_method, .data$filter_mode, .data$cascade_top_n) %>%
  dplyr::summarise(
    n_rows = dplyr::n(),
    n_or_wins_vs_pooled = sum(.data$winner_vs_pooled_de_or == "CASCADE", na.rm = TRUE),
    n_recall_wins_vs_pooled = sum(.data$winner_vs_pooled_de_recall == "CASCADE", na.rm = TRUE),
    prop_or_wins_vs_pooled = mean(.data$winner_vs_pooled_de_or == "CASCADE", na.rm = TRUE),
    prop_recall_wins_vs_pooled = mean(.data$winner_vs_pooled_de_recall == "CASCADE", na.rm = TRUE),
    mean_cascade_or = mean(.data$odds_ratio, na.rm = TRUE),
    mean_cascade_recall = mean(.data$recall, na.rm = TRUE),
    mean_pooled_de_or = mean(.data$pooled_de_odds_ratio, na.rm = TRUE),
    mean_pooled_de_recall = mean(.data$pooled_de_recall, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(
    dplyr::desc(.data$prop_or_wins_vs_pooled),
    dplyr::desc(.data$prop_recall_wins_vs_pooled),
    dplyr::desc(.data$mean_cascade_or)
  )


#######################################
# STEP 11: GT SUMMARY
#######################################

cascade_vs_de_hitset_gt_summary <- cascade_vs_de_hitset_winner_table %>%
  dplyr::group_by(.data$result_family, .data$gt_name, .data$gt_type) %>%
  dplyr::summarise(
    n_rows = dplyr::n(),
    n_cascade_or_wins_vs_pooled = sum(.data$winner_vs_pooled_de_or == "CASCADE", na.rm = TRUE),
    n_cascade_recall_wins_vs_pooled = sum(.data$winner_vs_pooled_de_recall == "CASCADE", na.rm = TRUE),
    prop_cascade_or_wins_vs_pooled = mean(.data$winner_vs_pooled_de_or == "CASCADE", na.rm = TRUE),
    prop_cascade_recall_wins_vs_pooled = mean(.data$winner_vs_pooled_de_recall == "CASCADE", na.rm = TRUE),
    median_cascade_or = median(.data$odds_ratio, na.rm = TRUE),
    median_cascade_recall = median(.data$recall, na.rm = TRUE),
    median_pooled_de_or = median(.data$pooled_de_odds_ratio, na.rm = TRUE),
    median_pooled_de_recall = median(.data$pooled_de_recall, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(
    dplyr::desc(.data$prop_cascade_or_wins_vs_pooled),
    dplyr::desc(.data$prop_cascade_recall_wins_vs_pooled)
  )


#######################################
# STEP 12: TOP-N SUMMARY
#######################################

cascade_vs_de_hitset_topn_summary <- cascade_vs_de_hitset_winner_table %>%
  dplyr::group_by(.data$cascade_top_n) %>%
  dplyr::summarise(
    n_rows = dplyr::n(),
    n_cascade_or_wins_vs_pooled = sum(.data$winner_vs_pooled_de_or == "CASCADE", na.rm = TRUE),
    n_cascade_recall_wins_vs_pooled = sum(.data$winner_vs_pooled_de_recall == "CASCADE", na.rm = TRUE),
    prop_cascade_or_wins_vs_pooled = mean(.data$winner_vs_pooled_de_or == "CASCADE", na.rm = TRUE),
    prop_cascade_recall_wins_vs_pooled = mean(.data$winner_vs_pooled_de_recall == "CASCADE", na.rm = TRUE),
    median_cascade_or = median(.data$odds_ratio, na.rm = TRUE),
    median_cascade_recall = median(.data$recall, na.rm = TRUE),
    median_pooled_de_or = median(.data$pooled_de_odds_ratio, na.rm = TRUE),
    median_pooled_de_recall = median(.data$pooled_de_recall, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(.data$cascade_top_n)


#######################################
# SAVE OUTPUTS
#######################################

readr::write_csv(
  cascade_vs_de_hitset_results,
  file = here::here("results", "tables", "cascade_vs_de_hitset_results.csv")
)

readr::write_csv(
  cascade_vs_de_hitset_summary,
  file = here::here("results", "tables", "cascade_vs_de_hitset_summary.csv")
)

readr::write_csv(
  cascade_vs_de_hitset_winner_table,
  file = here::here("results", "tables", "cascade_vs_de_hitset_winner_table.csv")
)

readr::write_csv(
  cascade_vs_de_hitset_combo_summary,
  file = here::here("results", "tables", "cascade_vs_de_hitset_combo_summary.csv")
)

readr::write_csv(
  cascade_vs_de_hitset_gt_summary,
  file = here::here("results", "tables", "cascade_vs_de_hitset_gt_summary.csv")
)

readr::write_csv(
  cascade_vs_de_hitset_topn_summary,
  file = here::here("results", "tables", "cascade_vs_de_hitset_topn_summary.csv")
)

readr::write_csv(
  de_hitset_summary,
  file = here::here("results", "tables", "de_hitset_summary.csv")
)

readr::write_csv(
  cascade_hitset_universe_completion_audit,
  file = here::here("results", "tables", "cascade_hitset_universe_completion_audit.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Saved CASCADE vs DE hit-set comparison.")
print(utils::head(de_hitset_summary, 20))
print(utils::head(cascade_vs_de_hitset_winner_table, 20))
print(cascade_vs_de_hitset_combo_summary)
print(cascade_vs_de_hitset_gt_summary)
print(cascade_vs_de_hitset_topn_summary)
