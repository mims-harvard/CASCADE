################################################################################
# Script: 16a_run_cascade_vs_de_ranking_comparison.R
#
# Purpose:
#   Compare CASCADE rankings against DE rankings using the same mapped gene
#   universe, across ALL CASCADE aggregation × filter combinations.
#
# Main question:
#   Does CASCADE rank biologically relevant genes better than a DE statistic?
#
# Outputs:
#   - results/tables/cascade_vs_de_ranking_results.csv
#   - results/tables/cascade_vs_de_ranking_summary.csv
#   - results/tables/cascade_vs_de_ranking_winner_table.csv
#   - results/tables/cascade_vs_de_ranking_combo_summary.csv
#   - results/tables/cascade_vs_de_ranking_gt_summary.csv
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
# STEP 1: BUILD GT OBJECTS
#######################################

thyroid_gt_binary <- build_thyroid_gt_binary_sets(gt_thyroid_response_standardized)

all_gt_binary <- dplyr::bind_rows(
  thyroid_gt_binary %>% dplyr::mutate(result_family = "thyroid_binary"),
  gt_mmc2_long %>%
    dplyr::select(.data$gene, .data$gt_name, .data$gt_type, .data$binary_label_main, .data$abs_score) %>%
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

cascade_completed_all_combos <- dplyr::bind_rows(
  lapply(
    aggregation_results_all_methods %>%
      dplyr::group_by(
        .data$aggregation_method,
        .data$filter_mode,
        .data$analysis_task,
        .data$perturbation_strategy
      ) %>%
      dplyr::group_split(),
    function(df_part) complete_cascade_combo_to_universe(df_part, cascade_task_strategy_universe)$data
  )
)


#######################################
# STEP 4: PREPARE DE RANKINGS
#######################################

de_ranked_objects <- dplyr::bind_rows(
  lapply(
    de_baselines_restricted_to_cascade_universe %>%
      dplyr::mutate(perturbation_strategy = .data$comparison_universe) %>%
      dplyr::group_by(
        .data$analysis_task,
        .data$perturbation_strategy,
        .data$baseline_type,
        .data$scope
      ) %>%
      dplyr::group_split(),
    function(df_part) {

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

      ranked <- rank_de_table(df_part)

      full_universe %>%
        dplyr::left_join(ranked, by = "gene") %>%
        dplyr::mutate(
          aggregated_score = dplyr::if_else(is.na(.data$aggregated_score), 0, .data$aggregated_score)
        ) %>%
        dplyr::arrange(dplyr::desc(.data$aggregated_score), .data$gene) %>%
        dplyr::mutate(
          aggregated_rank = dplyr::row_number(),
          source_label = dplyr::case_when(
            meta$baseline_type == "pooled" ~ paste0("DE_pooled_", meta$analysis_task),
            TRUE ~ paste0("DE_scoped_", meta$scope, "_", meta$analysis_task)
          ),
          analysis_task = meta$analysis_task,
          perturbation_strategy = meta$perturbation_strategy,
          baseline_type = meta$baseline_type,
          scope = meta$scope
        ) %>%
        dplyr::select(
          .data$source_label,
          .data$analysis_task,
          .data$perturbation_strategy,
          .data$baseline_type,
          .data$scope,
          .data$gene,
          .data$aggregated_score,
          .data$aggregated_rank
        )
    }
  )
)


#######################################
# STEP 5: CASCADE RANKING RESULTS
#######################################

cascade_ranking_results <- dplyr::bind_rows(
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
        lapply(unique(all_gt_binary$gt_name), function(gt_nm) {

          gt_sub <- all_gt_binary %>%
            dplyr::filter(.data$gt_name == gt_nm)

          merged <- df_part %>%
            dplyr::left_join(gt_sub, by = "gene")

          auc_value <- safe_auc(merged$binary_label_main, merged$aggregated_score)

          spearman_value <- if (unique(gt_sub$gt_type)[1] == "continuous") {
            safe_spearman(merged$aggregated_score, merged$abs_score)
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
              source_method = "CASCADE",
              source_label = paste0("CASCADE_", meta$aggregation_method, "_", meta$filter_mode),
              aggregation_method = meta$aggregation_method,
              filter_mode = meta$filter_mode,
              analysis_task = meta$analysis_task,
              perturbation_strategy = meta$perturbation_strategy,
              baseline_type = NA_character_,
              scope = NA_character_,
              result_family = unique(gt_sub$result_family)[1],
              gt_name = gt_nm,
              gt_type = unique(gt_sub$gt_type)[1],
              auc = auc_value,
              spearman_rho = spearman_value
            )
        })
      )
    }
  )
)


#######################################
# STEP 6: DE RANKING RESULTS
#######################################

de_ranking_results <- dplyr::bind_rows(
  lapply(
    de_ranked_objects %>%
      dplyr::group_by(
        .data$source_label,
        .data$analysis_task,
        .data$perturbation_strategy,
        .data$baseline_type,
        .data$scope
      ) %>%
      dplyr::group_split(),
    function(df_part) {

      meta <- df_part %>%
        dplyr::distinct(
          .data$source_label,
          .data$analysis_task,
          .data$perturbation_strategy,
          .data$baseline_type,
          .data$scope
        )

      dplyr::bind_rows(
        lapply(unique(all_gt_binary$gt_name), function(gt_nm) {

          gt_sub <- all_gt_binary %>%
            dplyr::filter(.data$gt_name == gt_nm)

          merged <- df_part %>%
            dplyr::left_join(gt_sub, by = "gene")

          auc_value <- safe_auc(merged$binary_label_main, merged$aggregated_score)

          spearman_value <- if (unique(gt_sub$gt_type)[1] == "continuous") {
            safe_spearman(merged$aggregated_score, merged$abs_score)
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
              source_method = "DE",
              source_label = meta$source_label,
              aggregation_method = NA_character_,
              filter_mode = NA_character_,
              analysis_task = meta$analysis_task,
              perturbation_strategy = meta$perturbation_strategy,
              baseline_type = meta$baseline_type,
              scope = meta$scope,
              result_family = unique(gt_sub$result_family)[1],
              gt_name = gt_nm,
              gt_type = unique(gt_sub$gt_type)[1],
              auc = auc_value,
              spearman_rho = spearman_value
            )
        })
      )
    }
  )
)


#######################################
# STEP 7: COMBINE + SUMMARISE
#######################################

cascade_vs_de_ranking_results <- dplyr::bind_rows(
  cascade_ranking_results,
  de_ranking_results
)

cascade_vs_de_ranking_summary <- cascade_vs_de_ranking_results %>%
  dplyr::group_by(
    .data$source_method,
    .data$source_label,
    .data$aggregation_method,
    .data$filter_mode,
    .data$analysis_task,
    .data$perturbation_strategy,
    .data$baseline_type,
    .data$scope,
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
# STEP 8: WINNER TABLES
#######################################

cascade_rows <- cascade_vs_de_ranking_summary %>%
  dplyr::filter(.data$source_method == "CASCADE")

pooled_de_rows <- cascade_vs_de_ranking_summary %>%
  dplyr::filter(.data$source_method == "DE", .data$baseline_type == "pooled") %>%
  dplyr::select(
    .data$analysis_task,
    .data$perturbation_strategy,
    .data$result_family,
    .data$gt_name,
    .data$gt_type,
    pooled_de_auc = .data$auc,
    pooled_de_or = .data$max_odds_ratio,
    pooled_de_spearman = .data$spearman_rho
  )

cascade_vs_de_ranking_winner_table <- cascade_rows %>%
  dplyr::left_join(
    pooled_de_rows,
    by = c("analysis_task", "perturbation_strategy", "result_family", "gt_name", "gt_type")
  ) %>%
  dplyr::mutate(
    winner_auc = dplyr::case_when(
      is.na(.data$pooled_de_auc) ~ "CASCADE",
      .data$auc > .data$pooled_de_auc ~ "CASCADE",
      .data$auc < .data$pooled_de_auc ~ "pooled_DE",
      TRUE ~ "tie"
    ),
    winner_or = dplyr::case_when(
      is.na(.data$pooled_de_or) ~ "CASCADE",
      .data$max_odds_ratio > .data$pooled_de_or ~ "CASCADE",
      .data$max_odds_ratio < .data$pooled_de_or ~ "pooled_DE",
      TRUE ~ "tie"
    )
  )

cascade_vs_de_ranking_combo_summary <- cascade_vs_de_ranking_winner_table %>%
  dplyr::group_by(.data$aggregation_method, .data$filter_mode) %>%
  dplyr::summarise(
    n_rows = dplyr::n(),
    n_auc_wins = sum(.data$winner_auc == "CASCADE", na.rm = TRUE),
    n_or_wins = sum(.data$winner_or == "CASCADE", na.rm = TRUE),
    prop_auc_wins = mean(.data$winner_auc == "CASCADE", na.rm = TRUE),
    prop_or_wins = mean(.data$winner_or == "CASCADE", na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(dplyr::desc(.data$prop_auc_wins), dplyr::desc(.data$prop_or_wins))

cascade_vs_de_ranking_gt_summary <- cascade_vs_de_ranking_winner_table %>%
  dplyr::group_by(.data$result_family, .data$gt_name, .data$gt_type) %>%
  dplyr::summarise(
    n_rows = dplyr::n(),
    n_auc_wins = sum(.data$winner_auc == "CASCADE", na.rm = TRUE),
    n_or_wins = sum(.data$winner_or == "CASCADE", na.rm = TRUE),
    prop_auc_wins = mean(.data$winner_auc == "CASCADE", na.rm = TRUE),
    prop_or_wins = mean(.data$winner_or == "CASCADE", na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(dplyr::desc(.data$prop_auc_wins), dplyr::desc(.data$prop_or_wins))


#######################################
# SAVE OUTPUTS
#######################################

readr::write_csv(
  cascade_vs_de_ranking_results,
  file = here::here("results", "tables", "cascade_vs_de_ranking_results.csv")
)

readr::write_csv(
  cascade_vs_de_ranking_summary,
  file = here::here("results", "tables", "cascade_vs_de_ranking_summary.csv")
)

readr::write_csv(
  cascade_vs_de_ranking_winner_table,
  file = here::here("results", "tables", "cascade_vs_de_ranking_winner_table.csv")
)

readr::write_csv(
  cascade_vs_de_ranking_combo_summary,
  file = here::here("results", "tables", "cascade_vs_de_ranking_combo_summary.csv")
)

readr::write_csv(
  cascade_vs_de_ranking_gt_summary,
  file = here::here("results", "tables", "cascade_vs_de_ranking_gt_summary.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Saved CASCADE vs DE ranking comparison.")
print(utils::head(cascade_vs_de_ranking_combo_summary, 20))
print(utils::head(cascade_vs_de_ranking_gt_summary, 20))
