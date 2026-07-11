################################################################################
# Script: 17_run_combined_set_gt_comparison.R
#
# Purpose:
#   Build combined CASCADE gene sets (intersection / union / unique parts),
#   build analogous combined DE hit sets, compare both to ground truths, and
#   determine where combined CASCADE sets beat combined DE hit sets.
#
# Outputs:
#   - results/tables/combined_set_gt_results.csv
#   - results/tables/combined_set_gt_summary.csv
#   - results/tables/combined_set_gt_winner_table.csv
#   - results/tables/combined_set_gt_combo_summary.csv
#   - results/tables/combined_set_gt_gt_summary.csv
#   - results/tables/combined_set_gt_topn_summary.csv
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

CASCADE_TOP_N_VALUES <- c(25, 50, 100, 200, 500)
DE_P_ADJ_THRESH <- 0.05
DE_ABS_LOGFC_THRESH <- 0.5


#######################################
# HELPER: BUILD COMBINED SETS
#######################################

build_combined_sets <- function(df_left, df_right, top_n_val, comparison_type, group_label, left_label, right_label, set_value_col) {

  left_set <- df_left %>%
    dplyr::select(.data$gene, left_value = all_of(set_value_col))

  right_set <- df_right %>%
    dplyr::select(.data$gene, right_value = all_of(set_value_col))

  merged <- left_set %>%
    dplyr::full_join(right_set, by = "gene") %>%
    dplyr::mutate(
      left_value = dplyr::if_else(is.na(.data$left_value), FALSE, .data$left_value),
      right_value = dplyr::if_else(is.na(.data$right_value), FALSE, .data$right_value)
    )

  dplyr::bind_rows(
    merged %>%
      dplyr::transmute(
        comparison_type = comparison_type,
        comparison_group = group_label,
        left_label = left_label,
        right_label = right_label,
        combination_type = "intersection",
        cascade_top_n = top_n_val,
        gene = .data$gene,
        in_combined_set = .data$left_value & .data$right_value
      ),
    merged %>%
      dplyr::transmute(
        comparison_type = comparison_type,
        comparison_group = group_label,
        left_label = left_label,
        right_label = right_label,
        combination_type = "union",
        cascade_top_n = top_n_val,
        gene = .data$gene,
        in_combined_set = .data$left_value | .data$right_value
      ),
    merged %>%
      dplyr::transmute(
        comparison_type = comparison_type,
        comparison_group = group_label,
        left_label = left_label,
        right_label = right_label,
        combination_type = "left_only",
        cascade_top_n = top_n_val,
        gene = .data$gene,
        in_combined_set = .data$left_value & !.data$right_value
      ),
    merged %>%
      dplyr::transmute(
        comparison_type = comparison_type,
        comparison_group = group_label,
        left_label = left_label,
        right_label = right_label,
        combination_type = "right_only",
        cascade_top_n = top_n_val,
        gene = .data$gene,
        in_combined_set = !.data$left_value & .data$right_value
      )
  )
}


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


#######################################
# STEP 5: BUILD COMBINED CASCADE SETS
#######################################

combo_groups <- cascade_completed_all_combos %>%
  dplyr::group_by(.data$aggregation_method, .data$filter_mode) %>%
  dplyr::group_split()

combined_cascade_sets <- dplyr::bind_rows(
  lapply(combo_groups, function(df_combo) {

    meta <- df_combo %>%
      dplyr::distinct(.data$aggregation_method, .data$filter_mode)

    out_list <- list()

    for (top_n_val in CASCADE_TOP_N_VALUES) {

      df_combo_top <- df_combo %>%
        dplyr::mutate(in_set = .data$aggregated_rank <= top_n_val)

      for (task_i in c("treatment", "THR")) {
        df_left <- df_combo_top %>%
          dplyr::filter(.data$analysis_task == task_i, .data$perturbation_strategy == "cell_type")

        df_right <- df_combo_top %>%
          dplyr::filter(.data$analysis_task == task_i, .data$perturbation_strategy == "treatment")

        if (nrow(df_left) > 0 && nrow(df_right) > 0) {
          out_list[[length(out_list) + 1]] <- build_combined_sets(
            df_left = df_left,
            df_right = df_right,
            top_n_val = top_n_val,
            comparison_type = "within_task_across_strategies",
            group_label = task_i,
            left_label = "cell_type",
            right_label = "treatment",
            set_value_col = "in_set"
          ) %>%
            dplyr::mutate(
              source_method = "CASCADE",
              source_label = paste0("CASCADE_", meta$aggregation_method, "_", meta$filter_mode),
              aggregation_method = meta$aggregation_method,
              filter_mode = meta$filter_mode,
              baseline_type = NA_character_,
              scope = NA_character_
            )
        }
      }

      for (strategy_i in c("cell_type", "treatment")) {
        df_left <- df_combo_top %>%
          dplyr::filter(.data$analysis_task == "treatment", .data$perturbation_strategy == strategy_i)

        df_right <- df_combo_top %>%
          dplyr::filter(.data$analysis_task == "THR", .data$perturbation_strategy == strategy_i)

        if (nrow(df_left) > 0 && nrow(df_right) > 0) {
          out_list[[length(out_list) + 1]] <- build_combined_sets(
            df_left = df_left,
            df_right = df_right,
            top_n_val = top_n_val,
            comparison_type = "within_strategy_across_tasks",
            group_label = strategy_i,
            left_label = "treatment",
            right_label = "THR",
            set_value_col = "in_set"
          ) %>%
            dplyr::mutate(
              source_method = "CASCADE",
              source_label = paste0("CASCADE_", meta$aggregation_method, "_", meta$filter_mode),
              aggregation_method = meta$aggregation_method,
              filter_mode = meta$filter_mode,
              baseline_type = NA_character_,
              scope = NA_character_
            )
        }
      }
    }

    dplyr::bind_rows(out_list)
  })
)


#######################################
# STEP 6: BUILD COMBINED DE SETS
#######################################

combined_de_sets <- dplyr::bind_rows(
  lapply(
    de_hit_sets %>%
      dplyr::group_by(.data$baseline_type) %>%
      dplyr::group_split(),
    function(df_de) {

      out_list <- list()

      # pooled DE only has one set per task x strategy, scoped DE has multiple scopes
      if (unique(df_de$baseline_type)[1] == "pooled") {

        for (task_i in c("treatment", "THR")) {
          df_left <- df_de %>%
            dplyr::filter(.data$analysis_task == task_i, .data$perturbation_strategy == "cell_type")

          df_right <- df_de %>%
            dplyr::filter(.data$analysis_task == task_i, .data$perturbation_strategy == "treatment")

          if (nrow(df_left) > 0 && nrow(df_right) > 0) {
            out_list[[length(out_list) + 1]] <- build_combined_sets(
              df_left = df_left,
              df_right = df_right,
              top_n_val = NA_real_,
              comparison_type = "within_task_across_strategies",
              group_label = task_i,
              left_label = "cell_type",
              right_label = "treatment",
              set_value_col = "is_de_hit"
            ) %>%
              dplyr::mutate(
                source_method = "DE",
                source_label = paste0("DE_pooled_combined_", task_i),
                aggregation_method = NA_character_,
                filter_mode = NA_character_,
                baseline_type = "pooled",
                scope = NA_character_
              )
          }
        }

        for (strategy_i in c("cell_type", "treatment")) {
          df_left <- df_de %>%
            dplyr::filter(.data$analysis_task == "treatment", .data$perturbation_strategy == strategy_i)

          df_right <- df_de %>%
            dplyr::filter(.data$analysis_task == "THR", .data$perturbation_strategy == strategy_i)

          if (nrow(df_left) > 0 && nrow(df_right) > 0) {
            out_list[[length(out_list) + 1]] <- build_combined_sets(
              df_left = df_left,
              df_right = df_right,
              top_n_val = NA_real_,
              comparison_type = "within_strategy_across_tasks",
              group_label = strategy_i,
              left_label = "treatment",
              right_label = "THR",
              set_value_col = "is_de_hit"
            ) %>%
              dplyr::mutate(
                source_method = "DE",
                source_label = paste0("DE_pooled_combined_", strategy_i),
                aggregation_method = NA_character_,
                filter_mode = NA_character_,
                baseline_type = "pooled",
                scope = NA_character_
              )
          }
        }
      }
      dplyr::bind_rows(out_list)
    }
  )
)


#######################################
# STEP 7: COMPARE ALL COMBINED SETS TO GTs
#######################################

all_combined_sets <- dplyr::bind_rows(
  combined_cascade_sets,
  combined_de_sets
)

combined_set_gt_results <- dplyr::bind_rows(
  lapply(
    all_combined_sets %>%
      dplyr::group_by(
        .data$source_method,
        .data$source_label,
        .data$aggregation_method,
        .data$filter_mode,
        .data$baseline_type,
        .data$scope,
        .data$comparison_type,
        .data$comparison_group,
        .data$left_label,
        .data$right_label,
        .data$combination_type,
        .data$cascade_top_n
      ) %>%
      dplyr::group_split(),
    function(df_set) {

      meta <- df_set %>%
        dplyr::distinct(
          .data$source_method,
          .data$source_label,
          .data$aggregation_method,
          .data$filter_mode,
          .data$baseline_type,
          .data$scope,
          .data$comparison_type,
          .data$comparison_group,
          .data$left_label,
          .data$right_label,
          .data$combination_type,
          .data$cascade_top_n
        )

      dplyr::bind_rows(
        lapply(unique(all_gt_binary$gt_name), function(gt_nm) {

          gt_sub <- all_gt_binary %>%
            dplyr::filter(.data$gt_name == gt_nm)

          merged <- df_set %>%
            dplyr::left_join(gt_sub, by = "gene")

          safe_fisher_set_vs_gt(
            set_flag = merged$in_combined_set,
            gt_flag = merged$binary_label_main
          ) %>%
            dplyr::mutate(
              source_method = meta$source_method,
              source_label = meta$source_label,
              aggregation_method = meta$aggregation_method,
              filter_mode = meta$filter_mode,
              baseline_type = meta$baseline_type,
              scope = meta$scope,
              comparison_type = meta$comparison_type,
              comparison_group = meta$comparison_group,
              left_label = meta$left_label,
              right_label = meta$right_label,
              combination_type = meta$combination_type,
              cascade_top_n = meta$cascade_top_n,
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
# STEP 8: WINNER TABLE
#######################################

cascade_rows <- combined_set_gt_results %>%
  dplyr::filter(.data$source_method == "CASCADE")

pooled_de_rows <- combined_set_gt_results %>%
  dplyr::filter(.data$source_method == "DE", .data$baseline_type == "pooled") %>%
  dplyr::select(
    .data$comparison_type,
    .data$comparison_group,
    .data$combination_type,
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

combined_set_gt_winner_table <- cascade_rows %>%
  dplyr::left_join(
    pooled_de_rows,
    by = c(
      "comparison_type",
      "comparison_group",
      "combination_type",
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
    )
  )


#######################################
# STEP 9: SUMMARIES
#######################################

combined_set_gt_summary <- combined_set_gt_results

combined_set_gt_combo_summary <- combined_set_gt_winner_table %>%
  dplyr::group_by(
    .data$aggregation_method,
    .data$filter_mode,
    .data$comparison_type,
    .data$combination_type,
    .data$cascade_top_n
  ) %>%
  dplyr::summarise(
    n_rows = dplyr::n(),
    n_or_wins_vs_pooled = sum(.data$winner_vs_pooled_de_or == "CASCADE", na.rm = TRUE),
    n_recall_wins_vs_pooled = sum(.data$winner_vs_pooled_de_recall == "CASCADE", na.rm = TRUE),
    prop_or_wins_vs_pooled = mean(.data$winner_vs_pooled_de_or == "CASCADE", na.rm = TRUE),
    prop_recall_wins_vs_pooled = mean(.data$winner_vs_pooled_de_recall == "CASCADE", na.rm = TRUE),
    median_cascade_or = median(.data$odds_ratio, na.rm = TRUE),
    median_cascade_recall = median(.data$recall, na.rm = TRUE),
    median_pooled_de_or = median(.data$pooled_de_odds_ratio, na.rm = TRUE),
    median_pooled_de_recall = median(.data$pooled_de_recall, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(
    dplyr::desc(.data$prop_or_wins_vs_pooled),
    dplyr::desc(.data$prop_recall_wins_vs_pooled)
  )

combined_set_gt_gt_summary <- combined_set_gt_winner_table %>%
  dplyr::group_by(
    .data$result_family,
    .data$gt_name,
    .data$gt_type,
    .data$comparison_type,
    .data$combination_type
  ) %>%
  dplyr::summarise(
    n_rows = dplyr::n(),
    n_or_wins_vs_pooled = sum(.data$winner_vs_pooled_de_or == "CASCADE", na.rm = TRUE),
    n_recall_wins_vs_pooled = sum(.data$winner_vs_pooled_de_recall == "CASCADE", na.rm = TRUE),
    prop_or_wins_vs_pooled = mean(.data$winner_vs_pooled_de_or == "CASCADE", na.rm = TRUE),
    prop_recall_wins_vs_pooled = mean(.data$winner_vs_pooled_de_recall == "CASCADE", na.rm = TRUE),
    median_cascade_or = median(.data$odds_ratio, na.rm = TRUE),
    median_cascade_recall = median(.data$recall, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(.data$gt_name, .data$comparison_type, .data$combination_type)

combined_set_gt_topn_summary <- combined_set_gt_winner_table %>%
  dplyr::group_by(
    .data$comparison_type,
    .data$combination_type,
    .data$cascade_top_n
  ) %>%
  dplyr::summarise(
    n_rows = dplyr::n(),
    n_or_wins_vs_pooled = sum(.data$winner_vs_pooled_de_or == "CASCADE", na.rm = TRUE),
    n_recall_wins_vs_pooled = sum(.data$winner_vs_pooled_de_recall == "CASCADE", na.rm = TRUE),
    prop_or_wins_vs_pooled = mean(.data$winner_vs_pooled_de_or == "CASCADE", na.rm = TRUE),
    prop_recall_wins_vs_pooled = mean(.data$winner_vs_pooled_de_recall == "CASCADE", na.rm = TRUE),
    median_cascade_or = median(.data$odds_ratio, na.rm = TRUE),
    median_cascade_recall = median(.data$recall, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(.data$comparison_type, .data$combination_type, .data$cascade_top_n)


#######################################
# SAVE OUTPUTS
#######################################

readr::write_csv(
  combined_set_gt_results,
  file = here::here("results", "tables", "combined_set_gt_results.csv")
)

readr::write_csv(
  combined_set_gt_summary,
  file = here::here("results", "tables", "combined_set_gt_summary.csv")
)

readr::write_csv(
  combined_set_gt_winner_table,
  file = here::here("results", "tables", "combined_set_gt_winner_table.csv")
)

readr::write_csv(
  combined_set_gt_combo_summary,
  file = here::here("results", "tables", "combined_set_gt_combo_summary.csv")
)

readr::write_csv(
  combined_set_gt_gt_summary,
  file = here::here("results", "tables", "combined_set_gt_gt_summary.csv")
)

readr::write_csv(
  combined_set_gt_topn_summary,
  file = here::here("results", "tables", "combined_set_gt_topn_summary.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Saved combined set vs GT comparison with DE baseline.")
print(utils::head(combined_set_gt_combo_summary, 20))
print(utils::head(combined_set_gt_gt_summary, 20))
print(combined_set_gt_topn_summary)
