################################################################################
# Script: gt_benchmark_helpers.R
#
# Purpose:
#   Shared ground-truth-benchmarking helpers used by 09, 11, 16, 16a, and 17.
#   These were previously copy-pasted near-identically across those five
#   scripts; consolidated here as the single source of truth.
#
# Notes:
#   Source this after 00_config.R (some helpers reference ANALYSIS_PARAMS
#   indirectly via their callers, not directly).
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
})


#######################################
# BUILD POOLED THYROID GT BINARY SETS
#######################################

build_thyroid_gt_binary_sets <- function(gt_thyroid_df) {

  pooled_main <- gt_thyroid_df %>%
    dplyr::group_by(.data$gene) %>%
    dplyr::summarise(
      binary_label_main = any(.data$sig_main, na.rm = TRUE),
      binary_label_sensitivity = any(.data$sig_sensitivity, na.rm = TRUE),
      abs_score_max = max(.data$abs_score, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      gt_name = "thyroid_response_pooled_main",
      gt_type = "binary"
    )

  pooled_top_abs <- gt_thyroid_df %>%
    dplyr::group_by(.data$gene) %>%
    dplyr::summarise(
      abs_score_max = max(.data$abs_score, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(.data$abs_score_max)) %>%
    dplyr::mutate(
      binary_label_main = dplyr::row_number() <= 200,
      binary_label_sensitivity = dplyr::row_number() <= 500,
      gt_name = "thyroid_response_pooled_top_abs_score",
      gt_type = "binary"
    )

  dplyr::bind_rows(pooled_main, pooled_top_abs) %>%
    dplyr::select(
      .data$gene,
      .data$gt_name,
      .data$gt_type,
      .data$binary_label_main,
      .data$binary_label_sensitivity,
      .data$abs_score_max
    )
}


#######################################
# METRIC HELPERS
#######################################

safe_auc <- function(labels, scores) {
  valid <- !is.na(labels) & !is.na(scores)
  labels <- labels[valid]
  scores <- scores[valid]

  if (length(labels) == 0 || length(unique(labels)) < 2) {
    return(NA_real_)
  }

  as.numeric(
    pROC::roc(response = labels, predictor = scores, quiet = TRUE)$auc
  )
}

safe_spearman <- function(x, y) {
  valid <- !is.na(x) & !is.na(y)
  x <- x[valid]
  y <- y[valid]

  if (length(x) < 3 || length(unique(x)) < 2 || length(unique(y)) < 2) {
    return(NA_real_)
  }

  suppressWarnings(as.numeric(stats::cor(x, y, method = "spearman")))
}

safe_fisher_from_topn <- function(labels, ranks, top_n) {
  valid <- !is.na(labels) & !is.na(ranks)
  labels <- labels[valid]
  ranks <- ranks[valid]

  if (length(labels) == 0) {
    return(tibble::tibble(
      top_n = top_n, odds_ratio = NA_real_, fisher_p = NA_real_,
      overlap = NA_real_, n_top = NA_real_, n_positive = NA_real_
    ))
  }

  top_flag <- ranks <= top_n
  a <- sum(top_flag & labels)
  b <- sum(top_flag & !labels)
  c <- sum(!top_flag & labels)
  d <- sum(!top_flag & !labels)

  ft <- tryCatch(stats::fisher.test(matrix(c(a, b, c, d), nrow = 2)), error = function(e) NULL)

  tibble::tibble(
    top_n = top_n,
    odds_ratio = if (!is.null(ft)) unname(ft$estimate) else NA_real_,
    fisher_p = if (!is.null(ft)) ft$p.value else NA_real_,
    overlap = a, n_top = sum(top_flag), n_positive = sum(labels)
  )
}

# Set-vs-GT comparison (adds precision/recall/Jaccard beyond safe_fisher_from_topn,
# used by the hit-set/combined-set scripts which compare fixed gene sets rather
# than full rankings).
safe_fisher_set_vs_gt <- function(set_flag, gt_flag) {
  valid <- !is.na(set_flag) & !is.na(gt_flag)
  set_flag <- set_flag[valid]
  gt_flag <- gt_flag[valid]

  if (length(set_flag) == 0) {
    return(tibble::tibble(
      odds_ratio = NA_real_, fisher_p = NA_real_, overlap = NA_real_,
      set_size = NA_real_, gt_size = NA_real_,
      precision = NA_real_, recall = NA_real_, jaccard = NA_real_
    ))
  }

  a <- sum(set_flag & gt_flag)
  b <- sum(set_flag & !gt_flag)
  c <- sum(!set_flag & gt_flag)
  d <- sum(!set_flag & !gt_flag)

  ft <- tryCatch(stats::fisher.test(matrix(c(a, b, c, d), nrow = 2)), error = function(e) NULL)

  precision <- if ((a + b) > 0) a / (a + b) else NA_real_
  recall <- if ((a + c) > 0) a / (a + c) else NA_real_
  jaccard <- if ((a + b + c) > 0) a / (a + b + c) else NA_real_

  tibble::tibble(
    odds_ratio = if (!is.null(ft)) unname(ft$estimate) else NA_real_,
    fisher_p = if (!is.null(ft)) ft$p.value else NA_real_,
    overlap = a, set_size = a + b, gt_size = a + c,
    precision = precision, recall = recall, jaccard = jaccard
  )
}


#######################################
# COMPLETE A CASCADE COMBO TO THE FULL GENE UNIVERSE
#
# Zero-fills genes present in the task/strategy universe but absent from a
# given aggregation×filter combo's ranking (e.g. below a topk_* cutoff), then
# re-ranks, so CASCADE and DE are always compared over an identical gene
# background.
#######################################

complete_cascade_combo_to_universe <- function(df_part, universe_df) {

  meta <- df_part %>%
    dplyr::distinct(.data$aggregation_method, .data$filter_mode,
                     .data$analysis_task, .data$perturbation_strategy)

  full_universe <- universe_df %>%
    dplyr::filter(.data$analysis_task == meta$analysis_task,
                  .data$perturbation_strategy == meta$perturbation_strategy) %>%
    dplyr::distinct(.data$gene)

  completed <- full_universe %>%
    dplyr::left_join(df_part %>% dplyr::select(.data$gene, .data$aggregated_score), by = "gene") %>%
    dplyr::mutate(aggregated_score = dplyr::if_else(is.na(.data$aggregated_score), 0, .data$aggregated_score)) %>%
    dplyr::arrange(dplyr::desc(.data$aggregated_score), .data$gene) %>%
    dplyr::mutate(
      aggregated_rank = dplyr::row_number(),
      aggregation_method = meta$aggregation_method,
      filter_mode = meta$filter_mode,
      analysis_task = meta$analysis_task,
      perturbation_strategy = meta$perturbation_strategy
    ) %>%
    dplyr::select(.data$aggregation_method, .data$filter_mode, .data$analysis_task,
                  .data$perturbation_strategy, .data$gene, .data$aggregated_score, .data$aggregated_rank)

  audit <- tibble::tibble(
    aggregation_method = meta$aggregation_method, filter_mode = meta$filter_mode,
    analysis_task = meta$analysis_task, perturbation_strategy = meta$perturbation_strategy,
    n_genes_in_universe = nrow(full_universe),
    n_genes_original = dplyr::n_distinct(df_part$gene),
    n_genes_completed = dplyr::n_distinct(completed$gene),
    n_missing_added = nrow(full_universe) - dplyr::n_distinct(df_part$gene)
  )

  list(data = completed, audit = audit)
}


#######################################
# RANK A DE BASELINE TABLE BY de_score
#######################################

rank_de_table <- function(df) {
  df %>%
    dplyr::mutate(comparison_score = .data$de_score) %>%
    dplyr::arrange(dplyr::desc(.data$comparison_score), .data$gene) %>%
    dplyr::mutate(aggregated_score = .data$comparison_score, aggregated_rank = dplyr::row_number()) %>%
    dplyr::select(.data$gene, .data$aggregated_score, .data$aggregated_rank)
}
