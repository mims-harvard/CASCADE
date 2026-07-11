################################################################################
# Script: 08_aggregation_functions.R
#
# Purpose:
#   Define reusable aggregation functions for converting cell-level CASCADE
#   explainability outputs into gene-level rankings.
#
# Notes:
#   All functions expect a long-format data frame containing at least:
#     - gene
#     - cell_key
#     - rank
#     - importance
#
#   They return:
#     - gene
#     - aggregated_score
#     - aggregated_rank
#     - n_cells_observed
################################################################################


#######################################
# LOAD PACKAGES
#######################################
select <- dplyr::select

suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
})


#######################################
# HELPER: RANK OUTPUT TABLE
#######################################

rank_aggregated_table <- function(df, score_col = "aggregated_score", decreasing = TRUE) {
  if (decreasing) {
    df %>%
      dplyr::arrange(dplyr::desc(.data[[score_col]]), .data$gene) %>%
      dplyr::mutate(aggregated_rank = dplyr::row_number())
  } else {
    df %>%
      dplyr::arrange(.data[[score_col]], .data$gene) %>%
      dplyr::mutate(aggregated_rank = dplyr::row_number())
  }
}

#######################################
# 1. COMBMNZ
#
# Definition:
#   Sum of min-max normalized importance across cells, multiplied by the
#   number of cells in which the gene appears.
#######################################

aggregate_combmnz <- function(df) {
  
  df_norm <- df %>%
    group_by(.data$cell_key) %>%
    mutate(
      importance_min = min(.data$importance, na.rm = TRUE),
      importance_max = max(.data$importance, na.rm = TRUE),
      importance_norm = dplyr::if_else(
        .data$importance_max > .data$importance_min,
        (.data$importance - .data$importance_min) /
          (.data$importance_max - .data$importance_min),
        0
      )
    ) %>%
    ungroup()
  
  out <- df_norm %>%
    group_by(.data$gene) %>%
    summarise(
      summed_norm = sum(.data$importance_norm, na.rm = TRUE),
      n_cells_observed = n_distinct(.data$cell_key),
      aggregated_score = .data$summed_norm[1] * .data$n_cells_observed[1],
      .groups = "drop"
    ) %>%
    dplyr::select(.data$gene, .data$aggregated_score, .data$n_cells_observed)
  
  rank_aggregated_table(out, score_col = "aggregated_score", decreasing = TRUE)
}


#######################################
# 2. BORDA COUNT
#
# Definition:
#   Sum of (N - rank + 1) across cells, where N is the max rank within a cell.
#######################################

aggregate_borda <- function(df) {
  
  df_borda <- df %>%
    group_by(.data$cell_key) %>%
    mutate(max_rank_in_cell = max(.data$rank, na.rm = TRUE),
           borda_points = .data$max_rank_in_cell - .data$rank + 1) %>%
    ungroup()
  
  out <- df_borda %>%
    group_by(.data$gene) %>%
    summarise(
      aggregated_score = sum(.data$borda_points, na.rm = TRUE),
      n_cells_observed = n_distinct(.data$cell_key),
      .groups = "drop"
    )
  
  rank_aggregated_table(out, score_col = "aggregated_score", decreasing = TRUE)
}


#######################################
# 3. MRR
#
# Definition:
#   Sum of reciprocal rank across cells.
#######################################

aggregate_mrr <- function(df) {
  
  out <- df %>%
    mutate(rr = 1 / .data$rank) %>%
    group_by(.data$gene) %>%
    summarise(
      aggregated_score = sum(.data$rr, na.rm = TRUE),
      n_cells_observed = n_distinct(.data$cell_key),
      .groups = "drop"
    )
  
  rank_aggregated_table(out, score_col = "aggregated_score", decreasing = TRUE)
}


#######################################
# 4. TOP-K VOTE
#
# Definition:
#   Fraction of cells where the gene appears in the top-k.
#######################################

aggregate_topk_vote <- function(df, k = 10) {
  
  n_total_cells <- n_distinct(df$cell_key)
  
  out <- df %>%
    filter(.data$rank <= k) %>%
    group_by(.data$gene) %>%
    summarise(
      n_cells_topk = n_distinct(.data$cell_key),
      aggregated_score = .data$n_cells_topk[1] / n_total_cells,
      n_cells_observed = n_distinct(.data$cell_key),
      .groups = "drop"
    ) %>%
    dplyr::select(.data$gene, .data$aggregated_score, .data$n_cells_observed)
  
  rank_aggregated_table(out, score_col = "aggregated_score", decreasing = TRUE)
}


#######################################
# 5. GEOMETRIC MEAN RANK
#
# Definition:
#   exp(mean(log(rank))) across cells.
#
# Note:
#   Lower is better, so ranking is ascending.
#######################################

aggregate_geom_mean_rank <- function(df) {
  
  out <- df %>%
    group_by(.data$gene) %>%
    summarise(
      aggregated_score = exp(mean(log(.data$rank), na.rm = TRUE)),
      n_cells_observed = n_distinct(.data$cell_key),
      .groups = "drop"
    )
  
  rank_aggregated_table(out, score_col = "aggregated_score", decreasing = FALSE)
}


#######################################
# 6. WEIGHTED BORDA
#
# Definition:
#   Borda points multiplied by cell-level importance.
#######################################

aggregate_weighted_borda <- function(df) {
  
  df_wb <- df %>%
    group_by(.data$cell_key) %>%
    mutate(
      max_rank_in_cell = max(.data$rank, na.rm = TRUE),
      borda_points = .data$max_rank_in_cell - .data$rank + 1,
      weighted_borda = .data$borda_points * .data$importance
    ) %>%
    ungroup()
  
  out <- df_wb %>%
    group_by(.data$gene) %>%
    summarise(
      aggregated_score = sum(.data$weighted_borda, na.rm = TRUE),
      n_cells_observed = n_distinct(.data$cell_key),
      .groups = "drop"
    )
  
  rank_aggregated_table(out, score_col = "aggregated_score", decreasing = TRUE)
}


#######################################
# WRAPPER
#######################################
run_aggregation_method <- function(df, method_name) {
  
  if (method_name == "combmnz") {
    out <- aggregate_combmnz(df)
  } else if (method_name == "borda") {
    out <- aggregate_borda(df)
  } else if (method_name == "mrr") {
    out <- aggregate_mrr(df)
  } else if (method_name == "topk_10") {
    out <- aggregate_topk_vote(df, k = 10)
  } else if (method_name == "topk_25") {
    out <- aggregate_topk_vote(df, k = 25)
  } else if (method_name == "topk_50") {
    out <- aggregate_topk_vote(df, k = 50)
  } else if (method_name == "topk_100") {
    out <- aggregate_topk_vote(df, k = 100)
  } else if (method_name == "topk_200") {
    out <- aggregate_topk_vote(df, k = 200)
  } else if (method_name == "geom_mean_rank") {
    out <- aggregate_geom_mean_rank(df)
  } else if (method_name == "weighted_borda") {
    out <- aggregate_weighted_borda(df)
  } else {
    stop("Unknown aggregation method: ", method_name)
  }
  
  out %>%
    dplyr::mutate(aggregation_method = method_name) %>%
    dplyr::select(.data$aggregation_method, dplyr::everything())
}