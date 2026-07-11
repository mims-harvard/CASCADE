################################################################################
# Script: 19_run_cell_type_specific_analysis_matched_universe.R
#
# Purpose:
#   Cell-type-specific CASCADE-vs-DE comparison for astrocytes and
#   glutamatergic neurons (treatment task, treatment-marker perturbation only;
#   Results "CASCADE-Explainer recovers cell-type-specific thyroid hormone
#   programmes in astrocytes and glutamatergic neurons").
#
#   Universe definition (per perturbation_strategy x cell type):
#     - CASCADE universe = union of top-100 genes across all cells of that type
#                        = borda gene pool (all methods use all top-100 genes)
#     - DE restricted to the matching CASCADE universe:
#         perturbation_strategy == "cell_type"  -> comparison_universe == "cell_type"
#         perturbation_strategy == "treatment"  -> comparison_universe == "treatment"
#     - topk_* CASCADE methods zero-filled to this universe (score=0 for genes
#       ranked 11-100 that are in the universe but below the k threshold)
#
#   The analysis is run separately for each perturbation_strategy so that
#   CASCADE and DE are always evaluated over the identical gene background.
#
# Outputs:
#   - results/tables/mu_cell_type_specific_cascade_results.csv
#   - results/tables/mu_cell_type_specific_cascade_summary.csv
#   - results/tables/mu_cell_type_specific_cascade_vs_de_winner_table.csv
#   - results/tables/mu_cell_type_specific_combo_summary.csv
#   - results/tables/mu_cell_type_specific_gt_summary.csv
#   - results/tables/mu_cascade_celltype_universe_completion_audit.csv
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
  library(pROC)
  library(stringr)
})


#######################################
# LOAD INPUT OBJECTS
#######################################

cascade_outputs_standardized <- readRDS(
  here::here("results", "rds", "cascade_outputs_standardized.rds")
)

de_baselines_restricted_to_cascade_universe <- readRDS(
  here::here("results", "rds", "de_baselines_restricted_to_cascade_universe.rds")
)

gt_thyroid_response_standardized <- readRDS(
  here::here("results", "rds", "gt_thyroid_response_standardized.rds")
)


#######################################
# PARAMETERS
#######################################

CELLTYPE_MIN_GENES    <- 200
CELLTYPE_MIN_GT_POS   <- 10
TOP_N_VALUES          <- c(25, 50, 100, 200, 500)
AGGREGATION_METHODS   <- c("combmnz", "borda", "mrr", "topk_10", "topk_25", "topk_50", "topk_100")
PERTURBATION_STRATEGIES <- c("cell_type", "treatment")


#######################################
# HELPERS
#######################################

safe_auc <- function(labels, scores) {
  valid  <- !is.na(labels) & !is.na(scores)
  labels <- labels[valid]
  scores <- scores[valid]

  if (length(labels) == 0 || length(unique(labels)) < 2) {
    return(NA_real_)
  }

  as.numeric(
    pROC::roc(response = labels, predictor = scores, quiet = TRUE)$auc
  )
}

safe_fisher_from_topn <- function(labels, ranks, top_n) {
  valid  <- !is.na(labels) & !is.na(ranks)
  labels <- labels[valid]
  ranks  <- ranks[valid]

  if (length(labels) == 0) {
    return(tibble::tibble(
      top_n            = top_n,
      odds_ratio       = NA_real_,
      fisher_p         = NA_real_,
      overlap          = NA_real_,
      n_top            = NA_real_,
      n_positive       = NA_real_,
      n_valid_universe = 0
    ))
  }

  top_flag <- ranks <= top_n

  a <- sum(top_flag & labels)
  b <- sum(top_flag & !labels)
  c <- sum(!top_flag & labels)
  d <- sum(!top_flag & !labels)

  ft <- tryCatch(
    stats::fisher.test(matrix(c(a, b, c, d), nrow = 2)),
    error = function(e) NULL
  )

  tibble::tibble(
    top_n            = top_n,
    odds_ratio       = if (!is.null(ft)) unname(ft$estimate) else NA_real_,
    fisher_p         = if (!is.null(ft)) ft$p.value else NA_real_,
    overlap          = a,
    n_top            = sum(top_flag),
    n_positive       = sum(labels),
    n_valid_universe = length(labels)
  )
}

aggregate_within_cell_type <- function(df_celltype, method_name) {

  aggregate_topk_vote <- function(df, k) {
    n_total_cells <- dplyr::n_distinct(df$cell_key)

    df %>%
      dplyr::filter(.data$rank <= k) %>%
      dplyr::group_by(.data$gene) %>%
      dplyr::summarise(
        n_cells_topk    = dplyr::n_distinct(.data$cell_key),
        aggregated_score = .data$n_cells_topk[1] / n_total_cells,
        .groups = "drop"
      ) %>%
      dplyr::arrange(dplyr::desc(.data$aggregated_score), .data$gene) %>%
      dplyr::mutate(aggregated_rank = dplyr::row_number())
  }

  aggregate_borda <- function(df) {
    df %>%
      dplyr::mutate(borda_score = max(.data$rank, na.rm = TRUE) - .data$rank + 1) %>%
      dplyr::group_by(.data$gene) %>%
      dplyr::summarise(
        aggregated_score = sum(.data$borda_score, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      dplyr::arrange(dplyr::desc(.data$aggregated_score), .data$gene) %>%
      dplyr::mutate(aggregated_rank = dplyr::row_number())
  }

  aggregate_mrr <- function(df) {
    df %>%
      dplyr::mutate(rr = 1 / .data$rank) %>%
      dplyr::group_by(.data$gene) %>%
      dplyr::summarise(
        aggregated_score = sum(.data$rr, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      dplyr::arrange(dplyr::desc(.data$aggregated_score), .data$gene) %>%
      dplyr::mutate(aggregated_rank = dplyr::row_number())
  }

  aggregate_combmnz <- function(df) {
    df %>%
      dplyr::group_by(.data$gene) %>%
      dplyr::summarise(
        aggregated_score = sum(.data$importance, na.rm = TRUE) *
          dplyr::n_distinct(.data$cell_key),
        .groups = "drop"
      ) %>%
      dplyr::arrange(dplyr::desc(.data$aggregated_score), .data$gene) %>%
      dplyr::mutate(aggregated_rank = dplyr::row_number())
  }

  out <- switch(
    method_name,
    "borda"    = aggregate_borda(df_celltype),
    "mrr"      = aggregate_mrr(df_celltype),
    "combmnz"  = aggregate_combmnz(df_celltype),
    "topk_10"  = aggregate_topk_vote(df_celltype, 10),
    "topk_25"  = aggregate_topk_vote(df_celltype, 25),
    "topk_50"  = aggregate_topk_vote(df_celltype, 50),
    "topk_100" = aggregate_topk_vote(df_celltype, 100),
    stop("Unknown method: ", method_name)
  )

  out %>%
    dplyr::mutate(aggregation_method = method_name)
}

rank_de_table <- function(df) {
  df %>%
    dplyr::group_by(.data$gene) %>%
    dplyr::summarise(
      aggregated_score  = max(.data$de_score, na.rm = TRUE),
      best_abs_log2fc   = max(abs(.data$log2fc), na.rm = TRUE),
      min_p_adj         = min(.data$p_adj, na.rm = TRUE),
      n_rows_collapsed  = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(.data$aggregated_score), .data$gene) %>%
    dplyr::mutate(aggregated_rank = dplyr::row_number())
}

# Expands a CASCADE ranked list to the full cell-type CASCADE universe,
# filling missing genes (below topk threshold) with score = 0, then re-ranks.
complete_cascade_celltype_to_universe <- function(df_part, universe_genes, ct_norm) {

  completed <- universe_genes %>%
    dplyr::left_join(
      df_part %>% dplyr::select(.data$gene, .data$aggregated_score),
      by = "gene"
    ) %>%
    dplyr::mutate(
      aggregated_score = dplyr::if_else(
        is.na(.data$aggregated_score), 0, .data$aggregated_score
      )
    ) %>%
    dplyr::arrange(dplyr::desc(.data$aggregated_score), .data$gene) %>%
    dplyr::mutate(aggregated_rank = dplyr::row_number())

  audit <- tibble::tibble(
    cascade_cell_type_norm = ct_norm,
    n_genes_in_universe    = nrow(universe_genes),
    n_genes_original       = dplyr::n_distinct(df_part$gene),
    n_genes_completed      = dplyr::n_distinct(completed$gene),
    n_missing_added        = nrow(universe_genes) - dplyr::n_distinct(df_part$gene)
  )

  list(data = completed, audit = audit)
}

map_to_thyroid_gt_cell_type <- function(x) {
  x_low <- stringr::str_to_lower(x)

  dplyr::case_when(
    stringr::str_detect(x_low, "astrocyte") ~ "astrocyte",

    stringr::str_detect(x_low, "oligodendrocyte precursor|\\bopc\\b") ~ "opc",
    stringr::str_detect(x_low, "oligodendrocyte") ~ "oligodendrocyte",

    stringr::str_detect(x_low, "gabaergic|interneuron|vip|lamp5|pvalb|sst|sncg") ~ "gaba_neuron",

    stringr::str_detect(
      x_low,
      "glutamatergic|intratelencephalic|extratelencephalic|corticothalamic|projecting glutamatergic|l2/3|l4/5|l5|l5/6|l6"
    ) ~ "glut_neuron",

    stringr::str_detect(x_low, "tanycyte") ~ "tanycyte",

    TRUE ~ NA_character_
  )
}


#######################################
# STEP 1: PREPARE THYROID CELL-TYPE GT
#######################################

thyroid_celltype_gt <- gt_thyroid_response_standardized %>%
  dplyr::transmute(
    gene               = .data$gene,
    gt_cell_type       = .data$cell_type,
    gt_cell_type_norm  = stringr::str_to_lower(.data$cell_type),
    gt_name            = paste0("thyroid_response_", .data$cell_type),
    gt_type            = "binary",
    binary_label_main  = .data$sig_main,
    abs_score          = .data$abs_score
  )

valid_gt_celltypes <- thyroid_celltype_gt %>%
  dplyr::group_by(.data$gt_cell_type, .data$gt_cell_type_norm, .data$gt_name) %>%
  dplyr::summarise(
    n_genes    = dplyr::n_distinct(.data$gene),
    n_positive = sum(.data$binary_label_main, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::filter(
    .data$n_genes    >= CELLTYPE_MIN_GENES,
    .data$n_positive >= CELLTYPE_MIN_GT_POS
  )


#######################################
# STEPS 2-6: RUN PER PERTURBATION STRATEGY
#
# For each strategy, CASCADE and DE share the same gene universe:
#   - CASCADE universe  = borda gene pool for that strategy x cell type
#   - DE universe       = de_baselines_restricted_to_cascade_universe
#                         filtered to comparison_universe == strategy
#######################################

per_strategy_results <- lapply(PERTURBATION_STRATEGIES, function(ps) {

  message("Running perturbation_strategy = ", ps)

  # --------------------------------------------------
  # STEP 2: CASCADE cell-level data for this strategy
  # --------------------------------------------------

  cascade_cell_level <- cascade_outputs_standardized %>%
    dplyr::mutate(
      cell_key               = paste0("batch_", .data$batch_idx, "__cell_", .data$sample_in_batch),
      cascade_cell_type_raw  = .data$cell_type,
      cascade_cell_type_norm = map_to_thyroid_gt_cell_type(.data$cell_type)
    ) %>%
    dplyr::filter(
      .data$analysis_task        == "treatment",
      .data$perturbation_strategy == ps,
      !is.na(.data$cascade_cell_type_norm)
    )

  # --------------------------------------------------
  # STEP 3: Cell-type-specific CASCADE rankings and universe
  # --------------------------------------------------

  celltype_groups <- cascade_cell_level %>%
    dplyr::semi_join(
      valid_gt_celltypes,
      by = c("cascade_cell_type_norm" = "gt_cell_type_norm")
    ) %>%
    dplyr::group_by(.data$cascade_cell_type_norm) %>%
    dplyr::group_split()

  cell_type_specific_cascade_rankings <- dplyr::bind_rows(
    lapply(celltype_groups, function(df_ct) {

      ct_meta <- df_ct %>%
        dplyr::distinct(.data$cascade_cell_type_norm)

      dplyr::bind_rows(
        lapply(AGGREGATION_METHODS, function(m) {
          aggregate_within_cell_type(df_ct, m) %>%
            dplyr::mutate(cascade_cell_type_norm = ct_meta$cascade_cell_type_norm)
        })
      )
    })
  )

  # CASCADE universe per cell type = borda gene pool (union of all top-100 genes)
  celltype_cascade_universe <- cell_type_specific_cascade_rankings %>%
    dplyr::filter(.data$aggregation_method == "borda") %>%
    dplyr::distinct(.data$cascade_cell_type_norm, .data$gene)

  # --------------------------------------------------
  # STEP 4: DE baselines matched to this strategy's universe
  # --------------------------------------------------

  de_filtered <- de_baselines_restricted_to_cascade_universe %>%
    dplyr::filter(
      .data$analysis_task        == "treatment",
      .data$baseline_type        == "cell_type",
      .data$comparison_universe  == ps          # matches perturbation_strategy
    ) %>%
    dplyr::mutate(
      de_cell_type_raw  = .data$cell_type,
      de_cell_type_norm = map_to_thyroid_gt_cell_type(.data$cell_type)
    ) %>%
    dplyr::filter(!is.na(.data$de_cell_type_norm)) %>%
    dplyr::semi_join(
      valid_gt_celltypes,
      by = c("de_cell_type_norm" = "gt_cell_type_norm")
    )

  # Mirror script 16a: left_join DE to CASCADE universe; zero-fill any gap
  cell_type_specific_de_rankings <- dplyr::bind_rows(
    lapply(
      celltype_cascade_universe %>%
        dplyr::distinct(.data$cascade_cell_type_norm) %>%
        dplyr::pull(.data$cascade_cell_type_norm),
      function(ct) {

        universe_genes <- celltype_cascade_universe %>%
          dplyr::filter(.data$cascade_cell_type_norm == ct) %>%
          dplyr::distinct(.data$gene)

        de_ct  <- de_filtered %>%
          dplyr::filter(.data$de_cell_type_norm == ct)

        ranked <- rank_de_table(de_ct)

        universe_genes %>%
          dplyr::left_join(ranked, by = "gene") %>%
          dplyr::mutate(
            aggregated_score = dplyr::if_else(
              is.na(.data$aggregated_score), 0, .data$aggregated_score
            )
          ) %>%
          dplyr::arrange(dplyr::desc(.data$aggregated_score), .data$gene) %>%
          dplyr::mutate(
            aggregated_rank   = dplyr::row_number(),
            de_cell_type_norm = ct,
            source_label      = paste0("DE_celltype_", ct)
          )
      }
    )
  )

  # --------------------------------------------------
  # STEP 5: Evaluate CASCADE (zero-fill topk to universe before scoring)
  # --------------------------------------------------

  cascade_results <- dplyr::bind_rows(
    lapply(
      cell_type_specific_cascade_rankings %>%
        dplyr::group_by(.data$cascade_cell_type_norm, .data$aggregation_method) %>%
        dplyr::group_split(),
      function(df_rank) {

        meta <- df_rank %>%
          dplyr::distinct(.data$cascade_cell_type_norm, .data$aggregation_method)

        universe_genes <- celltype_cascade_universe %>%
          dplyr::filter(.data$cascade_cell_type_norm == meta$cascade_cell_type_norm) %>%
          dplyr::distinct(.data$gene)

        completed <- complete_cascade_celltype_to_universe(
          df_rank, universe_genes, meta$cascade_cell_type_norm
        )
        df_rank <- completed$data

        gt_sub <- thyroid_celltype_gt %>%
          dplyr::filter(.data$gt_cell_type_norm == meta$cascade_cell_type_norm)

        merged <- df_rank %>%
          dplyr::left_join(
            gt_sub %>%
              dplyr::select(.data$gene, .data$gt_name, .data$binary_label_main, .data$abs_score),
            by = "gene"
          )

        auc_value <- safe_auc(merged$binary_label_main, merged$aggregated_score)

        fisher_tbl <- dplyr::bind_rows(
          lapply(
            TOP_N_VALUES,
            function(n) safe_fisher_from_topn(
              labels = merged$binary_label_main,
              ranks  = merged$aggregated_rank,
              top_n  = n
            )
          )
        )

        fisher_tbl %>%
          dplyr::mutate(
            source_method          = "CASCADE",
            source_label           = paste0("CASCADE_", meta$aggregation_method),
            aggregation_method     = meta$aggregation_method,
            cascade_cell_type_norm = meta$cascade_cell_type_norm,
            gt_name                = unique(gt_sub$gt_name)[1],
            gt_type                = "binary",
            auc                    = auc_value
          )
      }
    )
  )

  # Audit: universe completion per cell-type x method for this strategy
  cascade_audit <- dplyr::bind_rows(
    lapply(
      cell_type_specific_cascade_rankings %>%
        dplyr::group_by(.data$cascade_cell_type_norm, .data$aggregation_method) %>%
        dplyr::group_split(),
      function(df_rank) {
        meta <- df_rank %>%
          dplyr::distinct(.data$cascade_cell_type_norm, .data$aggregation_method)

        universe_genes <- celltype_cascade_universe %>%
          dplyr::filter(.data$cascade_cell_type_norm == meta$cascade_cell_type_norm) %>%
          dplyr::distinct(.data$gene)

        res <- complete_cascade_celltype_to_universe(
          df_rank, universe_genes, meta$cascade_cell_type_norm
        )
        res$audit %>%
          dplyr::mutate(aggregation_method = meta$aggregation_method)
      }
    )
  ) %>%
    dplyr::mutate(perturbation_strategy = ps)

  # --------------------------------------------------
  # STEP 6: Evaluate DE
  # --------------------------------------------------

  de_results <- dplyr::bind_rows(
    lapply(
      cell_type_specific_de_rankings %>%
        dplyr::group_by(.data$de_cell_type_norm, .data$source_label) %>%
        dplyr::group_split(),
      function(df_rank) {

        meta <- df_rank %>%
          dplyr::distinct(.data$de_cell_type_norm, .data$source_label)

        gt_sub <- thyroid_celltype_gt %>%
          dplyr::filter(.data$gt_cell_type_norm == meta$de_cell_type_norm)

        if (nrow(gt_sub) == 0) {
          return(NULL)
        }

        merged <- df_rank %>%
          dplyr::left_join(
            gt_sub %>%
              dplyr::select(.data$gene, .data$gt_name, .data$binary_label_main, .data$abs_score),
            by = "gene"
          )

        auc_value <- safe_auc(merged$binary_label_main, merged$aggregated_score)

        fisher_tbl <- dplyr::bind_rows(
          lapply(
            TOP_N_VALUES,
            function(n) safe_fisher_from_topn(
              labels = merged$binary_label_main,
              ranks  = merged$aggregated_rank,
              top_n  = n
            )
          )
        )

        fisher_tbl %>%
          dplyr::mutate(
            source_method          = "DE",
            source_label           = meta$source_label,
            aggregation_method     = NA_character_,
            cascade_cell_type_norm = meta$de_cell_type_norm,
            gt_name                = unique(gt_sub$gt_name)[1],
            gt_type                = "binary",
            auc                    = auc_value
          )
      }
    )
  )

  list(
    results = dplyr::bind_rows(cascade_results, de_results) %>%
      dplyr::mutate(perturbation_strategy = ps),
    audit   = cascade_audit
  )
})


#######################################
# STEP 7: COMBINE RESULTS ACROSS STRATEGIES
#######################################

cell_type_specific_cascade_results_all <- dplyr::bind_rows(
  lapply(per_strategy_results, `[[`, "results")
)

cascade_celltype_universe_completion_audit <- dplyr::bind_rows(
  lapply(per_strategy_results, `[[`, "audit")
)


#######################################
# STEP 8: SUMMARIES
#######################################

cell_type_specific_cascade_summary <- cell_type_specific_cascade_results_all %>%
  dplyr::group_by(
    .data$perturbation_strategy,
    .data$source_method,
    .data$source_label,
    .data$aggregation_method,
    .data$cascade_cell_type_norm,
    .data$gt_name,
    .data$gt_type
  ) %>%
  dplyr::summarise(
    auc                = max(.data$auc, na.rm = TRUE),
    max_odds_ratio     = max(.data$odds_ratio, na.rm = TRUE),
    min_fisher_p       = min(.data$fisher_p, na.rm = TRUE),
    best_overlap       = max(.data$overlap, na.rm = TRUE),
    max_valid_universe = max(.data$n_valid_universe, na.rm = TRUE),
    .groups = "drop"
  )


#######################################
# STEP 9: WINNER TABLE
#######################################

cascade_rows <- cell_type_specific_cascade_summary %>%
  dplyr::filter(.data$source_method == "CASCADE")

de_rows <- cell_type_specific_cascade_summary %>%
  dplyr::filter(.data$source_method == "DE") %>%
  dplyr::select(
    .data$perturbation_strategy,
    .data$cascade_cell_type_norm,
    .data$gt_name,
    .data$gt_type,
    de_label       = .data$source_label,
    de_auc         = .data$auc,
    de_max_or      = .data$max_odds_ratio,
    de_min_p       = .data$min_fisher_p,
    de_best_overlap = .data$best_overlap
  )

cell_type_specific_cascade_vs_de_winner_table <- cascade_rows %>%
  dplyr::left_join(
    de_rows,
    by = c("perturbation_strategy", "cascade_cell_type_norm", "gt_name", "gt_type")
  ) %>%
  dplyr::mutate(
    winner_auc = dplyr::case_when(
      is.na(.data$de_auc)          ~ "CASCADE",
      .data$auc > .data$de_auc     ~ "CASCADE",
      .data$auc < .data$de_auc     ~ "DE",
      TRUE                         ~ "tie"
    ),
    winner_odds_ratio = dplyr::case_when(
      is.na(.data$de_max_or)                   ~ "CASCADE",
      .data$max_odds_ratio > .data$de_max_or   ~ "CASCADE",
      .data$max_odds_ratio < .data$de_max_or   ~ "DE",
      TRUE                                     ~ "tie"
    )
  )


#######################################
# STEP 10: AGGREGATE SUMMARIES
#######################################

cell_type_specific_combo_summary <- cell_type_specific_cascade_vs_de_winner_table %>%
  dplyr::group_by(.data$perturbation_strategy, .data$aggregation_method) %>%
  dplyr::summarise(
    n_rows      = dplyr::n(),
    n_auc_wins  = sum(.data$winner_auc == "CASCADE", na.rm = TRUE),
    n_or_wins   = sum(.data$winner_odds_ratio == "CASCADE", na.rm = TRUE),
    prop_auc_wins = mean(.data$winner_auc == "CASCADE", na.rm = TRUE),
    prop_or_wins  = mean(.data$winner_odds_ratio == "CASCADE", na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(.data$perturbation_strategy, dplyr::desc(.data$prop_auc_wins))

cell_type_specific_gt_summary <- cell_type_specific_cascade_vs_de_winner_table %>%
  dplyr::group_by(.data$perturbation_strategy, .data$gt_name) %>%
  dplyr::summarise(
    n_rows      = dplyr::n(),
    n_auc_wins  = sum(.data$winner_auc == "CASCADE", na.rm = TRUE),
    n_or_wins   = sum(.data$winner_odds_ratio == "CASCADE", na.rm = TRUE),
    prop_auc_wins = mean(.data$winner_auc == "CASCADE", na.rm = TRUE),
    prop_or_wins  = mean(.data$winner_odds_ratio == "CASCADE", na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(.data$perturbation_strategy, dplyr::desc(.data$prop_auc_wins))


#######################################
# SAVE OUTPUTS
#######################################

readr::write_csv(
  cell_type_specific_cascade_results_all,
  file = here::here("results", "tables", "mu_cell_type_specific_cascade_results.csv")
)

readr::write_csv(
  cell_type_specific_cascade_summary,
  file = here::here("results", "tables", "mu_cell_type_specific_cascade_summary.csv")
)

readr::write_csv(
  cell_type_specific_cascade_vs_de_winner_table,
  file = here::here("results", "tables", "mu_cell_type_specific_cascade_vs_de_winner_table.csv")
)

readr::write_csv(
  cell_type_specific_combo_summary,
  file = here::here("results", "tables", "mu_cell_type_specific_combo_summary.csv")
)

readr::write_csv(
  cell_type_specific_gt_summary,
  file = here::here("results", "tables", "mu_cell_type_specific_gt_summary.csv")
)

readr::write_csv(
  cascade_celltype_universe_completion_audit,
  file = here::here("results", "tables", "mu_cascade_celltype_universe_completion_audit.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Saved matched-universe cell-type-specific CASCADE vs DE analysis.")
print(utils::head(cell_type_specific_combo_summary, 30))
print(utils::head(cell_type_specific_gt_summary, 20))
