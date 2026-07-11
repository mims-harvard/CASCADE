################################################################################
# Script: 04_standardize_de_baselines.R
#
# Purpose:
#   Standardize DE baseline files into a harmonized long-format object.
#
# Outputs:
#   - results/rds/de_baselines_standardized.rds
#   - results/tables/de_baselines_summary.csv
################################################################################


#######################################
# LOAD CONFIG
#######################################

source(here::here("scripts", "00_config.R"))


#######################################
# LOAD PACKAGES
#######################################

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(readr)
  library(tibble)
})


#######################################
# HELPER FUNCTION
#######################################

standardize_one_de_file <- function(dataset_name,
                                    task,
                                    baseline_type,
                                    path) {
  
  message("Standardizing DE file: ", dataset_name)
  
  df <- data.table::fread(
    input = path,
    data.table = FALSE,
    fill = TRUE,
    quote = "\"",
    encoding = "UTF-8"
  )
  
  required_cols <- c(
    "task", "scope", "names", "scores",
    "logfoldchanges", "pvals", "pvals_adj", "pct_nz_group"
  )
  
  missing_cols <- setdiff(required_cols, colnames(df))
  if (length(missing_cols) > 0) {
    stop(
      "Missing required columns in ", dataset_name, ": ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  # Some DE files may have cell-type-specific columns beyond the shared ones.
  extra_cols <- setdiff(colnames(df), required_cols)
  
  df_std <- df %>%
    transmute(
      dataset_name = dataset_name,
      source = "de",
      analysis_task = task,
      baseline_type = baseline_type,
      original_task = .data$task,
      scope = as.character(.data$scope),
      de_gene_raw = as.character(.data$names),
      gene = as.character(.data$names),
      gene_id_numeric = as.character(.data$names),
      gene_symbol = NA_character_,
      de_score = as.numeric(.data$scores),
      log2fc = as.numeric(.data$logfoldchanges),
      p_value = as.numeric(.data$pvals),
      p_adj = as.numeric(.data$pvals_adj),
      pct_nz_group = as.numeric(.data$pct_nz_group)
    )
  
  # Keep extra columns if they exist, especially for cell-type-specific files
  if (length(extra_cols) > 0) {
    df_std <- bind_cols(
      df_std,
      df[, extra_cols, drop = FALSE]
    )
  }
  
  summary_tbl <- tibble(
    dataset_name = dataset_name,
    analysis_task = task,
    baseline_type = baseline_type,
    n_rows = nrow(df_std),
    n_unique_genes = n_distinct(df_std$gene),
    n_unique_scopes = n_distinct(df_std$scope),
    min_p_adj = min(df_std$p_adj, na.rm = TRUE),
    max_abs_log2fc = max(abs(df_std$log2fc), na.rm = TRUE)
  )
  
  list(
    data = df_std,
    summary = summary_tbl
  )
}


#######################################
# RUN STANDARDIZATION
#######################################

de_results <- vector("list", nrow(DE_FILES))

for (i in seq_len(nrow(DE_FILES))) {
  de_results[[i]] <- standardize_one_de_file(
    dataset_name = DE_FILES$dataset_name[i],
    task = DE_FILES$task[i],
    baseline_type = DE_FILES$baseline_type[i],
    path = DE_FILES$path[i]
  )
}

de_baselines_standardized <- bind_rows(lapply(de_results, `[[`, "data"))
de_baselines_summary <- bind_rows(lapply(de_results, `[[`, "summary"))


#######################################
# SAVE OUTPUTS
#######################################

saveRDS(
  de_baselines_standardized,
  file = here::here("results", "rds", "de_baselines_standardized.rds")
)

readr::write_csv(
  de_baselines_summary,
  file = here::here("results", "tables", "de_baselines_summary.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Saved standardized DE baselines.")
print(de_baselines_summary)
