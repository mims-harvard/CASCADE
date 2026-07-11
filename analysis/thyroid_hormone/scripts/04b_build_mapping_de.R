################################################################################
# Script: 04b_build_de_mapping.R
#
# Purpose:
#   Read the DE↔CASCADE mapping file, standardize its columns, and create a
#   reusable mapping object for DE baseline translation into CASCADE space.
#
# Outputs:
#   - results/rds/de_to_cascade_mapping.rds
#   - results/tables/de_to_cascade_mapping_summary.csv
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
  library(janitor)
  library(tibble)
})


#######################################
# READ RAW MAPPING
#######################################

mapping_raw <- data.table::fread(
  MAPPING_FILES$path[1],
  data.table = FALSE,
  fill = TRUE,
  quote = "\"",
  encoding = "UTF-8"
)

original_colnames <- colnames(mapping_raw)
mapping_clean_names <- janitor::make_clean_names(original_colnames)
colnames(mapping_raw) <- mapping_clean_names


#######################################
# USER-EDITABLE COLUMN MAPPING
#######################################
# IMPORTANT:
# Replace these with the actual columns in mapping_m2_de.csv after inspecting:
# print(colnames(mapping_raw))

COL_DE_ID <- "index"
COL_CASCADE_ID <- "ensembl_gene_id"

# Optional extra identifiers if present
COL_DE_SYMBOL <- if ("de_symbol" %in% colnames(mapping_raw)) "de_symbol" else NA_character_
COL_CASCADE_SYMBOL <- if ("cascade_symbol" %in% colnames(mapping_raw)) "cascade_symbol" else NA_character_
COL_ENSEMBL <- if ("ensembl_id" %in% colnames(mapping_raw)) "ensembl_id" else NA_character_


#######################################
# VALIDATE REQUIRED COLUMNS
#######################################

required_cols <- c(COL_DE_ID, COL_CASCADE_ID)
missing_cols <- required_cols[!required_cols %in% colnames(mapping_raw)]

if (length(missing_cols) > 0) {
  stop(
    "The following mapping columns were not found in mapping_m2_de.csv: ",
    paste(missing_cols, collapse = ", "),
    ". Inspect colnames(mapping_raw) and update COL_DE_ID / COL_CASCADE_ID."
  )
}


#######################################
# STANDARDIZE MAPPING
#######################################

de_to_cascade_mapping <- mapping_raw %>%
  transmute(
    de_gene = as.character(.data[[COL_DE_ID]]),
    cascade_gene = as.character(.data[[COL_CASCADE_ID]]),
    de_symbol = if (!is.na(COL_DE_SYMBOL)) as.character(.data[[COL_DE_SYMBOL]]) else NA_character_,
    cascade_symbol = if (!is.na(COL_CASCADE_SYMBOL)) as.character(.data[[COL_CASCADE_SYMBOL]]) else NA_character_,
    ensembl_id = if (!is.na(COL_ENSEMBL)) as.character(.data[[COL_ENSEMBL]]) else NA_character_
  ) %>%
  filter(!is.na(.data$de_gene), !is.na(.data$cascade_gene)) %>%
  distinct()


#######################################
# SUMMARY
#######################################

de_to_cascade_mapping_summary <- tibble(
  n_rows = nrow(de_to_cascade_mapping),
  n_unique_de_gene = n_distinct(de_to_cascade_mapping$de_gene),
  n_unique_cascade_gene = n_distinct(de_to_cascade_mapping$cascade_gene),
  n_multi_map_de = de_to_cascade_mapping %>%
    count(.data$de_gene) %>%
    filter(.data$n > 1) %>%
    nrow(),
  n_multi_map_cascade = de_to_cascade_mapping %>%
    count(.data$cascade_gene) %>%
    filter(.data$n > 1) %>%
    nrow()
)


#######################################
# SAVE
#######################################

saveRDS(
  de_to_cascade_mapping,
  file = here::here("results", "rds", "de_to_cascade_mapping.rds")
)

readr::write_csv(
  de_to_cascade_mapping_summary,
  file = here::here("results", "tables", "de_to_cascade_mapping_summary.csv")
)

message("Saved DE↔CASCADE mapping.")
print(de_to_cascade_mapping_summary)
print(colnames(mapping_raw))