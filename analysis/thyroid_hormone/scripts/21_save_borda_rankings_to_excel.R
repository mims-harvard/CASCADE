################################################################################
# Script: 21_save_borda_rankings_to_excel.R
#
# Purpose:
#   Save Borda aggregated rankings to Excel workbooks:
#   1. all-cells rankings
#   2. cell-type-specific rankings (astrocytes and glutamatergic neurons)
#
# Outputs:
#   - results/xlsx/borda_all_cells_rankings.xlsx
#   - results/xlsx/borda_cell_type_specific_rankings.xlsx
################################################################################

#######################################
# LOAD CONFIG
#######################################

source(here::here("scripts", "00_config.R"))
TARGET_THYROID_CELL_TYPES <- c("astrocyte", "glut_neuron")

map_to_thyroid_gt_cell_type <- function(x) {
  x_low <- stringr::str_to_lower(x)

  dplyr::case_when(
    stringr::str_detect(x_low, "astrocyte") ~ "astrocyte",

    stringr::str_detect(
      x_low,
      "glutamatergic|intratelencephalic|extratelencephalic|corticothalamic|projecting glutamatergic|l2/3|l4/5|l5|l5/6|l6"
    ) ~ "glut_neuron",

    TRUE ~ NA_character_
  )
}

#######################################
# LOAD PACKAGES
#######################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tibble)
  library(openxlsx)
  library(stringr)
})

#######################################
# LOAD INPUT OBJECTS
#######################################

cascade_outputs_standardized <- readRDS(
  here::here("results", "rds", "cascade_outputs_standardized.rds")
)

mmc2_symbol_map <- readRDS(
  here::here("results", "rds", "mmc2_symbol_map.rds")
)

#######################################
# HELPERS
#######################################

make_cell_key <- function(df) {
  df %>%
    dplyr::mutate(
      cell_key = paste0("batch_", .data$batch_idx, "__cell_", .data$sample_in_batch)
    )
}

aggregate_borda <- function(df) {
  max_rank <- max(df$rank, na.rm = TRUE)

  df %>%
    dplyr::mutate(
      borda_score = max_rank - .data$rank + 1
    ) %>%
    dplyr::group_by(.data$gene) %>%
    dplyr::summarise(
      aggregated_score = sum(.data$borda_score, na.rm = TRUE),
      n_rows_used = dplyr::n(),
      n_cells_used = dplyr::n_distinct(.data$cell_key),
      .groups = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(.data$aggregated_score), .data$gene) %>%
    dplyr::mutate(
      aggregated_rank = dplyr::row_number()
    )
}

safe_sheet_name <- function(x) {
  x %>%
    gsub("[\\\\/*?:\\[\\]]", "_", .) %>%
    substr(1, 31)
}

write_pretty_sheet <- function(wb, sheet_name, df) {
  openxlsx::addWorksheet(wb, sheet_name, gridLines = FALSE)
  openxlsx::writeDataTable(
    wb, sheet = sheet_name, x = df, tableStyle = "TableStyleMedium2",
    withFilter = TRUE
  )

  header_style <- openxlsx::createStyle(
    textDecoration = "bold",
    fgFill = "#1F4E78",
    fontColour = "#FFFFFF",
    halign = "center"
  )
  openxlsx::addStyle(
    wb, sheet = sheet_name, style = header_style,
    rows = 1, cols = 1:ncol(df), gridExpand = TRUE, stack = TRUE
  )

  openxlsx::freezePane(wb, sheet = sheet_name, firstRow = TRUE)
  openxlsx::setColWidths(wb, sheet = sheet_name, cols = 1:ncol(df), widths = "auto")
}

#######################################
# GENE LOOKUP
#######################################

gene_lookup <- mmc2_symbol_map %>%
  dplyr::select(
    gene = .data$ensembl_gene_id,
    gene_symbol = .data$gene_symbol
  ) %>%
  dplyr::filter(!is.na(.data$gene), !is.na(.data$gene_symbol), .data$gene_symbol != "") %>%
  dplyr::distinct(.data$gene, .data$gene_symbol)

#######################################
# PREPARE INPUT
#######################################

cascade_cell_level <- cascade_outputs_standardized %>%
  make_cell_key()

#######################################
# ALL-CELLS BORDA
#######################################

borda_all_cells_rankings <- cascade_cell_level %>%
  dplyr::group_by(
    .data$dataset_name,
    .data$analysis_task,
    .data$perturbation_strategy
  ) %>%
  dplyr::group_split() %>%
  lapply(function(df_part) {

    meta <- df_part %>%
      dplyr::distinct(
        .data$dataset_name,
        .data$analysis_task,
        .data$perturbation_strategy
      )

    aggregate_borda(df_part) %>%
      dplyr::left_join(gene_lookup, by = "gene") %>%
      dplyr::mutate(
        dataset_name = meta$dataset_name,
        analysis_task = meta$analysis_task,
        perturbation_strategy = meta$perturbation_strategy,
        aggregation_method = "borda",
        aggregation_scope = "all_cells"
      ) %>%
      dplyr::select(
        .data$dataset_name,
        .data$analysis_task,
        .data$perturbation_strategy,
        .data$aggregation_method,
        .data$aggregation_scope,
        .data$gene,
        .data$gene_symbol,
        .data$aggregated_score,
        .data$aggregated_rank,
        .data$n_rows_used,
        .data$n_cells_used
      )
  }) %>%
  dplyr::bind_rows()

#######################################
# CELL-TYPE-SPECIFIC BORDA
#######################################
borda_cell_type_specific_rankings <- cascade_cell_level %>%
  dplyr::filter(!is.na(.data$cell_type), .data$cell_type != "") %>%
  dplyr::mutate(
    broad_cell_type = map_to_thyroid_gt_cell_type(.data$cell_type)
  ) %>%
  dplyr::filter(.data$broad_cell_type %in% TARGET_THYROID_CELL_TYPES) %>%
  dplyr::group_by(
    .data$dataset_name,
    .data$analysis_task,
    .data$perturbation_strategy,
    .data$broad_cell_type
  ) %>%
  dplyr::group_split() %>%
  lapply(function(df_part) {

    meta <- df_part %>%
      dplyr::distinct(
        .data$dataset_name,
        .data$analysis_task,
        .data$perturbation_strategy,
        .data$broad_cell_type
      )

    aggregate_borda(df_part) %>%
      dplyr::left_join(gene_lookup, by = "gene") %>%
      dplyr::mutate(
        dataset_name = meta$dataset_name,
        analysis_task = meta$analysis_task,
        perturbation_strategy = meta$perturbation_strategy,
        cell_type = meta$broad_cell_type,
        aggregation_method = "borda",
        aggregation_scope = "cell_type_specific"
      ) %>%
      dplyr::select(
        .data$dataset_name,
        .data$analysis_task,
        .data$perturbation_strategy,
        .data$cell_type,
        .data$aggregation_method,
        .data$aggregation_scope,
        .data$gene,
        .data$gene_symbol,
        .data$aggregated_score,
        .data$aggregated_rank,
        .data$n_rows_used,
        .data$n_cells_used
      )
  }) %>%
  dplyr::bind_rows()
#######################################
# WRITE EXCEL: ALL CELLS
#######################################

wb_all <- openxlsx::createWorkbook()

all_sheet_keys <- borda_all_cells_rankings %>%
  dplyr::distinct(.data$dataset_name, .data$analysis_task, .data$perturbation_strategy) %>%
  dplyr::arrange(.data$dataset_name, .data$analysis_task, .data$perturbation_strategy)

for (i in seq_len(nrow(all_sheet_keys))) {
  ds <- all_sheet_keys$dataset_name[i]
  task <- all_sheet_keys$analysis_task[i]
  strat <- all_sheet_keys$perturbation_strategy[i]

  df_sheet <- borda_all_cells_rankings %>%
    dplyr::filter(
      .data$dataset_name == ds,
      .data$analysis_task == task,
      .data$perturbation_strategy == strat
    ) %>%
    dplyr::arrange(.data$aggregated_rank)

  sheet_name <- safe_sheet_name(paste(ds, task, strat, sep = "_"))
  write_pretty_sheet(wb_all, sheet_name, df_sheet)
}

dir.create(here::here("results", "xlsx"), recursive = TRUE, showWarnings = FALSE)

openxlsx::saveWorkbook(
  wb_all,
  file = here::here("results", "xlsx", "borda_all_cells_rankings.xlsx"),
  overwrite = TRUE
)

#######################################
# WRITE EXCEL: CELL-TYPE-SPECIFIC
#######################################

wb_ct <- openxlsx::createWorkbook()

ct_sheet_keys <- borda_cell_type_specific_rankings %>%
  dplyr::distinct(.data$dataset_name, .data$analysis_task, .data$perturbation_strategy) %>%
  dplyr::arrange(.data$dataset_name, .data$analysis_task, .data$perturbation_strategy)

for (i in seq_len(nrow(ct_sheet_keys))) {
  ds <- ct_sheet_keys$dataset_name[i]
  task <- ct_sheet_keys$analysis_task[i]
  strat <- ct_sheet_keys$perturbation_strategy[i]

  df_sheet <- borda_cell_type_specific_rankings %>%
    dplyr::filter(
      .data$dataset_name == ds,
      .data$analysis_task == task,
      .data$perturbation_strategy == strat
    ) %>%
    dplyr::arrange(.data$cell_type, .data$aggregated_rank)

  sheet_name <- safe_sheet_name(paste(ds, task, strat, sep = "_"))
  write_pretty_sheet(wb_ct, sheet_name, df_sheet)
}

openxlsx::saveWorkbook(
  wb_ct,
  file = here::here("results", "xlsx", "borda_cell_type_specific_rankings.xlsx"),
  overwrite = TRUE
)

#######################################
# CONSOLE CHECK
#######################################

message("Saved Excel workbooks:")
message(" - results/xlsx/borda_all_cells_rankings.xlsx")
message(" - results/xlsx/borda_cell_type_specific_rankings.xlsx")
