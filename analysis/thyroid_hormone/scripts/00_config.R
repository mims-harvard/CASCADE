################################################################################
# Script: 00_config.R
#
# Purpose:
#   Central configuration file for the CASCADE thyroid validation analysis
#   (Figure 4; Methods "CASCADE-Explainer recovers thyroid hormone and receptor
#   signalling gene programmes"; Supplementary Notes 22-23).
#   Defines project paths, input files, constants, and helper directory setup.
#
# Notes:
#   - Source this script at the top of every downstream analysis script.
#   - All paths are resolved via here::here(), anchored at analysis/thyroid_hormone/
#     (see the .here sentinel file in that directory).
#   - This script creates required output directories if they do not exist.
#
# Usage:
#   source("scripts/00_config.R")
################################################################################


#######################################
# LOAD PACKAGES
#######################################

suppressPackageStartupMessages({
  library(here)
})


#######################################
# PROJECT ROOT
#######################################

PROJECT_ROOT <- here::here()


#######################################
# DIRECTORY STRUCTURE
#######################################

DIRS <- list(
  data_raw       = here::here("data"),
  data_processed = here::here("data", "processed"),
  scripts        = here::here("scripts"),
  src            = here::here("src"),
  results        = here::here("results"),
  results_tables = here::here("results", "tables"),
  results_figures= here::here("results", "figures"),
  results_rds    = here::here("results", "rds"),
  results_reports= here::here("results", "reports")
)


#######################################
# CREATE OUTPUT DIRECTORIES
#######################################

required_dirs <- unname(unlist(DIRS))

for (dir_path in required_dirs) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  }
}


#######################################
# INPUT FILES: CASCADE OUTPUTS
#######################################

CASCADE_FILES <- data.frame(
  dataset_name = c(
    "cascade_treatment_cell_type",
    "cascade_treatment_treatment_marker",
    "cascade_receptor_cell_type",
    "cascade_receptor_treatment_marker"
  ),
  task = c(
    "treatment",
    "treatment",
    "THR",
    "THR"
  ),
  perturbation_strategy = c(
    "cell_type",
    "treatment",
    "cell_type",
    "treatment"
  ),
  path = c(
    here::here("data", "cascade_outputs", "cell_type", "treatment", "thryroid_treatment.csv"),
    here::here("data", "cascade_outputs", "treatment", "treatment", "thryroid_treatment.csv"),
    here::here("data", "cascade_outputs", "cell_type", "thr", "thryroid_thr.csv"),
    here::here("data", "cascade_outputs", "treatment", "thr", "thryroid_thr.csv")
  ),
  stringsAsFactors = FALSE
)


#######################################
# INPUT FILES: PREDICTIONS
#######################################

PREDICTION_FILES <- data.frame(
  dataset_name = c(
    "pred_treatment_cell_type",
    "pred_treatment_treatment_marker",
    "pred_receptor_cell_type",
    "pred_receptor_treatment_marker"
  ),
  task = c(
    "treatment",
    "treatment",
    "THR",
    "THR"
  ),
  perturbation_strategy = c(
    "cell_type",
    "treatment",
    "cell_type",
    "treatment"
  ),
  path = c(
    here::here("data", "cascade_outputs", "cell_type", "treatment", "predictions_treatment_context_cells.csv"),
    here::here("data", "cascade_outputs", "treatment", "treatment", "predictions_treatment_context_disease.csv"),
    here::here("data", "cascade_outputs", "cell_type", "thr", "predictions_THR_context_cells.csv"),
    here::here("data", "cascade_outputs", "treatment", "thr", "predictions_THR_context_disease.csv")
  ),
  stringsAsFactors = FALSE
)


#######################################
# INPUT FILES: DE BASELINES
#######################################

DE_FILES <- data.frame(
  dataset_name = c(
    "de_pooled_treatment",
    "de_cell_type_specific_treatment",
    "de_pooled_THR",
    "de_cell_type_specific_THR"
  ),
  task = c(
    "treatment",
    "treatment",
    "THR",
    "THR"
  ),
  baseline_type = c(
    "pooled",
    "cell_type",
    "pooled",
    "cell_type"
  ),
  path = c(
    here::here("data", "baselines", "de_pooled_treatment.csv"),
    here::here("data", "baselines", "de_cell_type_specific_treatment.csv"),
    here::here("data", "baselines", "de_pooled_THR.csv"),
    here::here("data", "baselines", "de_cell_type_specific_THR.csv")
  ),
  stringsAsFactors = FALSE
)


#######################################
# INPUT FILES: GROUND TRUTHS
#######################################

GROUND_TRUTH_FILES <- data.frame(
  dataset_name = c(
    "gt_thyroid_response_emtab14810",
    "gt_mmc2"
  ),
  gt_family = c(
    "thyroid_response",
    "mmc2"
  ),
  path = c(
    here::here("data", "gt", "Task1_external_ground_truth_EMTAB14810.csv"),
    here::here("data", "gt", "mmc2.xlsx")
  ),
  stringsAsFactors = FALSE
)

MAPPING_FILES <- data.frame(
  dataset_name = "mapping_m2_de",
  mapping_type = "de_to_cascade",
  path = here::here("data", "gt", "mapping_m2_de.csv"),
  stringsAsFactors = FALSE
)

#######################################
# SPECIFIC GENES OF INTEREST
#######################################

GENES_OF_INTEREST <- c(
  "Ncor1", "Ncor2", "Thra", "Thrb", "Robo3",
  "Npnt", "Ephb1", "Ephb2", "Hr"
)


#######################################
# ANALYSIS CONSTANTS
#######################################

ANALYSIS_PARAMS <- list(
  top_k_values = c(10, 25, 50),
  overlap_top_n = c(50, 100, 200, 500),
  importance_threshold_absolute = 0.02,
  importance_threshold_quantile = 0.85,
  gt_adj_p_thresh = 0.05,
  gt_abs_logfc_thresh_main = 0.5,
  gt_abs_logfc_thresh_sensitivity = 1.0
)


#######################################
# COMBINED FILE REGISTRY
#######################################

suppressPackageStartupMessages({
  library(dplyr)
})

FILE_REGISTRY <- dplyr::bind_rows(
  CASCADE_FILES %>% dplyr::mutate(file_category = "cascade"),
  PREDICTION_FILES %>% dplyr::mutate(file_category = "predictions"),
  DE_FILES %>% dplyr::mutate(file_category = "de_baseline"),
  GROUND_TRUTH_FILES %>% dplyr::mutate(file_category = "ground_truth"),
  MAPPING_FILES %>% dplyr::mutate(file_category = "mapping")
) %>%
  dplyr::select(file_category, everything())


#######################################
# OUTPUT FILES
#######################################

OUTPUT_FILES <- list(
  input_manifest_csv = here::here("results", "tables", "input_file_manifest.csv"),
  input_preview_rds  = here::here("results", "rds", "input_file_previews.rds"),
  session_info_txt   = here::here("results", "tables", "session_info.txt")
)


#######################################
# HELPER MESSAGE
#######################################

message("Configuration loaded successfully.")
message("Project root: ", PROJECT_ROOT)
message("Registered files: ", nrow(FILE_REGISTRY))
