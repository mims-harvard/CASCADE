################################################################################
# Script: 05_build_ground_truth_objects.R
#
# Purpose:
#   Standardize and structure the thyroid response GT and MMC2 GT into explicit
#   analysis-ready objects.
#
# Outputs:
#   - results/rds/gt_thyroid_response_standardized.rds
#   - results/rds/gt_mmc2_long.rds
#   - results/rds/gt_mmc2_definitions.rds
#   - results/tables/gt_thyroid_response_summary.csv
#   - results/tables/gt_mmc2_summary.csv
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
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(tibble)
  library(stringr)
})


#######################################
# STEP 1: THYROID RESPONSE GROUND TRUTH
#######################################

gt_thyroid <- data.table::fread(
  input = GROUND_TRUTH_FILES$path[GROUND_TRUTH_FILES$dataset_name == "gt_thyroid_response_emtab14810"],
  data.table = FALSE,
  fill = TRUE,
  quote = "\"",
  encoding = "UTF-8"
)

required_thyroid_cols <- c(
  "gene_id", "p_val", "avg_log2FC", "pct.1", "pct.2",
  "p_val_adj", "gene_symbol", "cell_type", "abs_score"
)

missing_thyroid_cols <- setdiff(required_thyroid_cols, colnames(gt_thyroid))
if (length(missing_thyroid_cols) > 0) {
  stop(
    "Missing required thyroid GT columns: ",
    paste(missing_thyroid_cols, collapse = ", ")
  )
}

gt_thyroid_response_standardized <- gt_thyroid %>%
  transmute(
    gt_family = "thyroid_response",
    gene = as.character(.data$gene_id),
    gene_symbol = as.character(.data$gene_symbol),
    gene_id = as.character(.data$gene_id),
    p_value = as.numeric(.data$p_val),
    log2fc = as.numeric(.data$avg_log2FC),
    pct_1 = as.numeric(.data$`pct.1`),
    pct_2 = as.numeric(.data$`pct.2`),
    p_adj = as.numeric(.data$p_val_adj),
    cell_type = as.character(.data$cell_type),
    abs_score = as.numeric(.data$abs_score),
    direction = case_when(
      .data$log2fc > 0 ~ "up",
      .data$log2fc < 0 ~ "down",
      TRUE ~ "zero"
    ),
    sig_main = .data$p_adj < ANALYSIS_PARAMS$gt_adj_p_thresh &
      abs(.data$log2fc) > ANALYSIS_PARAMS$gt_abs_logfc_thresh_main,
    sig_sensitivity = .data$p_adj < ANALYSIS_PARAMS$gt_adj_p_thresh &
      abs(.data$log2fc) > ANALYSIS_PARAMS$gt_abs_logfc_thresh_sensitivity
  )

gt_thyroid_response_summary <- gt_thyroid_response_standardized %>%
  summarise(
    n_rows = n(),
    n_unique_genes = n_distinct(.data$gene),
    n_cell_types = n_distinct(.data$cell_type),
    n_sig_main = sum(.data$sig_main, na.rm = TRUE),
    n_sig_sensitivity = sum(.data$sig_sensitivity, na.rm = TRUE)
  )


#######################################
# STEP 2: MMC2 GROUND TRUTH
#######################################

gt_mmc2_raw <- readxl::read_excel(
  path = GROUND_TRUTH_FILES$path[GROUND_TRUTH_FILES$dataset_name == "gt_mmc2"],
  sheet = 1,
  na = c("", "NA", "N/A", "na")
) %>%
  as.data.frame()

# Clean names only minimally so downstream labels stay understandable
colnames(gt_mmc2_raw) <- trimws(colnames(gt_mmc2_raw))

required_mmc2_cols <- c(
  "Gene name",
  "ThraAMI/gnvs Control (PND14) BaseMean",
  "ThraAMI/gnvs Control (PND14) Log2FC",
  "ThraAMI/gnvs Control (PND14) Adjusted p-value",
  "Hypothyroid vs Euthyroid (PND21) BaseMean",
  "Hypothyroid vs Euthyroid (PND21) Log2(FC)",
  "Hypothyroid vs Euthyroid (PND21) Adjusted p-value",
  "TH-treated vs Hypothyroid (PND21) BaseMean",
  "TH-treated vs Hypothyroid (PND21) Log2(FC)",
  "TH-treated vs Hypothyroid (PND21) Adjusted p-value",
  "TRBS within 30kb of TSS",
  "DR4",
  "Down in ThraAMI/gn mice",
  "Down in Hypothyroid mice",
  "Up after TH treatment",
  "Up in ThraAMI/gn mice",
  "Up in Hypothyroid mice",
  "Down after TH treatment"
)

missing_mmc2_cols <- setdiff(required_mmc2_cols, colnames(gt_mmc2_raw))
if (length(missing_mmc2_cols) > 0) {
  stop(
    "Missing required MMC2 columns: ",
    paste(missing_mmc2_cols, collapse = ", ")
  )
}

gt_mmc2_clean <- gt_mmc2_raw %>%
  transmute(
    gene = as.character(`Gene name`),
    
    thra_base_mean = as.numeric(`ThraAMI/gnvs Control (PND14) BaseMean`),
    thra_log2fc = as.numeric(`ThraAMI/gnvs Control (PND14) Log2FC`),
    thra_p_adj = as.numeric(`ThraAMI/gnvs Control (PND14) Adjusted p-value`),
    
    hypo_base_mean = as.numeric(`Hypothyroid vs Euthyroid (PND21) BaseMean`),
    hypo_log2fc = as.numeric(`Hypothyroid vs Euthyroid (PND21) Log2(FC)`),
    hypo_p_adj = as.numeric(`Hypothyroid vs Euthyroid (PND21) Adjusted p-value`),
    
    th_treat_base_mean = as.numeric(`TH-treated vs Hypothyroid (PND21) BaseMean`),
    th_treat_log2fc = as.numeric(`TH-treated vs Hypothyroid (PND21) Log2(FC)`),
    th_treat_p_adj = as.numeric(`TH-treated vs Hypothyroid (PND21) Adjusted p-value`),
    
    trbs_30kb = if_else(`TRBS within 30kb of TSS` == "X", 1L, 0L, missing = 0L),
    dr4 = if_else(DR4 == "X", 1L, 0L, missing = 0L),
    
    down_thra = if_else(`Down in ThraAMI/gn mice` == "DOWN", 1L, 0L, missing = 0L),
    up_thra = if_else(`Up in ThraAMI/gn mice` == "UP", 1L, 0L, missing = 0L),
    
    down_hypo = if_else(`Down in Hypothyroid mice` == "DOWN", 1L, 0L, missing = 0L),
    up_hypo = if_else(`Up in Hypothyroid mice` == "UP", 1L, 0L, missing = 0L),
    
    up_after_th = if_else(`Up after TH treatment` == "UP", 1L, 0L, missing = 0L),
    down_after_th = if_else(`Down after TH treatment` == "DOWN", 1L, 0L, missing = 0L)
  )


#######################################
# STEP 3: EXPLICIT MMC2 GROUND TRUTH DEFINITIONS
#######################################

gt_mmc2_definitions <- tibble(
  gt_name = c(
    "ThraAMI_gn_vs_Control_PND14_continuous",
    "Hypothyroid_vs_Euthyroid_PND21_continuous",
    "TH_treated_vs_Hypothyroid_PND21_continuous",
    "TRBS_within_30kb_of_TSS",
    "DR4",
    "Down_in_ThraAMI_gn_mice",
    "Up_in_ThraAMI_gn_mice",
    "Down_in_Hypothyroid_mice",
    "Up_in_Hypothyroid_mice",
    "Up_after_TH_treatment",
    "Down_after_TH_treatment"
  ),
  gt_type = c(
    "continuous", "continuous", "continuous",
    "binary", "binary",
    "binary", "binary",
    "binary", "binary",
    "binary", "binary"
  ),
  score_col = c(
    "thra_log2fc",
    "hypo_log2fc",
    "th_treat_log2fc",
    "trbs_30kb",
    "dr4",
    "down_thra",
    "up_thra",
    "down_hypo",
    "up_hypo",
    "up_after_th",
    "down_after_th"
  ),
  p_adj_col = c(
    "thra_p_adj",
    "hypo_p_adj",
    "th_treat_p_adj",
    NA, NA, NA, NA, NA, NA, NA, NA
  )
)


#######################################
# STEP 4: LONG-FORM MMC2 OBJECT
#######################################

gt_mmc2_long <- bind_rows(
  
  # Continuous contrasts
  gt_mmc2_clean %>%
    transmute(
      gene,
      gt_name = "ThraAMI_gn_vs_Control_PND14_continuous",
      gt_type = "continuous",
      score = thra_log2fc,
      abs_score = abs(thra_log2fc),
      p_adj = thra_p_adj,
      binary_label_main = thra_p_adj < ANALYSIS_PARAMS$gt_adj_p_thresh &
        abs(thra_log2fc) > ANALYSIS_PARAMS$gt_abs_logfc_thresh_main,
      binary_label_sensitivity = thra_p_adj < ANALYSIS_PARAMS$gt_adj_p_thresh &
        abs(thra_log2fc) > ANALYSIS_PARAMS$gt_abs_logfc_thresh_sensitivity,
      direction = case_when(
        thra_log2fc > 0 ~ "up",
        thra_log2fc < 0 ~ "down",
        TRUE ~ "zero"
      )
    ),
  
  gt_mmc2_clean %>%
    transmute(
      gene,
      gt_name = "Hypothyroid_vs_Euthyroid_PND21_continuous",
      gt_type = "continuous",
      score = hypo_log2fc,
      abs_score = abs(hypo_log2fc),
      p_adj = hypo_p_adj,
      binary_label_main = hypo_p_adj < ANALYSIS_PARAMS$gt_adj_p_thresh &
        abs(hypo_log2fc) > ANALYSIS_PARAMS$gt_abs_logfc_thresh_main,
      binary_label_sensitivity = hypo_p_adj < ANALYSIS_PARAMS$gt_adj_p_thresh &
        abs(hypo_log2fc) > ANALYSIS_PARAMS$gt_abs_logfc_thresh_sensitivity,
      direction = case_when(
        hypo_log2fc > 0 ~ "up",
        hypo_log2fc < 0 ~ "down",
        TRUE ~ "zero"
      )
    ),
  
  gt_mmc2_clean %>%
    transmute(
      gene,
      gt_name = "TH_treated_vs_Hypothyroid_PND21_continuous",
      gt_type = "continuous",
      score = th_treat_log2fc,
      abs_score = abs(th_treat_log2fc),
      p_adj = th_treat_p_adj,
      binary_label_main = th_treat_p_adj < ANALYSIS_PARAMS$gt_adj_p_thresh &
        abs(th_treat_log2fc) > ANALYSIS_PARAMS$gt_abs_logfc_thresh_main,
      binary_label_sensitivity = th_treat_p_adj < ANALYSIS_PARAMS$gt_adj_p_thresh &
        abs(th_treat_log2fc) > ANALYSIS_PARAMS$gt_abs_logfc_thresh_sensitivity,
      direction = case_when(
        th_treat_log2fc > 0 ~ "up",
        th_treat_log2fc < 0 ~ "down",
        TRUE ~ "zero"
      )
    ),
  
  # Binary sets
  gt_mmc2_clean %>%
    transmute(
      gene, gt_name = "TRBS_within_30kb_of_TSS", gt_type = "binary",
      score = trbs_30kb, abs_score = trbs_30kb, p_adj = NA_real_,
      binary_label_main = trbs_30kb == 1L,
      binary_label_sensitivity = trbs_30kb == 1L,
      direction = NA_character_
    ),
  
  gt_mmc2_clean %>%
    transmute(
      gene, gt_name = "DR4", gt_type = "binary",
      score = dr4, abs_score = dr4, p_adj = NA_real_,
      binary_label_main = dr4 == 1L,
      binary_label_sensitivity = dr4 == 1L,
      direction = NA_character_
    ),
  
  gt_mmc2_clean %>%
    transmute(
      gene, gt_name = "Down_in_ThraAMI_gn_mice", gt_type = "binary",
      score = down_thra, abs_score = down_thra, p_adj = NA_real_,
      binary_label_main = down_thra == 1L,
      binary_label_sensitivity = down_thra == 1L,
      direction = "down"
    ),
  
  gt_mmc2_clean %>%
    transmute(
      gene, gt_name = "Up_in_ThraAMI_gn_mice", gt_type = "binary",
      score = up_thra, abs_score = up_thra, p_adj = NA_real_,
      binary_label_main = up_thra == 1L,
      binary_label_sensitivity = up_thra == 1L,
      direction = "up"
    ),
  
  gt_mmc2_clean %>%
    transmute(
      gene, gt_name = "Down_in_Hypothyroid_mice", gt_type = "binary",
      score = down_hypo, abs_score = down_hypo, p_adj = NA_real_,
      binary_label_main = down_hypo == 1L,
      binary_label_sensitivity = down_hypo == 1L,
      direction = "down"
    ),
  
  gt_mmc2_clean %>%
    transmute(
      gene, gt_name = "Up_in_Hypothyroid_mice", gt_type = "binary",
      score = up_hypo, abs_score = up_hypo, p_adj = NA_real_,
      binary_label_main = up_hypo == 1L,
      binary_label_sensitivity = up_hypo == 1L,
      direction = "up"
    ),
  
  gt_mmc2_clean %>%
    transmute(
      gene, gt_name = "Up_after_TH_treatment", gt_type = "binary",
      score = up_after_th, abs_score = up_after_th, p_adj = NA_real_,
      binary_label_main = up_after_th == 1L,
      binary_label_sensitivity = up_after_th == 1L,
      direction = "up"
    ),
  
  gt_mmc2_clean %>%
    transmute(
      gene, gt_name = "Down_after_TH_treatment", gt_type = "binary",
      score = down_after_th, abs_score = down_after_th, p_adj = NA_real_,
      binary_label_main = down_after_th == 1L,
      binary_label_sensitivity = down_after_th == 1L,
      direction = "down"
    )
)


#######################################
# STEP 5: SUMMARIES
#######################################

gt_mmc2_summary <- gt_mmc2_long %>%
  group_by(.data$gt_name, .data$gt_type) %>%
  summarise(
    n_rows = n(),
    n_unique_genes = n_distinct(.data$gene),
    n_positive_main = sum(.data$binary_label_main, na.rm = TRUE),
    n_positive_sensitivity = sum(.data$binary_label_sensitivity, na.rm = TRUE),
    .groups = "drop"
  )


#######################################
# SAVE OUTPUTS
#######################################

saveRDS(
  gt_thyroid_response_standardized,
  file = here::here("results", "rds", "gt_thyroid_response_standardized.rds")
)

saveRDS(
  gt_mmc2_long,
  file = here::here("results", "rds", "gt_mmc2_long.rds")
)

saveRDS(
  gt_mmc2_definitions,
  file = here::here("results", "rds", "gt_mmc2_definitions.rds")
)

readr::write_csv(
  gt_thyroid_response_summary,
  file = here::here("results", "tables", "gt_thyroid_response_summary.csv")
)

readr::write_csv(
  gt_mmc2_summary,
  file = here::here("results", "tables", "gt_mmc2_summary.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Saved standardized ground-truth objects.")
print(gt_thyroid_response_summary)
print(gt_mmc2_summary)
