################################################################################
# Script: 05c_rescue_mmc2_mapping_biomart.R
#
# Purpose:
#   Rescue unmapped MMC2 gene symbols using biomaRt after the primary
#   org.Mm.eg.db mapping pass.
#
# Inputs:
#   - results/rds/gt_mmc2_long_mapped.rds
#
# Outputs:
#   - results/rds/gt_mmc2_long_mapped_final.rds
#   - results/tables/gt_mmc2_mapping_audit_final.csv
#   - results/tables/gt_mmc2_biomart_rescue_table.csv
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
  library(biomaRt)
})


#######################################
# LOAD INPUT OBJECT
#######################################

gt_mmc2_long_mapped <- readRDS(
  here::here("results", "rds", "gt_mmc2_long_mapped.rds")
)


#######################################
# STEP 1: IDENTIFY UNMAPPED SYMBOLS
#######################################

unmapped_symbols <- gt_mmc2_long_mapped %>%
  distinct(.data$gene_symbol, .data$gene) %>%
  filter(is.na(.data$gene), !is.na(.data$gene_symbol), .data$gene_symbol != "") %>%
  pull(.data$gene_symbol) %>%
  unique() %>%
  sort()

message("Unmapped unique MMC2 symbols to rescue: ", length(unmapped_symbols))


#######################################
# STEP 2: QUERY BIOMART
#######################################

# NOTE:
# We query current mouse Ensembl genes and ask for both the standard external
# gene name and synonyms where possible.

mart <- biomaRt::useEnsembl(
  biomart = "genes",
  dataset = "mmusculus_gene_ensembl"
)

# Main query by official external gene name
biomart_main <- biomaRt::getBM(
  attributes = c("external_gene_name", "ensembl_gene_id"),
  filters = "external_gene_name",
  values = unmapped_symbols,
  mart = mart
) %>%
  as_tibble() %>%
  distinct() %>%
  dplyr::rename(
    gene_symbol = external_gene_name,
    ensembl_gene_id_biomart = ensembl_gene_id
  )

# Optional alias query if supported
biomart_alias <- tryCatch(
  {
    biomaRt::getBM(
      attributes = c("external_synonym", "ensembl_gene_id"),
      filters = "external_synonym",
      values = unmapped_symbols,
      mart = mart
    ) %>%
      as_tibble() %>%
      distinct() %>%
      rename(
        gene_symbol = external_synonym,
        ensembl_gene_id_biomart = ensembl_gene_id
      )
  },
  error = function(e) {
    message("Alias rescue query failed or not supported. Proceeding without alias query.")
    tibble(
      gene_symbol = character(),
      ensembl_gene_id_biomart = character()
    )
  }
)

biomart_combined <- bind_rows(biomart_main, biomart_alias) %>%
  filter(!is.na(.data$gene_symbol), .data$gene_symbol != "",
         !is.na(.data$ensembl_gene_id_biomart), .data$ensembl_gene_id_biomart != "") %>%
  distinct()


#######################################
# STEP 3: RESOLVE AMBIGUOUS BIOMART HITS
#######################################

# Keep only one-to-one symbol -> Ensembl mappings.
# Ambiguous symbols are excluded from automatic rescue.
biomart_rescue_table <- biomart_combined %>%
  group_by(.data$gene_symbol) %>%
  mutate(n_hits = n_distinct(.data$ensembl_gene_id_biomart)) %>%
  ungroup() %>%
  mutate(
    rescue_status = case_when(
      .data$n_hits == 1 ~ "rescued_unique",
      .data$n_hits > 1 ~ "ambiguous_multiple_ensembl",
      TRUE ~ "unresolved"
    )
  )

biomart_rescue_unique <- biomart_rescue_table %>%
  filter(.data$rescue_status == "rescued_unique") %>%
  distinct(.data$gene_symbol, .data$ensembl_gene_id_biomart, .keep_all = TRUE)


#######################################
# STEP 4: MERGE RESCUE INTO MMC2 OBJECT
#######################################

gt_mmc2_long_mapped_final <- gt_mmc2_long_mapped %>%
  left_join(
    biomart_rescue_unique %>%
      dplyr::select(.data$gene_symbol, .data$ensembl_gene_id_biomart),
    by = "gene_symbol"
  ) %>%
  mutate(
    gene_pre_rescue = .data$gene,
    gene = ifelse(is.na(.data$gene), .data$ensembl_gene_id_biomart, .data$gene),
    mapping_source = case_when(
      !is.na(.data$gene_pre_rescue) ~ "org.Mm.eg.db",
      is.na(.data$gene_pre_rescue) & !is.na(.data$ensembl_gene_id_biomart) ~ "biomaRt",
      TRUE ~ "unmapped"
    )
  )


#######################################
# STEP 5: FINAL AUDIT
#######################################

gt_mmc2_mapping_audit_final <- gt_mmc2_long_mapped_final %>%
  group_by(.data$gt_name, .data$gt_type) %>%
  summarise(
    n_rows = n(),
    n_unique_symbols = n_distinct(.data$gene_symbol),
    n_mapped_rows = sum(!is.na(.data$gene)),
    n_unmapped_rows = sum(is.na(.data$gene)),
    prop_mapped_rows = mean(!is.na(.data$gene)),
    n_unique_ensembl = n_distinct(.data$gene[!is.na(.data$gene)]),
    n_rows_orgdb = sum(.data$mapping_source == "org.Mm.eg.db"),
    n_rows_biomart = sum(.data$mapping_source == "biomaRt"),
    n_rows_still_unmapped = sum(.data$mapping_source == "unmapped"),
    .groups = "drop"
  )


#######################################
# SAVE OUTPUTS
#######################################

saveRDS(
  gt_mmc2_long_mapped_final,
  file = here::here("results", "rds", "gt_mmc2_long_mapped_final.rds")
)

readr::write_csv(
  gt_mmc2_mapping_audit_final,
  file = here::here("results", "tables", "gt_mmc2_mapping_audit_final.csv")
)

readr::write_csv(
  biomart_rescue_table,
  file = here::here("results", "tables", "gt_mmc2_biomart_rescue_table.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Saved final MMC2 mapping with biomaRt rescue.")
print(gt_mmc2_mapping_audit_final)

message("biomaRt rescue status summary:")
print(
  biomart_rescue_table %>%
    distinct(.data$gene_symbol, .data$rescue_status) %>%
    count(.data$rescue_status)
)
