################################################################################
# Script: 05b_map_mmc2_to_ensembl.R
#
# Purpose:
#   Map MMC2 gene symbols to mouse Ensembl gene IDs so that MMC2 can be joined
#   to CASCADE outputs in a shared identifier space.
#
# Outputs:
#   - results/rds/mmc2_symbol_map.rds
#   - results/rds/gt_mmc2_long_mapped.rds
#   - results/tables/gt_mmc2_mapping_audit.csv
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
  library(AnnotationDbi)
  library(org.Mm.eg.db)
})


#######################################
# LOAD MMC2 OBJECT
#######################################

gt_mmc2_long <- readRDS(
  here::here("results", "rds", "gt_mmc2_long.rds")
)


#######################################
# BUILD SYMBOL -> ENSEMBL MAP
#######################################

mmc2_symbol_map <- tibble(
  gene_symbol = sort(unique(gt_mmc2_long$gene))
) %>%
  mutate(
    ensembl_gene_id = AnnotationDbi::mapIds(
      x = org.Mm.eg.db,
      keys = .data$gene_symbol,
      keytype = "SYMBOL",
      column = "ENSEMBL",
      multiVals = "first"
    ) %>% as.character()
  )


#######################################
# MAP MMC2 INTO ENSEMBL SPACE
#######################################

gt_mmc2_long_mapped <- gt_mmc2_long %>%
  dplyr::rename(gene_symbol = gene) %>%
  dplyr::left_join(mmc2_symbol_map, by = "gene_symbol") %>%
  dplyr::mutate(
    gene = .data$ensembl_gene_id
  )


#######################################
# AUDIT
#######################################

gt_mmc2_mapping_audit <- gt_mmc2_long_mapped %>%
  group_by(.data$gt_name, .data$gt_type) %>%
  summarise(
    n_rows = n(),
    n_unique_symbols = n_distinct(.data$gene_symbol),
    n_mapped_rows = sum(!is.na(.data$gene)),
    n_unmapped_rows = sum(is.na(.data$gene)),
    prop_mapped_rows = mean(!is.na(.data$gene)),
    n_unique_ensembl = n_distinct(.data$gene[!is.na(.data$gene)]),
    .groups = "drop"
  )


#######################################
# SAVE
#######################################

# NOTE: mmc2_symbol_map is saved here (in addition to the mapped long-form
# object below) because 21_save_borda_rankings_to_excel.R needs the raw
# symbol->Ensembl lookup for annotating gene rankings with gene symbols.
saveRDS(
  mmc2_symbol_map,
  file = here::here("results", "rds", "mmc2_symbol_map.rds")
)

saveRDS(
  gt_mmc2_long_mapped,
  file = here::here("results", "rds", "gt_mmc2_long_mapped.rds")
)

readr::write_csv(
  gt_mmc2_mapping_audit,
  file = here::here("results", "tables", "gt_mmc2_mapping_audit.csv")
)

message("Saved mapped MMC2 object.")
print(gt_mmc2_mapping_audit)
