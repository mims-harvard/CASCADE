################################################################################
# Script: 06_define_gene_universes.R
#
# Purpose:
#   Define perturbation-strategy-specific CASCADE gene universes and create
#   DE baseline versions restricted to those same universes for fair comparison.
#
# Outputs:
#   - results/rds/cascade_gene_universes.rds
#   - results/rds/de_baselines_restricted_to_cascade_universe.rds
#   - results/tables/cascade_gene_universe_summary.csv
#   - results/tables/de_universe_restriction_audit.csv
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
})


#######################################
# LOAD INPUT OBJECTS
#######################################

cascade_outputs_standardized <- readRDS(
  here::here("results", "rds", "cascade_outputs_standardized.rds")
)

de_baselines_standardized <- readRDS(
  here::here("results", "rds", "de_baselines_standardized.rds")
)


de_to_cascade_mapping <- readRDS(
  here::here("results", "rds", "de_to_cascade_mapping.rds")
)

de_baselines_mapped_to_cascade_space <- de_baselines_standardized %>%
  left_join(
    de_to_cascade_mapping,
    by = c("de_gene_raw" = "de_gene")
  ) %>%
  mutate(
    gene_before_mapping = .data$gene,
    gene = .data$cascade_gene
  )

de_mapping_audit <- de_baselines_mapped_to_cascade_space %>%
  group_by(.data$dataset_name, .data$analysis_task, .data$baseline_type) %>%
  summarise(
    n_rows = n(),
    n_unique_de_gene_raw = n_distinct(.data$de_gene_raw),
    n_mapped_rows = sum(!is.na(.data$cascade_gene)),
    n_unmapped_rows = sum(is.na(.data$cascade_gene)),
    prop_mapped_rows = mean(!is.na(.data$cascade_gene)),
    n_unique_mapped_cascade_gene = n_distinct(.data$cascade_gene[!is.na(.data$cascade_gene)]),
    .groups = "drop"
  )
#######################################
# STEP 1: DEFINE CASCADE GENE UNIVERSES
#######################################

# Strategy-specific universes across all tasks
cascade_gene_universes <- bind_rows(
  
  cascade_outputs_standardized %>%
    filter(.data$perturbation_strategy == "cell_type") %>%
    distinct(gene) %>%
    mutate(
      universe_name = "cascade_universe_cell_type",
      perturbation_strategy = "cell_type",
      analysis_task = "all"
    ),
  
  cascade_outputs_standardized %>%
    filter(.data$perturbation_strategy == "treatment") %>%
    distinct(gene) %>%
    mutate(
      universe_name = "cascade_universe_treatment",
      perturbation_strategy = "treatment",
      analysis_task = "all"
    ),
  
  # Also keep task-specific universes for audit/sensitivity
  cascade_outputs_standardized %>%
    distinct(analysis_task, perturbation_strategy, gene) %>%
    mutate(
      universe_name = paste0(
        "cascade_universe_", perturbation_strategy, "_", analysis_task
      )
    )
) %>%
  distinct(universe_name, perturbation_strategy, analysis_task, gene)


#######################################
# STEP 2: SUMMARIZE CASCADE GENE UNIVERSES
#######################################

cascade_gene_universe_summary <- cascade_gene_universes %>%
  group_by(.data$universe_name, .data$perturbation_strategy, .data$analysis_task) %>%
  summarise(
    n_genes = n_distinct(.data$gene),
    .groups = "drop"
  ) %>%
  arrange(.data$perturbation_strategy, .data$analysis_task, .data$universe_name)


#######################################
# STEP 3: RESTRICT DE BASELINES TO MATCH CASCADE UNIVERSES
#######################################

cell_type_universe <- cascade_gene_universes %>%
  filter(.data$universe_name == "cascade_universe_cell_type") %>%
  pull(gene) %>%
  unique()

treatment_universe <- cascade_gene_universes %>%
  filter(.data$universe_name == "cascade_universe_treatment") %>%
  pull(gene) %>%
  unique()


de_baselines_restricted_to_cascade_universe <- bind_rows(
  
  de_baselines_mapped_to_cascade_space %>%
    filter(!is.na(.data$gene), .data$gene %in% cell_type_universe) %>%
    mutate(comparison_universe = "cell_type"),
  
  de_baselines_mapped_to_cascade_space %>%
    filter(!is.na(.data$gene), .data$gene %in% treatment_universe) %>%
    mutate(comparison_universe = "treatment")
)


#######################################
# STEP 4: AUDIT DE RESTRICTION
#######################################

de_before <- de_baselines_mapped_to_cascade_space %>%
  group_by(.data$dataset_name, .data$analysis_task, .data$baseline_type) %>%
  summarise(
    n_rows_before = n(),
    n_genes_before = n_distinct(.data$gene[!is.na(.data$gene)]),
    .groups = "drop"
  )

de_after <- de_baselines_restricted_to_cascade_universe %>%
  group_by(
    .data$dataset_name,
    .data$analysis_task,
    .data$baseline_type,
    .data$comparison_universe
  ) %>%
  summarise(
    n_rows_after = n(),
    n_genes_after = n_distinct(.data$gene),
    .groups = "drop"
  )

de_universe_restriction_audit <- de_after %>%
  left_join(
    de_before,
    by = c("dataset_name", "analysis_task", "baseline_type")
  ) %>%
  mutate(
    prop_genes_retained = .data$n_genes_after / .data$n_genes_before
  ) %>%
  arrange(.data$analysis_task, .data$baseline_type, .data$comparison_universe)


#######################################
# SAVE OUTPUTS
#######################################

saveRDS(
  de_baselines_mapped_to_cascade_space,
  file = here::here("results", "rds", "de_baselines_mapped_to_cascade_space.rds")
)

readr::write_csv(
  de_mapping_audit,
  file = here::here("results", "tables", "de_mapping_audit.csv")
)

saveRDS(
  cascade_gene_universes,
  file = here::here("results", "rds", "cascade_gene_universes.rds")
)

saveRDS(
  de_baselines_restricted_to_cascade_universe,
  file = here::here("results", "rds", "de_baselines_restricted_to_cascade_universe.rds")
)

readr::write_csv(
  cascade_gene_universe_summary,
  file = here::here("results", "tables", "cascade_gene_universe_summary.csv")
)

readr::write_csv(
  de_universe_restriction_audit,
  file = here::here("results", "tables", "de_universe_restriction_audit.csv")
)


#######################################
# CONSOLE CHECK
#######################################

message("Saved gene universes and DE restriction audit.")
print(cascade_gene_universe_summary)
print(de_universe_restriction_audit)


