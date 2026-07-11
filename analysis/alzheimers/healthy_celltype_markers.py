#!/usr/bin/env python3
"""
Healthy-only grouped cell-type marker analysis from chunked AnnData objects.

This script processes multiple .h5ad chunks without concatenating them into memory.
It keeps only healthy/control cells, maps fine-grained cell types to broader DEG
groups, then computes one-vs-rest marker genes for each grouped cell type.

For each grouped cell type, it compares:

    healthy cells of this grouped cell type
    vs
    healthy cells of all other grouped cell types

Example
-------
python -m analysis.alzheimers.healthy_celltype_markers \
    --h5ad-dir $CASCADE_DATA_ROOT/SEATTLE/adata_objects \
    --output-dir $CASCADE_DATA_ROOT/SEATTLE/healthy_grouped_cell_type_markers \
    --cell-type-col cell_type \
    --health-col disease \
    --healthy-values 0 \
    --gene-col ensembl_gene_id \
    --h5ad-pattern "adata_annotated_protein_clinical_fin_*.h5ad"
"""

from __future__ import annotations

import argparse
import gc
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


DEG_CT_MAP = {
    # Non-neuronal — 1:1
    "microglial cell": "Microglia",
    "astrocyte of the cerebral cortex": "Astrocytes",
    "oligodendrocyte": "Oligodendrocytes",
    "oligodendrocyte precursor cell": "OPC",
    "cerebral cortex endothelial cell": "Endothelial",
    "vascular leptomeningeal cell": "Endothelial",

    # Excitatory neurons → IO
    "L2/3-6 intratelencephalic projecting glutamatergic neuron": "IO",
    "L5 extratelencephalic projecting glutamatergic cortical neuron": "IO",
    "L6b glutamatergic cortical neuron": "IO",
    "corticothalamic-projecting glutamatergic cortical neuron": "IO",
    "near-projecting glutamatergic cortical neuron": "IO",

    # Inhibitory neurons → IO
    "VIP GABAergic cortical interneuron": "IO",
    "caudal ganglionic eminence derived interneuron": "IO",
    "chandelier pvalb GABAergic cortical interneuron": "IO",
    "lamp5 GABAergic cortical interneuron": "IO",
    "pvalb GABAergic cortical interneuron": "IO",
    "sncg GABAergic cortical interneuron": "IO",
    "sst GABAergic cortical interneuron": "IO",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute healthy-only one-vs-rest marker genes from chunked AnnData "
            "objects after mapping fine-grained cell types to broader DEG groups."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--h5ad-dir",
        type=Path,
        required=True,
        help="Directory containing chunked .h5ad files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where marker results will be saved.",
    )

    parser.add_argument(
        "--h5ad-pattern",
        type=str,
        default="*.h5ad",
        help="Filename pattern used to find AnnData chunks.",
    )

    parser.add_argument(
        "--cell-type-col",
        type=str,
        default="cell_type",
        help="Column in adata.obs containing fine-grained cell type labels.",
    )

    parser.add_argument(
        "--health-col",
        type=str,
        required=True,
        help=(
            "Column in adata.obs used to identify healthy/control cells, "
            "for example disease, condition, diagnosis, or disease_status."
        ),
    )

    parser.add_argument(
        "--healthy-values",
        nargs="+",
        required=True,
        help=(
            "Values in --health-col that should be treated as healthy/control. "
            "Examples: 0 healthy control normal."
        ),
    )

    parser.add_argument(
        "--gene-col",
        type=str,
        default=None,
        help=(
            "Column in adata.var containing gene names/IDs. "
            "For your SEATTLE objects, use: ensembl_gene_id. "
            "If omitted, adata.var_names is used."
        ),
    )

    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help=(
            "Optional AnnData layer to use as expression matrix. "
            "If omitted, adata.X is used."
        ),
    )

    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Use adata.raw.X instead of adata.X. Cannot be combined with --layer.",
    )

    parser.add_argument(
        "--keep-unmapped-cell-types",
        action="store_true",
        help=(
            "Keep cell types that are not present in DEG_CT_MAP under their original "
            "labels. By default, unmapped cell types are dropped."
        ),
    )

    parser.add_argument(
        "--min-cells-per-cell-type",
        type=int,
        default=10,
        help="Minimum number of healthy cells required for a grouped cell type.",
    )

    parser.add_argument(
        "--min-cells-rest",
        type=int,
        default=10,
        help="Minimum number of healthy rest cells required for one-vs-rest comparison.",
    )

    parser.add_argument(
        "--pseudocount",
        type=float,
        default=1e-9,
        help="Small value added before log2 fold-change calculation.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help=(
            "Optional number of top genes to save per grouped cell type. "
            "If omitted, all genes are saved."
        ),
    )

    return parser.parse_args()


def normalise_label(value: object) -> str:
    """Convert metadata labels to stable strings."""

    if pd.isna(value):
        return "nan"

    return str(value).strip()


def safe_filename(value: str) -> str:
    """Convert a cell type label into a safe filename component."""

    value = str(value).strip()
    value = re.sub(r"[^\w\-.]+", "_", value)
    value = value.strip("_")

    if value == "":
        value = "empty_label"

    return value


def map_cell_types_for_deg(
    cell_types: pd.Series,
    keep_unmapped: bool,
) -> pd.Series:
    """
    Map fine-grained cell-type labels to broader DEG groups.

    Args:
        cell_types:
            Original cell-type labels from adata.obs.
        keep_unmapped:
            Whether to keep unmapped cell types under their original labels.

    Returns:
        Series of mapped grouped cell-type labels. Unmapped labels become NA if
        keep_unmapped is False.
    """

    cell_types = cell_types.map(normalise_label)
    mapped = cell_types.map(DEG_CT_MAP)

    if keep_unmapped:
        mapped = mapped.fillna(cell_types)

    return mapped


def get_gene_names_from_var(
    var: pd.DataFrame,
    var_names: pd.Index,
    gene_col: Optional[str],
) -> np.ndarray:
    """
    Return gene names from adata.var[gene_col] or adata.var_names.

    Args:
        var:
            adata.var or adata.raw.var.
        var_names:
            adata.var_names or adata.raw.var_names.
        gene_col:
            Optional column in var containing gene IDs/names.

    Returns:
        Gene names as a string numpy array.
    """

    if gene_col is None:
        return var_names.astype(str).to_numpy()

    if gene_col not in var.columns:
        raise KeyError(
            f"{gene_col!r} not found in adata.var. "
            f"Available columns are: {list(var.columns)}"
        )

    gene_names = var[gene_col].astype(str).to_numpy()

    if pd.isna(gene_names).any():
        raise ValueError(f"Column {gene_col!r} contains missing gene identifiers.")

    duplicated = pd.Series(gene_names).duplicated().sum()
    if duplicated > 0:
        print(
            f"WARNING: gene column {gene_col!r} contains {duplicated} duplicated values. "
            "The script will still run, but duplicated gene IDs may appear in the output."
        )

    return gene_names


def get_expression_matrix(
    adata: ad.AnnData,
    layer: Optional[str],
    use_raw: bool,
    gene_col: Optional[str],
):
    """
    Return expression matrix and gene names.

    Args:
        adata:
            AnnData object.
        layer:
            Optional layer name.
        use_raw:
            Whether to use adata.raw.X.
        gene_col:
            Optional column in adata.var containing gene IDs/names.

    Returns:
        Tuple of expression matrix and gene names.
    """

    if use_raw and layer is not None:
        raise ValueError("Cannot use both --use-raw and --layer.")

    if use_raw:
        if adata.raw is None:
            raise ValueError("Requested --use-raw, but adata.raw is None.")

        gene_names = get_gene_names_from_var(
            var=adata.raw.var,
            var_names=adata.raw.var_names,
            gene_col=gene_col,
        )

        return adata.raw.X, gene_names

    gene_names = get_gene_names_from_var(
        var=adata.var,
        var_names=adata.var_names,
        gene_col=gene_col,
    )

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(
                f"Layer {layer!r} not found. "
                f"Available layers are: {list(adata.layers.keys())}"
            )

        return adata.layers[layer], gene_names

    return adata.X, gene_names


def to_1d_array(x) -> np.ndarray:
    """Convert sparse/dense matrix result to a flat numpy array."""

    return np.asarray(x).ravel()


def compute_sum(X) -> np.ndarray:
    """Compute column-wise sums."""

    return to_1d_array(X.sum(axis=0))


def compute_sumsq(X) -> np.ndarray:
    """Compute column-wise sums of squares."""

    if sp.issparse(X):
        return to_1d_array(X.power(2).sum(axis=0))

    return to_1d_array(np.square(X).sum(axis=0))


def compute_detected(X) -> np.ndarray:
    """Compute number of cells with expression > 0 for each gene."""

    return to_1d_array((X > 0).sum(axis=0))


def validate_gene_names(
    current_gene_names: np.ndarray,
    reference_gene_names: np.ndarray,
    h5ad_file: Path,
) -> None:
    """Ensure all chunks use the same genes in the same order."""

    if len(current_gene_names) != len(reference_gene_names):
        raise ValueError(
            f"Gene number mismatch in {h5ad_file}. "
            f"Expected {len(reference_gene_names)}, found {len(current_gene_names)}."
        )

    if not np.array_equal(current_gene_names, reference_gene_names):
        raise ValueError(
            f"Gene order mismatch in {h5ad_file}. "
            "All chunks must have the same genes in the same order."
        )


def get_healthy_mask(
    obs: pd.DataFrame,
    health_col: str,
    healthy_values: set[str],
) -> np.ndarray:
    """Return boolean mask for healthy/control cells."""

    if health_col not in obs.columns:
        raise KeyError(
            f"{health_col!r} not found in adata.obs. "
            f"Available obs columns include: {list(obs.columns)}"
        )

    labels = obs[health_col].map(normalise_label)
    return labels.isin(healthy_values).to_numpy()


def scan_cell_types_and_genes(
    h5ad_files: List[Path],
    cell_type_col: str,
    health_col: str,
    healthy_values: set[str],
    layer: Optional[str],
    use_raw: bool,
    gene_col: Optional[str],
    keep_unmapped_cell_types: bool,
) -> Tuple[np.ndarray, List[str]]:
    """
    First pass over chunks to identify gene names and healthy grouped cell types.
    """

    reference_gene_names: Optional[np.ndarray] = None
    all_grouped_cell_types: set[str] = set()
    raw_cell_type_counter: defaultdict[str, int] = defaultdict(int)
    mapped_cell_type_counter: defaultdict[str, int] = defaultdict(int)
    unmapped_cell_type_counter: defaultdict[str, int] = defaultdict(int)

    for h5ad_file in tqdm(h5ad_files, desc="Scanning chunks"):
        adata = ad.read_h5ad(h5ad_file)

        if cell_type_col not in adata.obs.columns:
            raise KeyError(
                f"{cell_type_col!r} not found in {h5ad_file}. "
                f"Available obs columns include: {list(adata.obs.columns)}"
            )

        X, gene_names = get_expression_matrix(
            adata=adata,
            layer=layer,
            use_raw=use_raw,
            gene_col=gene_col,
        )

        if reference_gene_names is None:
            reference_gene_names = gene_names
        else:
            validate_gene_names(
                current_gene_names=gene_names,
                reference_gene_names=reference_gene_names,
                h5ad_file=h5ad_file,
            )

        healthy_mask = get_healthy_mask(
            obs=adata.obs,
            health_col=health_col,
            healthy_values=healthy_values,
        )

        if healthy_mask.sum() > 0:
            healthy_obs = adata.obs.loc[healthy_mask]

            raw_cell_types = healthy_obs[cell_type_col].map(normalise_label)
            mapped_cell_types = map_cell_types_for_deg(
                cell_types=healthy_obs[cell_type_col],
                keep_unmapped=keep_unmapped_cell_types,
            )

            for raw_ct, count in raw_cell_types.value_counts(dropna=False).items():
                raw_cell_type_counter[str(raw_ct)] += int(count)

            for mapped_ct, count in mapped_cell_types.value_counts(dropna=False).items():
                if pd.isna(mapped_ct):
                    continue
                mapped_cell_type_counter[str(mapped_ct)] += int(count)

            unmapped_mask = raw_cell_types.map(lambda x: x not in DEG_CT_MAP)
            for raw_ct, count in raw_cell_types.loc[unmapped_mask].value_counts().items():
                unmapped_cell_type_counter[str(raw_ct)] += int(count)

            mapped_cell_types = mapped_cell_types.dropna()
            all_grouped_cell_types.update(mapped_cell_types.unique())

        del adata, X
        gc.collect()

    if reference_gene_names is None:
        raise RuntimeError("Could not determine gene names.")

    if len(all_grouped_cell_types) == 0:
        raise RuntimeError(
            "No healthy mapped cell types found. "
            "Check --cell-type-col, --health-col, --healthy-values, and DEG_CT_MAP."
        )

    print("\nRaw healthy cell types observed:")
    for ct, count in sorted(raw_cell_type_counter.items(), key=lambda x: x[0]):
        print(f"  {ct}: {count}")

    print("\nGrouped healthy cell types that will be analysed:")
    for ct, count in sorted(mapped_cell_type_counter.items(), key=lambda x: x[0]):
        print(f"  {ct}: {count}")

    if len(unmapped_cell_type_counter) > 0 and not keep_unmapped_cell_types:
        print("\nUnmapped healthy cell types that will be dropped:")
        for ct, count in sorted(unmapped_cell_type_counter.items(), key=lambda x: x[0]):
            print(f"  {ct}: {count}")

    if len(unmapped_cell_type_counter) > 0 and keep_unmapped_cell_types:
        print("\nUnmapped healthy cell types kept under original labels:")
        for ct, count in sorted(unmapped_cell_type_counter.items(), key=lambda x: x[0]):
            print(f"  {ct}: {count}")

    return reference_gene_names, sorted(all_grouped_cell_types)


def initialise_statistics(
    cell_types: Iterable[str],
    n_genes: int,
):
    """Initialise global and cell-type-specific streaming statistics."""

    total_sum = np.zeros(n_genes, dtype=np.float64)
    total_sumsq = np.zeros(n_genes, dtype=np.float64)
    total_detected = np.zeros(n_genes, dtype=np.float64)
    total_n = 0

    ct_sum: Dict[str, np.ndarray] = {
        ct: np.zeros(n_genes, dtype=np.float64) for ct in cell_types
    }

    ct_sumsq: Dict[str, np.ndarray] = {
        ct: np.zeros(n_genes, dtype=np.float64) for ct in cell_types
    }

    ct_detected: Dict[str, np.ndarray] = {
        ct: np.zeros(n_genes, dtype=np.float64) for ct in cell_types
    }

    ct_n: Dict[str, int] = defaultdict(int)

    return (
        total_sum,
        total_sumsq,
        total_detected,
        total_n,
        ct_sum,
        ct_sumsq,
        ct_detected,
        ct_n,
    )


def accumulate_statistics(
    h5ad_files: List[Path],
    reference_gene_names: np.ndarray,
    cell_types: List[str],
    cell_type_col: str,
    health_col: str,
    healthy_values: set[str],
    layer: Optional[str],
    use_raw: bool,
    gene_col: Optional[str],
    keep_unmapped_cell_types: bool,
):
    """
    Second pass over chunks to accumulate healthy-only grouped-cell-type statistics.
    """

    n_genes = len(reference_gene_names)

    (
        total_sum,
        total_sumsq,
        total_detected,
        total_n,
        ct_sum,
        ct_sumsq,
        ct_detected,
        ct_n,
    ) = initialise_statistics(cell_types=cell_types, n_genes=n_genes)

    for h5ad_file in tqdm(h5ad_files, desc="Accumulating healthy-cell statistics"):
        adata = ad.read_h5ad(h5ad_file)

        X, gene_names = get_expression_matrix(
            adata=adata,
            layer=layer,
            use_raw=use_raw,
            gene_col=gene_col,
        )

        validate_gene_names(
            current_gene_names=gene_names,
            reference_gene_names=reference_gene_names,
            h5ad_file=h5ad_file,
        )

        healthy_mask = get_healthy_mask(
            obs=adata.obs,
            health_col=health_col,
            healthy_values=healthy_values,
        )

        if healthy_mask.sum() == 0:
            del adata, X
            gc.collect()
            continue

        X_healthy = X[healthy_mask, :]

        if sp.issparse(X_healthy):
            X_healthy = X_healthy.tocsr()

        grouped_cell_types_series = map_cell_types_for_deg(
            cell_types=adata.obs.loc[healthy_mask, cell_type_col],
            keep_unmapped=keep_unmapped_cell_types,
        )

        valid_mapped_mask = grouped_cell_types_series.notna().to_numpy()

        if valid_mapped_mask.sum() == 0:
            del adata, X, X_healthy
            gc.collect()
            continue

        X_healthy = X_healthy[valid_mapped_mask, :]
        grouped_cell_types = grouped_cell_types_series.loc[
            grouped_cell_types_series.notna()
        ].to_numpy()

        total_sum += compute_sum(X_healthy)
        total_sumsq += compute_sumsq(X_healthy)
        total_detected += compute_detected(X_healthy)
        total_n += X_healthy.shape[0]

        for ct in np.unique(grouped_cell_types):
            idx = np.where(grouped_cell_types == ct)[0]

            if len(idx) == 0:
                continue

            X_ct = X_healthy[idx, :]

            ct_sum[ct] += compute_sum(X_ct)
            ct_sumsq[ct] += compute_sumsq(X_ct)
            ct_detected[ct] += compute_detected(X_ct)
            ct_n[ct] += X_ct.shape[0]

        del adata, X, X_healthy
        gc.collect()

    return (
        total_sum,
        total_sumsq,
        total_detected,
        total_n,
        ct_sum,
        ct_sumsq,
        ct_detected,
        ct_n,
    )


def compute_one_vs_rest_markers(
    cell_type: str,
    gene_names: np.ndarray,
    total_sum: np.ndarray,
    total_sumsq: np.ndarray,
    total_detected: np.ndarray,
    total_n: int,
    ct_sum: Dict[str, np.ndarray],
    ct_sumsq: Dict[str, np.ndarray],
    ct_detected: Dict[str, np.ndarray],
    ct_n: Dict[str, int],
    min_cells_per_cell_type: int,
    min_cells_rest: int,
    pseudocount: float,
) -> Optional[pd.DataFrame]:
    """
    Compute one-vs-rest marker statistics for one grouped cell type.
    """

    n1 = ct_n[cell_type]
    n2 = total_n - n1

    if n1 < min_cells_per_cell_type:
        print(
            f"Skipping {cell_type!r}: only {n1} healthy cells "
            f"(< {min_cells_per_cell_type})."
        )
        return None

    if n2 < min_cells_rest:
        print(
            f"Skipping {cell_type!r}: only {n2} rest cells "
            f"(< {min_cells_rest})."
        )
        return None

    sum1 = ct_sum[cell_type]
    sum2 = total_sum - sum1

    sumsq1 = ct_sumsq[cell_type]
    sumsq2 = total_sumsq - sumsq1

    detected1 = ct_detected[cell_type]
    detected2 = total_detected - detected1

    mean1 = sum1 / n1
    mean2 = sum2 / n2

    var1 = (sumsq1 - ((sum1 ** 2) / n1)) / max(n1 - 1, 1)
    var2 = (sumsq2 - ((sum2 ** 2) / n2)) / max(n2 - 1, 1)

    var1 = np.maximum(var1, 0)
    var2 = np.maximum(var2, 0)

    se = np.sqrt((var1 / n1) + (var2 / n2))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = (mean1 - mean2) / se

    numerator = ((var1 / n1) + (var2 / n2)) ** 2
    denominator = ((var1 / n1) ** 2 / max(n1 - 1, 1)) + (
        (var2 / n2) ** 2 / max(n2 - 1, 1)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        degrees_freedom = numerator / denominator

    with np.errstate(divide="ignore", invalid="ignore"):
        pvals = 2 * student_t.sf(np.abs(t_stat), degrees_freedom)

    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    t_stat = np.where(np.isfinite(t_stat), t_stat, 0.0)
    degrees_freedom = np.where(np.isfinite(degrees_freedom), degrees_freedom, 0.0)

    padj = multipletests(pvals, method="fdr_bh")[1]

    log2fc = np.log2((mean1 + pseudocount) / (mean2 + pseudocount))

    pct_expr_cell_type = detected1 / n1
    pct_expr_rest = detected2 / n2
    pct_expr_difference = pct_expr_cell_type - pct_expr_rest

    result = pd.DataFrame(
        {
            "grouped_cell_type": cell_type,
            "gene": gene_names,
            "n_grouped_cell_type": n1,
            "n_rest": n2,
            "mean_grouped_cell_type": mean1,
            "mean_rest": mean2,
            "log2FC": log2fc,
            "t_stat": t_stat,
            "degrees_freedom": degrees_freedom,
            "pval": pvals,
            "padj": padj,
            "pct_expr_grouped_cell_type": pct_expr_cell_type,
            "pct_expr_rest": pct_expr_rest,
            "pct_expr_difference": pct_expr_difference,
        }
    )

    result = result.sort_values(
        by=["padj", "log2FC", "pct_expr_difference"],
        ascending=[True, False, False],
    )

    return result


def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    h5ad_files: List[Path],
    cell_types: List[str],
    total_n: int,
    ct_n: Dict[str, int],
) -> None:
    """Save run metadata, grouped cell counts, and the mapping used."""

    metadata = {
        "h5ad_dir": str(args.h5ad_dir),
        "h5ad_pattern": args.h5ad_pattern,
        "n_h5ad_files": len(h5ad_files),
        "cell_type_col": args.cell_type_col,
        "health_col": args.health_col,
        "healthy_values": " ".join(args.healthy_values),
        "gene_col": args.gene_col,
        "layer": args.layer,
        "use_raw": args.use_raw,
        "keep_unmapped_cell_types": args.keep_unmapped_cell_types,
        "total_healthy_mapped_cells": total_n,
        "n_grouped_cell_types": len(cell_types),
        "min_cells_per_cell_type": args.min_cells_per_cell_type,
        "min_cells_rest": args.min_cells_rest,
        "pseudocount": args.pseudocount,
        "top_n": args.top_n,
    }

    pd.Series(metadata).to_csv(output_dir / "run_metadata.csv", header=False)

    counts = pd.DataFrame(
        {
            "grouped_cell_type": list(ct_n.keys()),
            "n_healthy_cells": list(ct_n.values()),
        }
    ).sort_values("n_healthy_cells", ascending=False)

    counts.to_csv(output_dir / "healthy_cell_counts_by_grouped_cell_type.csv", index=False)

    mapping = pd.DataFrame(
        {
            "original_cell_type": list(DEG_CT_MAP.keys()),
            "grouped_cell_type": list(DEG_CT_MAP.values()),
        }
    ).sort_values(["grouped_cell_type", "original_cell_type"])

    mapping.to_csv(output_dir / "deg_cell_type_mapping_used.csv", index=False)


def main() -> None:
    """Run healthy-only grouped cell-type marker gene analysis."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_raw and args.layer is not None:
        raise ValueError("Please use either --use-raw or --layer, not both.")

    h5ad_files = sorted(args.h5ad_dir.glob(args.h5ad_pattern))

    if len(h5ad_files) == 0:
        raise FileNotFoundError(
            f"No AnnData files matching {args.h5ad_pattern!r} "
            f"found in {args.h5ad_dir}."
        )

    healthy_values = {normalise_label(x) for x in args.healthy_values}

    print(f"Found {len(h5ad_files)} AnnData chunks.")
    print(f"Keeping healthy/control values from {args.health_col!r}: {healthy_values}")
    print("Applying DEG_CT_MAP to group fine-grained cell types.")

    if args.keep_unmapped_cell_types:
        print("Unmapped cell types will be kept under their original labels.")
    else:
        print("Unmapped cell types will be dropped.")

    if args.gene_col is not None:
        print(f"Using gene IDs from adata.var[{args.gene_col!r}].")
    else:
        print("Using adata.var_names as gene IDs.")

    gene_names, cell_types = scan_cell_types_and_genes(
        h5ad_files=h5ad_files,
        cell_type_col=args.cell_type_col,
        health_col=args.health_col,
        healthy_values=healthy_values,
        layer=args.layer,
        use_raw=args.use_raw,
        gene_col=args.gene_col,
        keep_unmapped_cell_types=args.keep_unmapped_cell_types,
    )

    print(f"\nFound {len(gene_names)} genes.")
    print(f"Found {len(cell_types)} grouped healthy cell types:")
    for ct in cell_types:
        print(f"  - {ct}")

    (
        total_sum,
        total_sumsq,
        total_detected,
        total_n,
        ct_sum,
        ct_sumsq,
        ct_detected,
        ct_n,
    ) = accumulate_statistics(
        h5ad_files=h5ad_files,
        reference_gene_names=gene_names,
        cell_types=cell_types,
        cell_type_col=args.cell_type_col,
        health_col=args.health_col,
        healthy_values=healthy_values,
        layer=args.layer,
        use_raw=args.use_raw,
        gene_col=args.gene_col,
        keep_unmapped_cell_types=args.keep_unmapped_cell_types,
    )

    if total_n == 0:
        raise RuntimeError(
            "No healthy mapped cells were retained after filtering. "
            "Check --health-col, --healthy-values, --cell-type-col, and DEG_CT_MAP."
        )

    print(f"\nTotal healthy mapped cells retained: {total_n:,}")

    save_metadata(
        output_dir=args.output_dir,
        args=args,
        h5ad_files=h5ad_files,
        cell_types=cell_types,
        total_n=total_n,
        ct_n=ct_n,
    )

    all_results: List[pd.DataFrame] = []

    for cell_type in tqdm(cell_types, desc="Computing one-vs-rest markers"):
        result = compute_one_vs_rest_markers(
            cell_type=cell_type,
            gene_names=gene_names,
            total_sum=total_sum,
            total_sumsq=total_sumsq,
            total_detected=total_detected,
            total_n=total_n,
            ct_sum=ct_sum,
            ct_sumsq=ct_sumsq,
            ct_detected=ct_detected,
            ct_n=ct_n,
            min_cells_per_cell_type=args.min_cells_per_cell_type,
            min_cells_rest=args.min_cells_rest,
            pseudocount=args.pseudocount,
        )

        if result is None:
            continue

        if args.top_n is not None:
            result_to_save = result.head(args.top_n).copy()
        else:
            result_to_save = result

        output_name = f"markers_healthy_only_{safe_filename(cell_type)}.csv"
        result_to_save.to_csv(args.output_dir / output_name, index=False)

        all_results.append(result_to_save)

    if len(all_results) == 0:
        raise RuntimeError("No marker results were produced.")

    combined = pd.concat(all_results, axis=0, ignore_index=True)
    combined.to_csv(
        args.output_dir / "markers_healthy_only_all_grouped_cell_types.csv",
        index=False,
    )

    print("\nDone.")
    print(f"Results saved to: {args.output_dir}")
    print(
        "Main result file: "
        f"{args.output_dir / 'markers_healthy_only_all_grouped_cell_types.csv'}"
    )


if __name__ == "__main__":
    main()