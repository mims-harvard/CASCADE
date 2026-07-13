#!/usr/bin/env python3
"""
Differential expression baseline for the M2 (mouse-thyroid) treatment and DN-THRα
prediction tasks: "findings derived by CASCADE-Explainer were compared with a
matched differential expression (DE) baseline for both the treatment and DN-THRα
prediction tasks" (Methods, thyroid hormone / receptor-signalling section). Produces
the per-task `de_pooled_*.csv` / `de_cell_type_specific_*.csv` files consumed by
`04_standardize_de_baselines.R`.

Task filters mirror the corresponding CASCADE prediction tasks:
- treatment task:
    keep (Cre is None) OR (Cre == 0 AND THR == 1)
    DE target: treatment (class 1 vs class 0)
- THR task:
    keep treatment == 1
    then keep Cre > 0
    then (optionally) keep neuronal cell types only
    DE target: THR (class 1 vs class 0)

For each task, the script computes:
1) pooled (cell-type-unspecific) DE
2) cell-type-specific DE (within each cell type, when both classes are present)

Usage:
    python -m analysis.thyroid_hormone.scripts.03_m2_differential_expression \
        --h5ad-path $CASCADE_DATA_ROOT/M2/adata_objects/adata_annotated_protein_coding_clinical.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from cascade.explainer.config import CASCADE_DATA_ROOT

DEFAULT_H5AD = CASCADE_DATA_ROOT / "M2/adata_objects/adata_annotated_protein_coding_clinical.h5ad"

NON_NEURONAL_TYPES = {
    "astrocyte",
    "vascular leptomeningeal cell (VLMC)",
    "perivascular macrophage",
    "oligodendrocyte",
    "oligodendrocyte precursor cell (OPC)",
    "endothelial cell",
    "microglial cell",
    "pericyte",
}


def _is_one(v) -> bool:
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    try:
        return float(v) == 1.0
    except (TypeError, ValueError):
        return False


def _cre_is_none(v) -> bool:
    return v is None


def _cre_is_zero(v) -> bool:
    try:
        return float(v) == 0.0
    except (TypeError, ValueError):
        return False


def _cre_is_positive(v) -> bool:
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    if isinstance(v, str) and v.strip() == "":
        return False
    try:
        return float(v) > 0.0
    except (TypeError, ValueError):
        return False


def _make_binary_labels(values: Iterable) -> np.ndarray:
    """Map values to {0,1} when possible; everything else -> NaN."""
    out = np.full(len(values), np.nan, dtype=float)
    vals = np.asarray(values, dtype=object)
    for i, v in enumerate(vals):
        if _is_one(v):
            out[i] = 1.0
            continue
        try:
            fv = float(v)
            if fv == 0.0:
                out[i] = 0.0
        except Exception:
            pass
    return out


def _prepare_treatment_task(adata: sc.AnnData) -> sc.AnnData:
    if "Cre" not in adata.obs or "THR" not in adata.obs or "treatment" not in adata.obs:
        raise KeyError("treatment task requires obs keys: Cre, THR, treatment")

    cre_vals = adata.obs["Cre"].to_numpy(dtype=object)
    thr_vals = adata.obs["THR"].to_numpy(dtype=object)

    cre_none = np.array([_cre_is_none(v) for v in cre_vals], dtype=bool)
    cre_zero = np.array([_cre_is_zero(v) for v in cre_vals], dtype=bool)
    thr_one = np.array([_is_one(v) for v in thr_vals], dtype=bool)
    keep = cre_none | (cre_zero & thr_one)

    out = adata[keep].copy()
    labels = _make_binary_labels(out.obs["treatment"].to_numpy(dtype=object))
    out.obs["de_label"] = labels
    out = out[np.isfinite(out.obs["de_label"].to_numpy())].copy()
    out.obs["de_label"] = out.obs["de_label"].astype(int).astype(str)
    return out


def _prepare_thr_task(adata: sc.AnnData, exclude_non_neuronal: bool, cell_type_key: str) -> sc.AnnData:
    if "treatment" not in adata.obs or "Cre" not in adata.obs or "THR" not in adata.obs:
        raise KeyError("THR task requires obs keys: treatment, Cre, THR")

    treatment_vals = adata.obs["treatment"].to_numpy(dtype=object)
    keep_treatment = np.array([_is_one(v) for v in treatment_vals], dtype=bool)
    out = adata[keep_treatment].copy()

    cre_vals = out.obs["Cre"].to_numpy(dtype=object)
    keep_cre = np.array([_cre_is_positive(v) for v in cre_vals], dtype=bool)
    out = out[keep_cre].copy()

    if exclude_non_neuronal and cell_type_key in out.obs:
        ct = out.obs[cell_type_key].to_numpy(dtype=object)
        keep_neuron = ~np.isin(ct, list(NON_NEURONAL_TYPES))
        out = out[keep_neuron].copy()

    labels = _make_binary_labels(out.obs["THR"].to_numpy(dtype=object))
    out.obs["de_label"] = labels
    out = out[np.isfinite(out.obs["de_label"].to_numpy())].copy()
    out.obs["de_label"] = out.obs["de_label"].astype(int).astype(str)
    return out


def _run_rank_genes_groups(
    adata: sc.AnnData,
    group_col: str,
    group_one: str = "1",
    group_zero: str = "0",
    use_raw: bool = True,
    method: str = "wilcoxon",
) -> pd.DataFrame:
    groups_present = set(adata.obs[group_col].astype(str).unique())
    if group_one not in groups_present or group_zero not in groups_present:
        return pd.DataFrame()

    key = "rank_genes_groups_de"
    sc.tl.rank_genes_groups(
        adata,
        groupby=group_col,
        groups=[group_one],
        reference=group_zero,
        method=method,
        use_raw=use_raw,
        pts=True,
        key_added=key,
    )
    df = sc.get.rank_genes_groups_df(adata, group=group_one, key=key)
    return df


def _compute_task_de(
    task_name: str,
    adata_task: sc.AnnData,
    output_dir: Path,
    cell_type_key: str,
    min_cells_per_group: int,
    use_raw: bool,
    top_n: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Task: {task_name} ===")
    print(f"Cells after task filter: {adata_task.n_obs}")
    print(f"Genes: {adata_task.n_vars}")
    print(f"Label counts: {adata_task.obs['de_label'].value_counts().to_dict()}")

    # Pooled DE
    pooled_df = _run_rank_genes_groups(adata_task, "de_label", use_raw=use_raw)
    if top_n is not None and not pooled_df.empty:
        pooled_df = pooled_df.head(top_n)
    pooled_df.insert(0, "task", task_name)
    pooled_df.insert(1, "scope", "pooled")
    pooled_out = task_dir / f"de_pooled_{task_name}.csv"
    pooled_df.to_csv(pooled_out, index=False)
    print(f"Saved pooled DE: {pooled_out}")

    # Cell-type-specific DE
    ct_rows = []
    if cell_type_key not in adata_task.obs:
        print(f"'{cell_type_key}' not in obs; skipping cell-type-specific DE.")
        ct_df = pd.DataFrame()
    else:
        for ct in sorted(adata_task.obs[cell_type_key].astype(str).unique()):
            ad_ct = adata_task[adata_task.obs[cell_type_key].astype(str) == ct].copy()
            counts = ad_ct.obs["de_label"].value_counts().to_dict()
            n0 = int(counts.get("0", 0))
            n1 = int(counts.get("1", 0))
            if n0 < min_cells_per_group or n1 < min_cells_per_group:
                continue

            df_ct = _run_rank_genes_groups(ad_ct, "de_label", use_raw=use_raw)
            if df_ct.empty:
                continue
            if top_n is not None:
                df_ct = df_ct.head(top_n)
            df_ct.insert(0, "task", task_name)
            df_ct.insert(1, "scope", "cell_type_specific")
            df_ct.insert(2, "cell_type", ct)
            df_ct.insert(3, "n_group1", n1)
            df_ct.insert(4, "n_group0", n0)
            ct_rows.append(df_ct)

        ct_df = pd.concat(ct_rows, axis=0, ignore_index=True) if ct_rows else pd.DataFrame()
        ct_out = task_dir / f"de_cell_type_specific_{task_name}.csv"
        ct_df.to_csv(ct_out, index=False)
        print(f"Saved cell-type-specific DE: {ct_out}")

    return pooled_df, ct_df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad-path", type=str, default=DEFAULT_H5AD)
    parser.add_argument("--output-dir", type=str, default="./m2_de_results")
    parser.add_argument("--cell-type-key", type=str, default="cell_type")
    parser.add_argument("--min-cells-per-group", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=None, help="Keep top N genes per DE table (default: all)")
    parser.add_argument("--no-use-raw", action="store_true", help="Do not use adata.raw for DE")
    parser.add_argument(
        "--include-non-neuronal-for-thr",
        action="store_true",
        help="Do not exclude non-neuronal cell types for THR task",
    )
    args = parser.parse_args()

    h5ad_path = Path(args.h5ad_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AnnData: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"obs keys: {list(adata.obs.columns)}")

    use_raw = (not args.no_use_raw) and (adata.raw is not None)
    print(f"use_raw for DE: {use_raw}")

    ad_treatment = _prepare_treatment_task(adata)
    ad_thr = _prepare_thr_task(
        adata,
        exclude_non_neuronal=(not args.include_non_neuronal_for_thr),
        cell_type_key=args.cell_type_key,
    )

    pooled_t, ct_t = _compute_task_de(
        "treatment", ad_treatment, out_dir, args.cell_type_key, args.min_cells_per_group, use_raw, args.top_n)
    pooled_h, ct_h = _compute_task_de(
        "THR", ad_thr, out_dir, args.cell_type_key, args.min_cells_per_group, use_raw, args.top_n)

    combined_pooled = pd.concat([pooled_t, pooled_h], axis=0, ignore_index=True)
    combined_ct = pd.concat([ct_t, ct_h], axis=0, ignore_index=True)
    pooled_combined_path = out_dir / "de_pooled_all_tasks.csv"
    ct_combined_path = out_dir / "de_cell_type_specific_all_tasks.csv"
    combined_pooled.to_csv(pooled_combined_path, index=False)
    combined_ct.to_csv(ct_combined_path, index=False)
    print(f"\nSaved combined pooled DE: {pooled_combined_path}")
    print(f"Saved combined cell-type-specific DE: {ct_combined_path}")


if __name__ == "__main__":
    main()
