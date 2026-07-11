#!/usr/bin/env python3
"""
Sensitivity analysis on the stability of the non-zero per-gene median used as the
context-reference statistic for fold-change tokenization (Methods 9.4; see
Supplementary Notes 2-3): bootstraps 70% cell subsets and measures how much each
gene's non-zero median expression shifts relative to the full-data value.

Usage:
    python -m analysis.benchmarking.sensitivity.median_stability --h5ad /path/to/data.h5ad --output-dir ./results
    python -m analysis.benchmarking.sensitivity.median_stability --h5ad ... --output-dir ... --stratify-obs tissue disease cell_type
    python -m analysis.benchmarking.sensitivity.median_stability --from-npz /path/to/median_leakage_bootstrap.npz
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from tqdm import tqdm


def per_gene_median_expression(X, axis=0) -> np.ndarray:
    """
    Per-gene median across cells using **non-zero** values only (axis=0 -> one value per gene).

    - Sparse X: uses stored entries per column only (implicit zeros excluded); drops
      any explicit zeros in .data.
    - Dense X: nanmedian after masking zeros with NaN (vectorized).
    All-zero genes -> 0.0.
    """
    if axis != 0:
        raise ValueError("only axis=0 (median across cells per gene) is supported")

    if sp.issparse(X):
        xcsc = X.tocsc()
        n_genes = xcsc.shape[1]
        out = np.zeros(n_genes, dtype=np.float64)
        for j in range(n_genes):
            data = xcsc.getcol(j).data
            if data.size == 0:
                continue
            data = data[data != 0]
            if data.size:
                out[j] = float(np.median(data))
        return out

    xm = np.asarray(X, dtype=np.float64)
    masked = np.where(xm == 0, np.nan, xm)
    out = np.nanmedian(masked, axis=0)
    out = np.where(np.isfinite(out), out, 0.0)
    return np.asarray(out, dtype=np.float64).ravel()


def _stratify_seed(base_seed: int, col: str, cat: str) -> int:
    """Stable per-(col, category) seed (not Python's salted hash())."""
    h = hashlib.sha256(f"{col}\0{cat}".encode("utf-8")).digest()
    return (base_seed + int.from_bytes(h[:4], "little")) % (2**31)


def slugify_obs_label(label: str, max_len: int = 80) -> str:
    """Filesystem-safe token from an obs category string."""
    s = str(label).strip()
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return (s[:max_len] if s else "empty")


def compute_median_leakage(
    adata: sc.AnnData,
    n_samples: int,
    frac: float,
    rng: np.random.Generator,
    *,
    show_bootstrap_progress: bool = True,
    subset_medians_dtype: np.dtype = np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (gene_names, full_median, subset_medians (n_samples, n_genes), median_deviation).

    subset_medians_dtype: use np.float32 for stratified runs to cut peak RAM (~half)
    for the bootstrap accumulator; summaries stay accurate enough for plotting/saving.
    """
    genes = np.asarray(adata.var_names.astype(str))
    n_obs = int(adata.n_obs)
    n_genes = int(adata.n_vars)
    if n_obs < 2:
        raise ValueError("need at least 2 cells for bootstrap subsets")

    full_median = per_gene_median_expression(adata.X, axis=0)
    subset_size = min(max(1, int(frac * n_obs)), n_obs - 1)

    acc_dt = np.dtype(subset_medians_dtype)
    subset_medians_arr = np.empty((n_samples, n_genes), dtype=acc_dt)
    it = range(n_samples)
    if show_bootstrap_progress:
        it = tqdm(it, desc="bootstrap medians")
    for i in it:
        subset_indices = rng.choice(n_obs, size=subset_size, replace=False)
        subset_median = per_gene_median_expression(adata[subset_indices].X, axis=0)
        subset_medians_arr[i] = subset_median.astype(acc_dt, copy=False)

    median_deviation = np.median(
        subset_medians_arr.astype(np.float64, copy=False) - full_median, axis=0,
    )
    return genes, full_median, subset_medians_arr, median_deviation


def save_leakage_bundle(
    genes, full_median, subset_medians, median_deviation, n_samples, frac, n_obs, n_vars,
    npz_path: Path, csv_path: Path, fig_path: Path, title: str | None = None, extra_npz_arrays: dict | None = None,
) -> None:
    """Write .npz (compressed), summary CSV, and scatter PNG."""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "gene_names": genes,
        "full_median": full_median.astype(np.float64),
        "subset_medians": subset_medians.astype(np.float64),
        "median_deviation": median_deviation.astype(np.float64),
        "n_bootstrap": n_samples,
        "subset_frac": frac,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "median_rule": np.array("nonzero_per_cell_values", dtype=object),
    }
    if extra_npz_arrays:
        payload.update(extra_npz_arrays)
    np.savez_compressed(npz_path, **payload)
    print(f"Saved {npz_path}")

    pd.DataFrame({
        "gene": genes, "full_median": full_median, "median_deviation_70pct_subsets": median_deviation,
    }).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    plot_median_effect(full_median, median_deviation, fig_path, title=title)


def plot_median_effect(full_median, median_deviation, out_png: Path, title: str | None = None) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=full_median, y=median_deviation, alpha=0.5, s=8, linewidth=0)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Full data: median expr. among non-zero cells (per gene)")
    plt.ylabel("Median deviation in 70% subsets (same non-zero rule)")
    plt.title(title or "Per-gene median changes among non-zero cells (random 70% subsets)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")


def plot_from_npz(npz_path: Path, out_png: Path | None = None) -> None:
    z = np.load(npz_path)
    full_median = np.asarray(z["full_median"]).ravel()
    median_deviation = np.asarray(z["median_deviation"]).ravel()
    if out_png is None:
        out_png = npz_path.with_suffix(".png")
    title = None
    if "stratify_column" in z and "stratify_value" in z:
        col = str(np.asarray(z["stratify_column"]).ravel()[0])
        val = str(np.asarray(z["stratify_value"]).ravel()[0])
        title = f"Median leakage ({col}={val})"
    plot_median_effect(full_median, median_deviation, out_png, title=title)


def _npz_looks_complete(path: Path, min_bytes: int = 512) -> bool:
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    try:
        with np.load(path) as z:
            return "median_deviation" in z.files and "full_median" in z.files
    except Exception:
        return False


def run_stratified(
    adata: sc.AnnData, obs_columns: list[str], n_samples: int, frac: float, base_seed: int,
    output_dir: Path, dataset_name: str, *, resume: bool = False, force: bool = False,
) -> None:
    """Same leakage analysis per unique value in each obs column (skips tiny groups)."""
    tasks: list[tuple[str, str]] = []
    for col in obs_columns:
        if col not in adata.obs.columns:
            print(f"Warning: obs['{col}'] missing; skip stratification for this column.")
            continue
        cats = pd.Series(adata.obs[col]).dropna().astype(str).unique().tolist()
        cats.sort()
        tasks.extend((col, cat) for cat in cats)

    for col, cat in tqdm(tasks, desc="stratified groups"):
        tag = f"{slugify_obs_label(col, max_len=40)}__{slugify_obs_label(cat, max_len=80)}"
        npz_path = output_dir / f"{dataset_name}_median_leakage_bootstrap__{tag}.npz"

        if resume and not force and _npz_looks_complete(npz_path):
            tqdm.write(f"  Resume: skip existing {npz_path.name}")
            continue

        mask = pd.Series(adata.obs[col]).astype(str) == str(cat)
        # View only - avoid .copy(), which duplicates X/obs and commonly OOMs on large slices.
        sub = adata[mask.values]
        n_obs = int(sub.n_obs)
        if n_obs < 2:
            print(f"  Skip {col}={cat!r}: only {n_obs} cell(s)")
            continue
        rng = np.random.default_rng(_stratify_seed(base_seed, col, cat))

        try:
            genes, full_median, subset_medians, median_deviation = compute_median_leakage(
                sub, n_samples=n_samples, frac=frac, rng=rng,
                show_bootstrap_progress=False, subset_medians_dtype=np.float32,
            )
        except ValueError as e:
            print(f"  Skip {col}={cat!r}: {e}")
            continue

        save_leakage_bundle(
            genes, full_median, subset_medians, median_deviation, n_samples, frac, n_obs, int(sub.n_vars),
            npz_path, output_dir / f"{dataset_name}_median_leakage_summary__{tag}.csv",
            output_dir / f"{dataset_name}_median_effect__{tag}.png",
            title=f"Median leakage: {col} = {cat} (n={n_obs})",
            extra_npz_arrays={"stratify_column": np.array(str(col), dtype=object), "stratify_value": np.array(str(cat), dtype=object)},
        )
        del sub, genes, full_median, subset_medians, median_deviation
        gc.collect()


def main(
    h5ad_path: Path, output_dir: Path, dataset_name: str = "DATASET",
    stratify_obs: list[str] | None = None, *, resume: bool = False, force: bool = False, stratified_only: bool = False,
) -> None:
    if stratified_only and not stratify_obs:
        raise ValueError("stratified_only requires stratify_obs")

    base_seed = 0
    rng = np.random.default_rng(base_seed)
    adata = sc.read_h5ad(h5ad_path)

    n_samples, frac = 20, 0.7
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{dataset_name}_median_leakage_bootstrap.npz"
    csv_path = output_dir / f"{dataset_name}_median_leakage_summary.csv"
    fig_path = output_dir / f"{dataset_name}_median_effect.png"

    if not stratified_only:
        if resume and not force and _npz_looks_complete(npz_path):
            print(f"Resume: skip global run (exists): {npz_path}")
        else:
            genes, full_median, subset_medians, median_deviation = compute_median_leakage(adata, n_samples=n_samples, frac=frac, rng=rng)
            save_leakage_bundle(
                genes, full_median, subset_medians, median_deviation, n_samples, frac,
                int(adata.n_obs), int(adata.n_vars), npz_path, csv_path, fig_path,
                title="Per-gene median among non-zero cells (all cells; 70% subsets)",
            )
            del genes, full_median, subset_medians, median_deviation
            gc.collect()

    if stratify_obs:
        run_stratified(adata, stratify_obs, n_samples, frac, base_seed, output_dir, dataset_name, resume=resume, force=force)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", type=Path, default=None, help="Path to the AnnData (.h5ad) file")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dataset-name", type=str, default="DATASET")
    p.add_argument("--from-npz", type=Path, default=None, help="Only load a saved bootstrap bundle and write PNG next to it")
    p.add_argument("--out-png", type=Path, default=None)
    p.add_argument("--stratify-obs", nargs="*", default=None, metavar="COL",
                    help="obs column names (e.g. tissue disease cell_type); repeats the bootstrap per unique value")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stratified-only", action="store_true")
    args = p.parse_args()

    if args.from_npz is not None:
        plot_from_npz(args.from_npz, out_png=args.out_png)
    else:
        if args.h5ad is None or args.output_dir is None:
            p.error("--h5ad and --output-dir are required unless --from-npz is set")
        main(
            args.h5ad, args.output_dir, dataset_name=args.dataset_name,
            stratify_obs=list(args.stratify_obs) if args.stratify_obs else None,
            resume=args.resume, force=args.force, stratified_only=args.stratified_only,
        )
