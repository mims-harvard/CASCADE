#!/usr/bin/env python3
"""
Ground-truth validation of CASCADE-Explainer gene rankings against canonical
cell-type marker sets (Methods 2.4, "CASCADE-Explainer recovers canonical
cell-type gene programmes"; healthy-only SEATTLE cell-type explainer
validation).

Two complementary analyses:
  - analyze_ground_truth(): proportion of GT markers ranked better than the
    cell type's own average, with a permutation null (per ranking file, per
    GT source).
  - analyze_ground_truth_auroc(): per-cell-type AUROC treating "is a GT
    marker" as the binary label and the gene's rank as the score, with a
    permutation null p-value. Used by create_combined_gt_viz.py to compare
    multiple ranking sources (CASCADE / attention-baseline / DEG) on a
    shared scale.

Ground-truth marker sources: a folder of one-gene-per-line .txt files
(in-house/experimental), an Excel workbook (literature/GT_paper), and
optional up-/down-regulated-only variants.

Usage:
    python -m analysis.alzheimers.gt_analysis \
        --data-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation \
        --ground-truth-sources txt paper
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cascade.explainer.config import CASCADE_DATA_ROOT

CELL_TYPE_MAP = {
    "Astro": "Astrocytes",
    "Endo": "Endothelial",
    "IO": "IO",
    "Micro": "Microglia",
    "Oligo": "Oligodendrocytes",
    "OPC": "OPC",
    # Not present in current median_gene_ranks_by_cell_type.csv
    "Bcells": "B cells",
    "Tcells": "T cells",
    "Neuron": "Neurons",
}

LEGEND_LABELS = {
    "GT-paper": "GT paper",
    "GT-experimental": "GT experimental",
    "GT-down": "GT down",
    "GT-updown": "GT up+down",
}

FONT_TITLE = 16
FONT_AXIS = 14
FONT_TICK = 12
FONT_LEGEND = 13


def _legend_label(display_label: str, random: bool) -> str:
    base = LEGEND_LABELS.get(display_label, display_label)
    return f"{base} (random)" if random else base


def p_to_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def load_biomart_mapping(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, usecols=["approved_symbol", "alias_symbol", "ensembl_gene_id"])

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        ensembl = str(row["ensembl_gene_id"]).strip()
        if not ensembl or ensembl == "nan":
            continue

        approved = str(row["approved_symbol"]).strip()
        if approved and approved != "nan":
            mapping.setdefault(approved.upper(), ensembl)

        alias_raw = str(row.get("alias_symbol", "")).strip()
        if alias_raw and alias_raw != "nan":
            for alias in re.split(r"[|,;]", alias_raw):
                alias = alias.strip()
                if alias:
                    mapping.setdefault(alias.upper(), ensembl)

    return mapping


def load_ground_truth_markers(path: Path) -> dict[str, list[str]]:
    markers: dict[str, list[str]] = {}
    for txt_path in sorted(path.glob("*.txt")):
        key = txt_path.stem.split("_")[0]
        with txt_path.open() as handle:
            genes = [line.strip() for line in handle if line.strip()]
        markers[key] = genes
    return markers


def load_ground_truth_from_excel(path: Path, sheet_name: str = "cell type marker signatures") -> dict[str, list[str]]:
    df = pd.read_excel(path, sheet_name=sheet_name)
    markers: dict[str, list[str]] = {}
    for column in df.columns:
        symbols = [str(val).strip() for val in df[column].tolist() if pd.notna(val) and str(val).strip()]
        if symbols:
            markers[column] = symbols
    return markers


def load_ranking_file(path: Path) -> pd.DataFrame:
    """Load a gene-importance ranking file, normalizing column names across
    the different ranking sources (CASCADE median-rank exports, attention
    baseline, DEG). Handles an optional 'direction' column (up/down) by
    flipping down-regulated ranks so both directions share one scale."""
    df = pd.read_csv(path).copy()

    if "gene" not in df.columns:
        if "ensembl_gene_id" in df.columns:
            df["gene"] = df["ensembl_gene_id"]
        else:
            raise ValueError(f"Ranking file {path} must contain a 'gene' or 'ensembl_gene_id' column.")

    if "median_rank" not in df.columns:
        if "median_rank_position" in df.columns:
            df["median_rank"] = df["median_rank_position"]
        else:
            raise ValueError(f"Ranking file {path} must contain 'median_rank' or 'median_rank_position'.")

    if "n_rows" not in df.columns:
        df["n_rows"] = df["cells_contributing"] if "cells_contributing" in df.columns else 1

    if "cell_type" not in df.columns:
        raise ValueError(f"Ranking file {path} must contain 'cell_type'.")

    if "direction" in df.columns:
        pieces: list[pd.DataFrame] = []
        for cell_type, sub in df.groupby("cell_type"):
            up_part = sub[sub["direction"].str.lower() == "up"].copy()
            down_part = sub[sub["direction"].str.lower() == "down"].copy()
            pieces.append(up_part)
            if not down_part.empty:
                max_rank = down_part["median_rank"].max()
                down_part["median_rank"] = max_rank + (max_rank + 1 - down_part["median_rank"])
                pieces.append(down_part)
        df = pd.concat(pieces, ignore_index=True)

    df["gene"] = df["gene"].astype(str).str.strip()
    return df


def load_cell_type_mapping(path: Path) -> dict[str, str]:
    """Optional fine-grained -> coarse cell-type label mapping (e.g. so
    attention-baseline subtypes can be aggregated to match CASCADE's coarse
    cell types)."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    fine_col, coarse_col = "Fine label", "Coarse label"
    if fine_col not in df.columns or coarse_col not in df.columns:
        return {}
    mapping = {}
    for _, row in df.iterrows():
        fine, coarse = str(row[fine_col]).strip(), str(row[coarse_col]).strip()
        if fine and coarse and coarse.lower() != "nan":
            mapping[fine] = coarse
    return mapping


def analyze_ground_truth_auroc(
    *,
    gt_markers: dict[str, list[str]],
    gt_source: str,
    biomart_mapping: dict[str, str],
    median_df: pd.DataFrame,
    rng_seed: int | None,
    bootstrap_iters: int = 1000,
) -> tuple[pd.DataFrame, list[float], list[float], list[float]]:
    """Per-cell-type AUROC for gene ranking validation.

    Binary label: 1 = gene is a GT marker, 0 = not. Score: a composite of
    n_rows (cells contributing) and rank position, so CASCADE's cell-level
    aggregated rankings and single-shot DEG/attention rankings are scored on
    a comparable footing without an n_rows filter (both sources use exactly
    the same gene universe). Significance is a permutation null on the
    labels (bootstrap_iters shuffles).
    """
    rng = np.random.default_rng(rng_seed)
    available_cell_types = set(median_df["cell_type"].unique())

    rows = []
    for gt_key, symbols in gt_markers.items():
        cell_type = CELL_TYPE_MAP.get(gt_key, gt_key)
        if cell_type not in available_cell_types:
            continue

        cell_df = median_df[median_df["cell_type"] == cell_type].copy()

        mapped_ensembl = {sym: biomart_mapping[sym.upper()] for sym in symbols if sym.upper() in biomart_mapping}
        positive_ids = set(mapped_ensembl.values())

        labels = cell_df["gene"].isin(positive_ids).astype(int).to_numpy()
        # Composite score: n_rows x (max_rank + 1 - median_rank). For CASCADE
        # (max_rank=100): genes ranking near 1 in many cells score highest.
        # For DEG/Attention where n_rows=1: reduces to (max_rank+1-median_rank),
        # equivalent to -median_rank but preserving positivity.
        rank_vals = cell_df["median_rank"].to_numpy()
        n_rows_vals = cell_df["n_rows"].to_numpy() if "n_rows" in cell_df.columns else np.ones(len(cell_df))
        max_rank = rank_vals.max()
        scores = n_rows_vals * (max_rank + 1 - rank_vals)

        if labels.sum() < 2 or (labels == 0).sum() < 2:
            rows.append({"gt_source": gt_source, "cell_type": cell_type, "auroc": np.nan,
                        "rand_mean": np.nan, "rand_se": np.nan, "p_value": np.nan,
                        "n_positive": int(labels.sum()), "n_universe": len(labels)})
            continue

        auroc = float(roc_auc_score(labels, scores))
        null_aurocs = np.array([float(roc_auc_score(labels, rng.permutation(scores))) for _ in range(bootstrap_iters)])
        p_val = float((null_aurocs >= auroc).sum() + 1) / (bootstrap_iters + 1)

        rows.append({
            "gt_source": gt_source, "cell_type": cell_type, "auroc": auroc,
            "rand_mean": float(null_aurocs.mean()), "rand_se": float(null_aurocs.std(ddof=1) / np.sqrt(bootstrap_iters)),
            "p_value": p_val, "n_positive": int(labels.sum()), "n_universe": len(labels),
        })
        print(f"  [{gt_source}] {cell_type}: AUROC={auroc:.3f}  n_pos={labels.sum()}, n_universe={len(labels)}, p={p_val:.4f}")

    plot_df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    return plot_df, plot_df["rand_mean"].tolist(), plot_df["rand_se"].tolist(), plot_df["p_value"].tolist()


def analyze_ground_truth(
    *,
    gt_markers: dict[str, list[str]],
    gt_source: str,
    biomart_mapping: dict[str, str],
    median_df: pd.DataFrame,
    args: argparse.Namespace,
    rng_seed: int | None,
):
    """Proportion-of-GT-markers-ranked-better-than-average analysis, with a
    permutation null (per cell type)."""
    summary_rows: list[dict[str, object]] = []
    marker_rows: list[dict[str, object]] = []
    unmapped_rows: list[dict[str, object]] = []

    available_cell_types = set(median_df["cell_type"].unique())
    rng = np.random.default_rng(rng_seed)

    for gt_key, symbols in gt_markers.items():
        cell_type = CELL_TYPE_MAP.get(gt_key, gt_key)
        if cell_type not in available_cell_types:
            summary_rows.append({"gt_source": gt_source, "gt_key": gt_key, "cell_type": cell_type,
                                "status": "missing_cell_type", "marker_count": len(symbols)})
            continue

        cell_df = median_df[median_df["cell_type"] == cell_type]
        n_rows_threshold = None
        if args.n_rows_percentile is not None:
            n_rows_threshold = cell_df["n_rows"].quantile(args.n_rows_percentile)
            cell_df = cell_df[cell_df["n_rows"] > n_rows_threshold]
        if args.ranking_mode == "extreme":
            cell_df = cell_df.assign(rank_score=(cell_df["median_rank"] - 50).abs())
        else:
            cell_df = cell_df.assign(rank_score=cell_df["median_rank"])

        overall_avg = cell_df["rank_score"].mean()
        overall_median = cell_df["rank_score"].median()

        mapped_ensembl: dict[str, str] = {}
        for symbol in symbols:
            ensembl = biomart_mapping.get(symbol.upper())
            if ensembl:
                mapped_ensembl[symbol] = ensembl
            else:
                unmapped_rows.append({"gt_source": gt_source, "gt_key": gt_key, "cell_type": cell_type, "symbol": symbol})

        marker_df = cell_df[cell_df["gene"].isin(mapped_ensembl.values())]
        marker_avg = marker_df["rank_score"].mean() if not marker_df.empty else None
        marker_median = marker_df["rank_score"].median() if not marker_df.empty else None

        better_than_avg = None
        prop_se = None
        if not marker_df.empty:
            if args.ranking_mode == "extreme":
                better_than_avg = (marker_df["rank_score"] > overall_avg).mean()
            else:
                better_than_avg = (marker_df["rank_score"] < overall_avg).mean()
            n_markers = len(marker_df)
            if n_markers > 0 and better_than_avg is not None:
                prop_se = (better_than_avg * (1 - better_than_avg) / n_markers) ** 0.5

        summary_rows.append({
            "gt_source": gt_source, "gt_key": gt_key, "cell_type": cell_type, "status": "ok",
            "marker_count": len(symbols), "mapped_markers": len(mapped_ensembl), "markers_in_matrix": len(marker_df),
            "ranking_mode": args.ranking_mode, "overall_avg_rank": overall_avg, "overall_median_rank": overall_median,
            "marker_avg_rank": marker_avg, "marker_median_rank": marker_median,
            "prop_markers_better_than_avg": better_than_avg, "prop_markers_se": prop_se,
        })

        ensembl_to_symbol = {v: k for k, v in mapped_ensembl.items()}
        for _, row in marker_df.iterrows():
            marker_rows.append({
                "gt_source": gt_source, "gt_key": gt_key, "cell_type": cell_type,
                "symbol": ensembl_to_symbol.get(row["gene"], ""), "ensembl_gene_id": row["gene"],
                "n_rows": row["n_rows"], "median_rank": row["median_rank"], "rank_score": row["rank_score"],
                "better_than_avg": (row["rank_score"] > overall_avg if args.ranking_mode == "extreme"
                                    else row["rank_score"] < overall_avg),
            })

        prefix = f"[{gt_source}] "
        if n_rows_threshold is None:
            print(f"\n{prefix}{cell_type}")
        else:
            print(f"\n{prefix}{cell_type} (n_rows > {n_rows_threshold:.2f})")
        if marker_df.empty:
            print("  No mapped markers found after filtering.")
        else:
            print("  Least common markers:")
            for _, row in marker_df.nsmallest(5, "n_rows").iterrows():
                print(f"    {ensembl_to_symbol.get(row['gene'], '')}\t{row['gene']}\t{row['n_rows']}")
            print("  Most common markers:")
            for _, row in marker_df.nlargest(5, "n_rows").iterrows():
                print(f"    {ensembl_to_symbol.get(row['gene'], '')}\t{row['gene']}\t{row['n_rows']}")

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty or "status" not in summary_df.columns:
        return summary_rows, marker_rows, unmapped_rows, pd.DataFrame(
            columns=["cell_type", "prop_markers_better_than_avg", "prop_markers_se"]), [], [], []

    plot_df = summary_df[summary_df["status"] == "ok"].copy()
    plot_df = plot_df.sort_values(["prop_markers_better_than_avg", "cell_type"], ascending=[False, True])

    random_baselines: list[list[float]] = []
    for cell_type in plot_df["cell_type"]:
        cell_df = median_df[median_df["cell_type"] == cell_type]
        markers = [row["ensembl_gene_id"] for row in marker_rows
                   if row["cell_type"] == cell_type and row["gt_source"] == gt_source]
        if not markers:
            random_baselines.append([])
            continue

        if args.ranking_mode == "extreme":
            cell_df = cell_df.assign(rank_score=(cell_df["median_rank"] - 50).abs())
        else:
            cell_df = cell_df.assign(rank_score=cell_df["median_rank"])

        overall_avg = cell_df["rank_score"].mean()
        gene_index = cell_df["gene"].tolist()
        random_props: list[float] = []
        for _ in range(args.bootstrap_iters):
            shuffled_rank = pd.Series(rng.permutation(cell_df["rank_score"].to_numpy()), index=gene_index)
            marker_ranks = shuffled_rank.loc[shuffled_rank.index.intersection(markers)]
            if marker_ranks.empty:
                continue
            if args.ranking_mode == "extreme":
                random_props.append((marker_ranks > overall_avg).mean())
            else:
                random_props.append((marker_ranks < overall_avg).mean())
        random_baselines.append(random_props)

    random_means, random_ses, p_values = [], [], []
    for random_vals, observed in zip(random_baselines, plot_df["prop_markers_better_than_avg"]):
        if not random_vals:
            random_means.append(np.nan)
            random_ses.append(np.nan)
            p_values.append(np.nan)
            continue
        mean_val = float(np.mean(random_vals))
        random_means.append(mean_val)
        random_ses.append(float(np.std(random_vals, ddof=1) / np.sqrt(len(random_vals))))
        p_values.append((sum(val >= observed for val in random_vals) + 1) / (len(random_vals) + 1))

    print("\nRandom vs markers p-values (one-sided, random >= observed):")
    for cell_type, p_val in zip(plot_df["cell_type"], p_values):
        print(f"  [{gt_source}] {cell_type}\t{'NA' if np.isnan(p_val) else f'{p_val:.4f}'}")

    return summary_rows, marker_rows, unmapped_rows, plot_df, random_means, random_ses, p_values


def plot_performance(*, plot_df, random_means, random_ses, args, label, ranking_label, out_path):
    mask = plot_df["cell_type"] != "Endothelial"
    plot_df = plot_df[mask]
    random_means = [val for val, keep in zip(random_means, mask) if keep]
    random_ses = [val for val, keep in zip(random_ses, mask) if keep]
    if plot_df.empty:
        print(f"Plot skipped for {label}: empty after filtering.")
        return

    plt.figure(figsize=(5, 4))
    x = np.arange(len(plot_df)) * 0.5
    offset = 0.10

    plt.errorbar(x - offset, plot_df["prop_markers_better_than_avg"], yerr=plot_df["prop_markers_se"],
                fmt="o", color="tab:blue", label=_legend_label(label, False), capsize=3)
    plt.errorbar(x + offset, random_means, yerr=random_ses, fmt="o", color="#999999",
                label=_legend_label(label, True), capsize=3)

    plt.xticks(x, plot_df["cell_type"], rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Ratio of markers > non-markers", fontsize=FONT_AXIS)
    plt.xlabel("Cell type", fontsize=FONT_AXIS)
    plt.title("Gene explainer benchmarking", fontsize=FONT_TITLE)
    plt.gca().tick_params(axis="both", labelsize=FONT_TICK)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_comparison_entries(entries, title, out_path):
    entries = [e for e in entries if e.get("plot_df") is not None and not e["plot_df"].empty
               and len(e.get("random_means", [])) == len(e["plot_df"])]
    if not entries:
        return

    all_cell_types = sorted({ct for entry in entries for ct in entry["plot_df"]["cell_type"] if ct != "Endothelial"})
    if not all_cell_types:
        print("Comparison plot skipped: no cell types after filtering.")
        return

    x_gap = 1.2
    x = np.arange(len(all_cell_types)) * x_gap
    total_series = len(entries) * 2
    spacing = 0.10
    offsets = [(i - (total_series - 1) / 2) * spacing for i in range(total_series)]

    source_colors = {"GT-experimental": "#4878CF", "GT-paper": "#6ACC65", "GT-down": "#D65F5F", "GT-updown": "#B47CC7"}
    fallback_colors = list(plt.cm.tab10.colors)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    band_half = x_gap / 2
    for i in range(len(all_cell_types)):
        if i % 2 == 0:
            ax.axvspan(x[i] - band_half, x[i] + band_half, color="#f0f0f0", zorder=0)

    series_idx = 0
    for src_idx, entry in enumerate(entries):
        display_label = entry.get("display_label", entry.get("label", ""))
        color = source_colors.get(display_label, fallback_colors[src_idx % len(fallback_colors)])
        aligned = entry["plot_df"].set_index("cell_type").reindex(all_cell_types)
        p_aligned = pd.Series(entry["p_values"], index=entry["plot_df"]["cell_type"]).reindex(all_cell_types)

        y, yerr = aligned["prop_markers_better_than_avg"], aligned["prop_markers_se"]
        offset = offsets[series_idx]
        ax.errorbar(x + offset, y, yerr=yerr, fmt="o", color=color, alpha=1.0,
                    label=_legend_label(display_label, False), capsize=3, zorder=3)
        stars = [p_to_stars(p) for p in p_aligned]
        yerr_clean = yerr.fillna(0)
        for xpos, yval, yerr_val, star in zip(x, y, yerr_clean, stars):
            if star:
                ax.text(xpos + offset, yval + (yerr_val if not pd.isna(yerr_val) else 0) + 0.02, star,
                        ha="center", va="bottom", fontsize=11, fontweight="bold", color="#000000", zorder=4)
        series_idx += 1

        y = pd.Series(entry["random_means"], index=aligned.index)
        yerr = pd.Series(entry["random_ses"], index=aligned.index)
        offset = offsets[series_idx]
        ax.errorbar(x + offset, y, yerr=yerr, fmt="o", color=color, alpha=0.35,
                    label=_legend_label(display_label, True), capsize=3, zorder=3)
        series_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(all_cell_types, rotation=25, ha="right")
    ax.set_xlim(x[0] - band_half, x[-1] + band_half)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Ratio of markers > non-markers", fontsize=FONT_AXIS)
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.tick_params(axis="both", labelsize=FONT_TICK)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.30),
             ncol=4, frameon=True, fancybox=True, columnspacing=1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_side_by_side_rankings(entries, rank_labels, out_path, title="Gene explainer benchmarking (side by side)"):
    if len(rank_labels) != 2:
        return
    grouped: dict[str, list[dict[str, object]]] = {}
    for e in entries:
        grouped.setdefault(e.get("ranking_label", ""), []).append(e)
    left_entries, right_entries = grouped.get(rank_labels[0], []), grouped.get(rank_labels[1], [])
    if not left_entries or not right_entries:
        return

    all_cell_types = sorted({ct for entry in entries for ct in entry["plot_df"]["cell_type"] if ct != "Endothelial"})
    if not all_cell_types:
        return

    color_map = {"GT-experimental": "#1f77b4", "GT-paper": "#ff7f0e", "GT-down": "#2ca02c", "GT-updown": "#d62728"}

    def plot_panel(ax, panel_entries, panel_title):
        x = np.arange(len(all_cell_types)) * 0.5
        offsets = {"GT-experimental": -0.12, "GT-paper": -0.04, "GT-down": 0.04, "GT-updown": 0.12}
        for entry in panel_entries:
            display_label = entry.get("display_label", entry.get("label", ""))
            color = color_map.get(display_label, "#555555")
            aligned = entry["plot_df"].set_index("cell_type").reindex(all_cell_types)
            offset = offsets.get(display_label, 0.0)
            ax.errorbar(x + offset, aligned["prop_markers_better_than_avg"], yerr=aligned["prop_markers_se"],
                       fmt="o", color=color, alpha=1.0, label=_legend_label(display_label, False), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(all_cell_types, rotation=25, ha="right", fontsize=FONT_TICK)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Ratio of markers > non-markers", fontsize=FONT_AXIS)
        ax.set_xlabel("Cell type", fontsize=FONT_AXIS)
        ax.set_title(panel_title, fontsize=FONT_TITLE)
        ax.tick_params(axis="both", labelsize=FONT_TICK)

    fig, axes = plt.subplots(1, 2, figsize=(6, 4), sharey=True)
    plot_panel(axes[0], left_entries, rank_labels[0])
    plot_panel(axes[1], right_entries, rank_labels[1])
    fig.suptitle(title, fontsize=FONT_TITLE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=CASCADE_DATA_ROOT / "SEATTLE" / "gene_explainer_validation",
                        help="Directory containing ground_truth/, GT_paper.xlsx, biomart_cleaned.csv, and the default ranking file.")
    parser.add_argument("--n-rows-percentile", type=float, default=0.25,
                        help="Filter genes to those with n_rows above this percentile (0-1).")
    parser.add_argument("--ranking-mode", choices=["lower", "extreme"], default="extreme",
                        help="'lower' treats smaller median_rank as better; 'extreme' treats distance from 50 as better.")
    parser.add_argument("--bootstrap-iters", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--ground-truth-sources", choices=["txt", "paper", "down", "updown"], nargs="+", default=["txt"])
    parser.add_argument("--paper-path", type=Path, default=None)
    parser.add_argument("--paper-sheet", default="cell type marker signatures")
    parser.add_argument("--rank-files", type=Path, nargs="+", default=None,
                        help="Ranking files to compare. Defaults to median_gene_ranks_by_cell_type.csv.")
    parser.add_argument("--rank-labels", nargs="+", default=None)
    parser.add_argument("--down-path", type=Path, default=None)
    parser.add_argument("--updown-path", type=Path, default=None)
    args = parser.parse_args()

    base_dir = args.data_dir
    ground_truth_dir = base_dir / "ground_truth"
    paper_path = args.paper_path or (base_dir / "GT_paper.xlsx")
    down_dir = args.down_path or (base_dir / "ground_truth_down")
    updown_dir = args.updown_path or (base_dir / "ground_truth_updown")
    biomart_path = base_dir / "biomart_cleaned.csv"
    mapping_path = base_dir / "mapping-final-seattle.csv"
    rank_files = args.rank_files or [base_dir / "median_gene_ranks_by_cell_type.csv"]
    rank_labels = args.rank_labels or [path.stem for path in rank_files]
    if len(rank_files) != len(rank_labels):
        raise ValueError("Number of --rank-files must match --rank-labels.")
    rank_files = [path if path.is_absolute() else (base_dir / path) for path in rank_files]

    biomart_mapping = load_biomart_mapping(biomart_path)
    cell_type_mapping = load_cell_type_mapping(mapping_path)

    combined_summary_rows, combined_marker_rows, combined_unmapped_rows = [], [], []
    all_comparison_entries: list[dict[str, object]] = []

    for rank_path, rank_label in zip(rank_files, rank_labels):
        median_df = load_ranking_file(rank_path)
        if cell_type_mapping:
            median_df["cell_type"] = median_df["cell_type"].map(lambda v: cell_type_mapping.get(str(v).strip(), str(v).strip()))

        comparison_entries: list[dict[str, object]] = []
        for src in args.ground_truth_sources:
            if src == "txt":
                gt_markers, label, display_label = load_ground_truth_markers(ground_truth_dir), "ground_truth_folder", "GT-experimental"
            elif src == "paper":
                gt_markers, label, display_label = load_ground_truth_from_excel(paper_path, sheet_name=args.paper_sheet), "GT_paper", "GT-paper"
            elif src == "down":
                gt_markers, label, display_label = load_ground_truth_markers(down_dir), "ground_truth_down", "GT-down"
            else:
                gt_markers, label, display_label = load_ground_truth_markers(updown_dir), "ground_truth_updown", "GT-updown"

            summary_rows, marker_rows, unmapped_rows, plot_df, random_means, random_ses, p_values = analyze_ground_truth(
                gt_markers=gt_markers, gt_source=label, biomart_mapping=biomart_mapping,
                median_df=median_df, args=args, rng_seed=args.random_seed,
            )
            for row in summary_rows + marker_rows + unmapped_rows:
                row["ranking_label"] = rank_label

            combined_summary_rows.extend(summary_rows)
            combined_marker_rows.extend(marker_rows)
            combined_unmapped_rows.extend(unmapped_rows)

            keep_mask = plot_df["cell_type"] != "Endothelial"
            comparison_entries.append({
                "label": label, "display_label": display_label, "ranking_label": rank_label,
                "plot_df": plot_df[keep_mask],
                "random_means": [v for v, k in zip(random_means, keep_mask) if k],
                "random_ses": [v for v, k in zip(random_ses, keep_mask) if k],
                "p_values": [v for v, k in zip(p_values, keep_mask) if k],
            })

            per_plot_path = base_dir / f"gt_analysis_performance_{label}_{rank_label}.png"
            plot_performance(plot_df=plot_df, random_means=random_means, random_ses=random_ses,
                            args=args, label=display_label, ranking_label=rank_label, out_path=per_plot_path)
            print(f"Wrote per-source plot: {per_plot_path}")

        if len(comparison_entries) > 1:
            comparison_plot_path = base_dir / f"gt_analysis_performance_comparison_{rank_label}.png"
            _plot_comparison_entries(comparison_entries, title="Gene explainer benchmarking", out_path=comparison_plot_path)
            print(f"Wrote comparison plot: {comparison_plot_path}")

        all_comparison_entries.extend(comparison_entries)

    if len(all_comparison_entries) > 1:
        comparison_plot_path = base_dir / "gt_analysis_performance_comparison.png"
        _plot_comparison_entries(all_comparison_entries, title="Gene explainer benchmarking (all rankings)", out_path=comparison_plot_path)
        print(f"Wrote combined comparison plot: {comparison_plot_path}")
        if len(rank_labels) == 2:
            side_by_side_path = base_dir / "gt_analysis_performance_comparison_side_by_side.png"
            _plot_side_by_side_rankings(all_comparison_entries, rank_labels=rank_labels, out_path=side_by_side_path)
            print(f"Wrote side-by-side comparison plot: {side_by_side_path}")

    pd.DataFrame(combined_summary_rows).to_csv(base_dir / "gt_analysis_summary.csv", index=False)
    pd.DataFrame(combined_marker_rows).to_csv(base_dir / "gt_analysis_markers.csv", index=False)
    pd.DataFrame(combined_unmapped_rows).to_csv(base_dir / "gt_analysis_unmapped_markers.csv", index=False)
    print("Wrote combined tables to", base_dir)


if __name__ == "__main__":
    main()
