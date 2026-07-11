#!/usr/bin/env python3
"""
Sensitivity analysis on the effect of sequence length (Methods 9.4; see
Supplementary Note 4): truncates gene-token sequences to varying lengths and
measures how many differentially-expressed (DE) / OpenTargets genes remain.

Per cell: fraction of tokens-of-interest retained after truncation vs. the full
sequence, then unweighted mean of that percentage across cells.

Sequence convention (matches cascade.data.collator):
  - Upregulated: genes ordered high -> low after <up>; keeping the first L tokens
    keeps the head (strongest up in that ordering).
  - Downregulated: genes ordered after <down>, in the tail of input_ids; keeping the
    last L tokens (suffix) measures retention when truncating from the end.

Usage:
    python -m analysis.benchmarking.sensitivity.sequence_length_ablation --dataset LUCA
    python -m analysis.benchmarking.sensitivity.sequence_length_ablation --dataset LUCA --inspect
"""
from __future__ import annotations

import argparse
import os
import pickle
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from datasets import concatenate_datasets
from tqdm import tqdm

from cascade.explainer.config import CASCADE_DATA_ROOT

# ---------------------------------------------------------------------------
# Per-dataset config: h5ad/tokenized paths, disease key, healthy labels, and
# (for LUCA) per-disease-vs-normal OpenTargets export filenames.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "LUCA": {
        "h5ad": CASCADE_DATA_ROOT / "LUCA/adata_annotated_protein_coding_clinical.h5ad",
        "tokenized_dir": CASCADE_DATA_ROOT / "LUCA/DISEASE/DISEASE-ID.dataset",
        "output_dir": CASCADE_DATA_ROOT / "LUCA",
        "tokenizer_dict": CASCADE_DATA_ROOT / "LUCA/tokenizer_dictionary_LUCA.pkl",
        "disease_key": "disease",
        "healthy_labels": ["normal"],
        "ot_dir": CASCADE_DATA_ROOT / "LUCA/opentargets",
        "per_disease_ot_files": {
            "lung adenocarcinoma": "OT-EFO_0000571-lung adenocarcinoma.tsv",
            "squamous cell lung carcinoma": "OT-EFO_0000708-squamous cell lung carcinoma.tsv",
            "non-small cell lung carcinoma": "OT-EFO_0003060-non-small cell lung carcinoma.tsv",
            "chronic obstructive pulmonary disease": "OT-EFO_0000341-chronic obstructive pulmonary disease.tsv",
        },
        "per_disease_analysis": True,
        "ot_tsv": CASCADE_DATA_ROOT / "LUCA/opentargets/OT-EFO_0003060-non-small cell lung carcinoma.tsv",
    },
    "AUTISM": {
        "h5ad": CASCADE_DATA_ROOT / "AUTISM/adata_annotated_protein_coding_clinical.h5ad",
        "tokenized_dir": CASCADE_DATA_ROOT / "AUTISM/tokenized_data_DISEASE_out-ref-final-DOWNCORRECTED/DISEASE-ID.dataset",
        "output_dir": CASCADE_DATA_ROOT / "AUTISM",
        "tokenizer_dict": CASCADE_DATA_ROOT / "AUTISM/tokenizer_dictionary_AUTISM.pkl",
        "disease_key": "disease",
        "healthy_labels": ["Control", "control"],
        "per_disease_analysis": False,
        "ot_tsv": CASCADE_DATA_ROOT / "AUTISM/opentargets/OT-EFO_0003060-associated-targets.tsv",
    },
}


HEALTHY_SET: Set = set()


def _is_healthy_cell(example: dict) -> bool:
    return str(example.get("disease")) in HEALTHY_SET


def _as_token_list(row_input_ids) -> List:
    """Arrow / numpy / list -> plain Python list for indexing."""
    if row_input_ids is None:
        return []
    if hasattr(row_input_ids, "tolist"):
        return list(row_input_ids.tolist())
    return list(row_input_ids)


def truncate_and_count(row: dict, length_threshold: int, tokens_of_interest: Set, analysis_type: str = "upregulated") -> dict:
    """Per row (cell): fraction of tokens_of_interest that still appear after truncation."""
    seq = _as_token_list(row.get("input_ids"))
    if not seq:
        return {"percentage_retained": 0.0, "n_total_hits": 0, "n_retained_hits": 0}

    L = min(int(length_threshold), len(seq))
    if analysis_type == "upregulated":
        truncated = seq[:L]
    elif analysis_type == "downregulated":
        truncated = seq[-L:]
    else:
        raise ValueError("analysis_type must be 'upregulated' or 'downregulated'")

    total_hits = sum(1 for t in seq if t in tokens_of_interest)
    retained_hits = sum(1 for t in truncated if t in tokens_of_interest)
    pct = (retained_hits / total_hits) * 100.0 if total_hits > 0 else 0.0
    return {"percentage_retained": pct, "n_total_hits": total_hits, "n_retained_hits": retained_hits}


def mean_retained_percent(ds: datasets.Dataset, tokens: Set, thresh: int, analysis_type: str, num_proc: Optional[int] = None) -> float:
    """Mean over cells of percentage_retained (unweighted)."""
    mapped = ds.map(
        truncate_and_count,
        fn_kwargs={"tokens_of_interest": tokens, "length_threshold": thresh, "analysis_type": analysis_type},
        remove_columns=ds.column_names, num_proc=num_proc,
    )
    arr = np.asarray(mapped["percentage_retained"], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else 0.0


def _load_curve_checkpoint(pkl_path: Path) -> Dict[int, float]:
    if not pkl_path.is_file():
        return {}
    try:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError) as e:
        print(f"Warning: could not read {pkl_path} ({e}); starting this curve from scratch.")
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, float] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _atomic_pickle_dump(pkl_path: Path, obj: object) -> None:
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pkl_path.with_suffix(pkl_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, pkl_path)


def retention_curve_with_checkpoints(
    ds: datasets.Dataset, genes: Set, thresholds: Sequence[int], analysis_type: str,
    num_proc: Optional[int], pkl_path: Path, resume: bool, desc: str,
) -> Dict[int, float]:
    """For each length threshold, mean % retained; checkpointed to pkl_path after every point."""
    need = {int(t) for t in thresholds}
    curve: Dict[int, float] = {}
    if resume:
        prev = _load_curve_checkpoint(pkl_path)
        curve = {k: v for k, v in prev.items() if k in need}
        if curve:
            print(f"  Resume {pkl_path.name}: loaded {len(curve)} / {len(need)} thresholds")

    for thresh in tqdm(thresholds, desc=desc):
        t = int(thresh)
        if t in curve:
            continue
        curve[t] = mean_retained_percent(ds, genes, t, analysis_type, num_proc=num_proc)
        _atomic_pickle_dump(pkl_path, curve)

    print(f"  Wrote {pkl_path} ({len(curve)} thresholds)")
    return curve


def load_merged_arrow_dataset(tokenized_dir: Path, keep_columns: Sequence[str] = ("input_ids", "disease", "cell_type")) -> datasets.Dataset:
    """
    Load all Arrow shards under tokenized_dir and concatenate, keeping only keep_columns
    (avoids schema mismatches on unused metadata across shards and reduces memory).
    """
    arrow_files = sorted(str(p) for p in Path(tokenized_dir).iterdir() if "arrow" in p.name and "cache" not in p.name)
    if not arrow_files:
        raise FileNotFoundError(f"No arrow shards under {tokenized_dir}")

    keep = list(keep_columns)
    parts = []
    for p in arrow_files:
        ds = datasets.Dataset.from_file(p)
        missing = [c for c in keep if c not in ds.column_names]
        if missing:
            raise ValueError(f"Arrow shard {Path(p).name} missing columns {missing}. Available (sample): {list(ds.column_names)[:25]}...")
        parts.append(ds.select_columns(keep))

    return concatenate_datasets(parts)


def strip_ensembl_version_series(col: pd.Series) -> pd.Series:
    """Strip Ensembl version suffixes (ENSG....12 -> ENSG....) so keys match the tokenizer dictionary."""
    ss = col.astype(str)
    mask = ss.str.startswith("ENS", na=False) & ss.str.contains(".", regex=False, na=False)
    stripped = ss.str.split(".", n=1, regex=False).str[0]
    return stripped.where(mask, ss)


def map_genes_to_token_ids(de_df: pd.DataFrame, gene_col: str, var_df: pd.DataFrame, token_dictionary: dict) -> pd.DataFrame:
    """Map DE/OpenTargets gene names (or Ensembl IDs) to tokenizer token IDs, dropping unmapped genes."""
    genes = strip_ensembl_version_series(de_df[gene_col])
    names_are_ensembl = genes.str.startswith("ENS").mean() > 0.5

    if names_are_ensembl:
        m = de_df.copy()
        m["ensembl_gene_id"] = genes
        m["token_ids"] = m["ensembl_gene_id"].replace(token_dictionary)
    else:
        m = pd.merge(de_df.rename(columns={gene_col: "gene_name"}), var_df[["gene_name", "ensembl_gene_id"]], on="gene_name", how="left")
        m["ensembl_gene_id"] = strip_ensembl_version_series(m["ensembl_gene_id"])
        m["token_ids"] = m["ensembl_gene_id"].replace(token_dictionary)

    bad = m["token_ids"].astype(str).str.startswith("ENS", na=False)
    n_mapped = (~bad).sum()
    print(f"  [map_genes_to_token_ids] {n_mapped} / {len(m)} genes successfully mapped to token IDs")
    return m[~bad].copy()


def _cell_disease_equals(example: dict, label: str) -> bool:
    return str(example.get("disease")) == str(label)


def var_df_for_merge(adata: sc.AnnData) -> pd.DataFrame:
    """adata.var as a flat DataFrame with one gene_name column for merges."""
    v = adata.var.copy()
    if v.index.name == "gene_name" and "gene_name" in v.columns:
        return v.drop(columns=["gene_name"]).reset_index()
    if v.index.name == "gene_name":
        return v.reset_index()
    if "gene_name" in v.columns:
        return v.copy()
    return v.reset_index()


def disease_slug(label: str) -> str:
    return str(label).replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def compute_de_multigroup_vs_reference(adata: sc.AnnData, disease_key: str, reference_label: str) -> pd.DataFrame:
    """For each non-reference disease, DE genes vs reference_label (e.g. 'normal')."""
    if disease_key not in adata.obs:
        raise KeyError(f"{disease_key=} not in adata.obs")
    ref = str(reference_label)
    cats = adata.obs[disease_key].astype(str).unique()
    if ref not in set(cats):
        raise ValueError(f"Reference {ref!r} not in obs['{disease_key}']. Unique: {sorted(cats)}")
    sc.tl.rank_genes_groups(adata, disease_key, method="wilcoxon", reference=ref)
    de_df = sc.get.rank_genes_groups_df(adata, group=None)
    de_df["regulation"] = np.where(de_df["logfoldchanges"] > 0, "Upregulated", "Downregulated")
    return de_df


def compute_de_disease_vs_healthy(adata: sc.AnnData, disease_key: str, healthy_labels: Sequence[str]) -> pd.DataFrame:
    """Binary contrast: healthy vs disease (all non-healthy labels pooled)."""
    ad = adata.copy()
    healthy_set = set(str(x) for x in healthy_labels)
    ad.obs["__case__"] = ad.obs[disease_key].map(lambda x: "healthy" if str(x) in healthy_set else "disease")
    n_h, n_d = (ad.obs["__case__"] == "healthy").sum(), (ad.obs["__case__"] == "disease").sum()
    if n_h == 0 or n_d == 0:
        raise ValueError(f"Need both healthy and disease cells; counts healthy={n_h}, disease={n_d}. Adjust healthy_labels (see --inspect).")

    sc.tl.rank_genes_groups(ad, "__case__", method="wilcoxon", reference="healthy")
    de_df = sc.get.rank_genes_groups_df(ad, group="disease")
    de_df["regulation"] = np.where(de_df["logfoldchanges"] > 0, "Upregulated", "Downregulated")
    return de_df


def load_open_targets_genes(ot_path: Path, min_global_score: float) -> pd.DataFrame:
    ot = pd.read_csv(ot_path, sep="\t")
    if "globalScore" not in ot.columns:
        raise ValueError(f"Expected column globalScore in {ot_path}")
    return ot.loc[ot["globalScore"] >= min_global_score].copy()


def inspect_obs(h5ad_path: Path, disease_key: str) -> None:
    ad = sc.read_h5ad(h5ad_path)
    print(f"Columns (first 30): {list(ad.obs.columns)[:30]}")
    if disease_key not in ad.obs:
        print(f"Missing obs['{disease_key}']. Available: {list(ad.obs.columns)}")
        return
    print(f"\nobs['{disease_key}'].value_counts():\n{ad.obs[disease_key].astype(str).value_counts()}\n")
    print("Set CONFIG[...]['healthy_labels'] to the exact strings that mean healthy/control.")


def run_open_targets(merged: datasets.Dataset, ot_df_mapped: pd.DataFrame, thresholds: Sequence[int],
                      output_dir: Path, tag: str, num_proc: Optional[int], resume: bool) -> dict:
    genes = set(ot_df_mapped["token_ids"])
    return retention_curve_with_checkpoints(merged, genes, thresholds, "upregulated", num_proc, output_dir / f"OT_{tag}.pkl", resume, desc="OpenTargets")


def parse_args():
    p = argparse.ArgumentParser(description="DE / OpenTargets retention vs sequence truncation.")
    p.add_argument("--dataset", default="LUCA", choices=list(DEFAULT_CONFIG.keys()))
    p.add_argument("--inspect", action="store_true", help="Print disease value_counts and exit.")
    p.add_argument("--mode", choices=("de", "ot", "both"), default="both")
    p.add_argument("--healthy-label", action="append", default=None, help="Exact healthy label(s) (repeat flag). Overrides config list.")
    p.add_argument("--de-cell-subset", choices=("disease", "healthy", "all"), default="disease",
                    help="Cells for retention curves: disease=cells matching that condition; healthy=reference only; all=all cells.")
    p.add_argument("--diseases", nargs="*", default=None, help="If set (exact obs['disease'] strings), only run these conditions in per-disease mode.")
    p.add_argument("--threshold-min", type=int, default=100)
    p.add_argument("--threshold-max", type=int, default=10101, help="Exclusive upper bound (Python range end).")
    p.add_argument("--threshold-step", type=int, default=1111)
    p.add_argument("--ot-min-score", type=float, default=0.01)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-dir", type=Path, default=None)
    p.add_argument("--num-proc", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="For each retention .pkl, skip thresholds already stored (safe to restart).")
    p.add_argument("--arrow-keep-cols", nargs="*", default=None, help="Default: input_ids disease cell_type.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG[args.dataset].copy()
    h5ad, token_dir, output_dir, token_dict_path = Path(cfg["h5ad"]), Path(cfg["tokenized_dir"]), Path(cfg["output_dir"]), Path(cfg["tokenizer_dict"])
    disease_key = cfg["disease_key"]
    healthy_labels: List[str] = list(args.healthy_label) if args.healthy_label else list(cfg.get("healthy_labels", []))

    global HEALTHY_SET
    HEALTHY_SET = set(str(x) for x in healthy_labels)

    if args.inspect:
        inspect_obs(h5ad, disease_key)
        return
    if not HEALTHY_SET:
        raise SystemExit("Set healthy labels in DEFAULT_CONFIG or pass --healthy-label EXACT (use --inspect).")

    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = list(range(args.threshold_min, args.threshold_max, args.threshold_step))
    if not thresholds:
        raise SystemExit("Empty threshold grid: adjust --threshold-min/--threshold-max/--threshold-step.")
    print(f"Retention grid: {len(thresholds)} thresholds {thresholds[0]} .. {thresholds[-1]} (step {args.threshold_step})")

    genedata = sc.read_h5ad(h5ad)
    with open(token_dict_path, "rb") as f:
        token_dictionary = pickle.load(f)
    var_df = var_df_for_merge(genedata)

    keep_cols = tuple(args.arrow_keep_cols) if args.arrow_keep_cols else ("input_ids", "disease", "cell_type")
    merged = load_merged_arrow_dataset(token_dir, keep_columns=keep_cols)
    print(f"Loaded tokenized dataset: {len(merged)} rows")

    per_disease = bool(cfg.get("per_disease_analysis", False))
    ref_label = str(healthy_labels[0])
    disease_allow: Optional[Set[str]] = set(args.diseases) if args.diseases else None

    if per_disease:
        if args.mode in ("ot", "both"):
            ot_dir = Path(cfg.get("ot_dir", output_dir))
            per_ot: Dict[str, str] = dict(cfg.get("per_disease_ot_files") or {})
            if not per_ot:
                print("per_disease_analysis True but per_disease_ot_files empty; skip OT.")
            for disease_name, fname in per_ot.items():
                if disease_allow is not None and disease_name not in disease_allow:
                    continue
                ot_path = ot_dir / fname
                if not ot_path.is_file():
                    print(f"Skip OT {disease_name!r}: missing file {ot_path}")
                    continue
                ot_df = load_open_targets_genes(ot_path, args.ot_min_score)
                ot_m = pd.merge(ot_df.rename(columns={"symbol": "gene_name"}), var_df[["gene_name", "ensembl_gene_id"]], on="gene_name", how="inner")
                ot_m["ensembl_gene_id"] = strip_ensembl_version_series(ot_m["ensembl_gene_id"])
                ot_m["token_ids"] = ot_m["ensembl_gene_id"].replace(token_dictionary)
                ot_m = ot_m[~ot_m["token_ids"].astype(str).str.startswith("ENS", na=False)]
                ds_ot = merged.filter(partial(_cell_disease_equals, label=disease_name))
                if len(ds_ot) == 0:
                    print(f"Skip OT {disease_name!r}: 0 cells in tokenized data with that disease label.")
                    continue
                run_open_targets(ds_ot, ot_m, thresholds, output_dir, tag=f"{args.dataset}_{disease_slug(disease_name)}_OT_cells_matching", num_proc=args.num_proc, resume=args.resume)

        if args.mode in ("de", "both"):
            de_full = compute_de_multigroup_vs_reference(genedata, disease_key, ref_label)
            de_full.to_csv(output_dir / f"{args.dataset}_DE_each_vs_{disease_slug(ref_label)}.csv", index=False)

            for g in sorted(pd.unique(de_full["group"].astype(str))):
                if disease_allow is not None and g not in disease_allow:
                    continue
                de_m = map_genes_to_token_ids(de_full.loc[de_full["group"].astype(str) == g].copy(), "names", var_df, token_dictionary)

                if args.de_cell_subset == "all":
                    ds_de = merged
                elif args.de_cell_subset == "healthy":
                    ds_de = merged.filter(_is_healthy_cell)
                else:
                    ds_de = merged.filter(partial(_cell_disease_equals, label=g))
                if len(ds_de) == 0:
                    print(f"  Skip DE group {g!r}: 0 cells for subset={args.de_cell_subset}")
                    continue

                slug = disease_slug(g)
                for direction, genes, analysis in [
                    ("up", set(de_m.loc[de_m["regulation"] == "Upregulated", "token_ids"]), "upregulated"),
                    ("down", set(de_m.loc[de_m["regulation"] == "Downregulated", "token_ids"]), "downregulated"),
                ]:
                    if not genes:
                        continue
                    path = output_dir / f"DE_{args.dataset}_{slug}_vs_{disease_slug(ref_label)}_{args.de_cell_subset}_{direction}.pkl"
                    retention_curve_with_checkpoints(ds_de, genes, thresholds, analysis, args.num_proc, path, args.resume, desc=f"DE_{slug}_{direction}")
    else:
        # Pooled binary DE + a single OpenTargets file (no per-disease split).
        if args.mode in ("ot", "both"):
            ot_path = Path(cfg["ot_tsv"])
            if not ot_path.is_file():
                print(f"Skip OT: missing file {ot_path}")
            else:
                ot_df = load_open_targets_genes(ot_path, args.ot_min_score)
                ot_m = pd.merge(ot_df.rename(columns={"symbol": "gene_name"}), var_df[["gene_name", "ensembl_gene_id"]], on="gene_name", how="inner")
                ot_m["ensembl_gene_id"] = strip_ensembl_version_series(ot_m["ensembl_gene_id"])
                ot_m["token_ids"] = ot_m["ensembl_gene_id"].replace(token_dictionary)
                ot_m = ot_m[~ot_m["token_ids"].astype(str).str.startswith("ENS", na=False)]
                if args.de_cell_subset == "all":
                    ds_ot = merged
                elif args.de_cell_subset == "healthy":
                    ds_ot = merged.filter(_is_healthy_cell)
                else:
                    ds_ot = merged.filter(lambda x: not _is_healthy_cell(x))
                run_open_targets(ds_ot, ot_m, thresholds, output_dir, tag=f"{args.dataset}_subset_{args.de_cell_subset}", num_proc=args.num_proc, resume=args.resume)

        if args.mode in ("de", "both"):
            de_df = compute_de_disease_vs_healthy(genedata, disease_key, healthy_labels)
            de_df.to_csv(output_dir / f"{args.dataset}_DE_disease_vs_healthy.csv", index=False)
            de_m = map_genes_to_token_ids(de_df, "names", var_df, token_dictionary)

            if args.de_cell_subset == "all":
                ds_de = merged
            elif args.de_cell_subset == "healthy":
                ds_de = merged.filter(_is_healthy_cell)
            else:
                ds_de = merged.filter(lambda x: not _is_healthy_cell(x))

            prefix = f"{args.dataset}_disease_vs_healthy"
            for direction, genes, analysis in [
                ("up", set(de_m.loc[de_m["regulation"] == "Upregulated", "token_ids"]), "upregulated"),
                ("down", set(de_m.loc[de_m["regulation"] == "Downregulated", "token_ids"]), "downregulated"),
            ]:
                if not genes:
                    continue
                path = output_dir / f"DE_{prefix}_{args.de_cell_subset}_{direction}.pkl"
                retention_curve_with_checkpoints(ds_de, genes, thresholds, analysis, args.num_proc, path, args.resume, desc=f"DE_{direction}")

    if args.plot:
        plot_dir = Path(args.plot_dir or output_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        for pkl in sorted(output_dir.glob("*.pkl")):
            if "DE_" not in pkl.name and "OT_" not in pkl.name:
                continue
            with open(pkl, "rb") as f:
                d = pickle.load(f)
            xs = sorted(d.keys())
            plt.figure(figsize=(9, 5))
            plt.plot(xs, [d[k] for k in xs], marker="o", ms=3)
            plt.xlabel("Sequence length L (tokens kept from start [up] or end [down])")
            plt.ylabel("Mean % DE / OT genes retained (unweighted over cells)")
            plt.title(pkl.stem)
            plt.grid(True, alpha=0.3)
            plt.savefig(plot_dir / f"{pkl.stem}.png", dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved {plot_dir / f'{pkl.stem}.png'}")


if __name__ == "__main__":
    main()
