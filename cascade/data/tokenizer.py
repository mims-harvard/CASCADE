"""
Context-aware transcriptome tokenizer (Methods 9.4): converts each cell's gene
expression profile into ranked up-/down-regulated gene token sequences relative to a
context-specific reference (the non-zero median expression computed by
`preprocessing/context_median_reference.py`), rather than tokenizing raw expression.

For a context k (cell type, disease, tissue, or treatment), the reference group is
all cells *not* in that context category. For each detected gene g in a cell, the
fold-change score FC_g = x_g / median_nonzero_ref(g | k) ranks the gene; genes are
then split into an up-regulated list (highest fold-change) and a down-regulated list
(lowest fold-change), each truncated to the configured sequence length and separated
by `<UP>`/`<DOWN>` direction tokens downstream (`_map_append_cls` below prepends the
`<ctx>`/`<ref>`/`<cls>` special tokens; the `<UP>`/`<DOWN>` tokens are added by the
collator). Adapted from the Geneformer tokenizer (Theodoris et al. 2023) to support
this context-conditional, multi-reference normalisation instead of a single global
gene-median reference, and to write output in batched chunks to bound memory on
million-cell datasets.

Usage:
    from cascade.data.tokenizer import TranscriptomeTokenizer
    tk = TranscriptomeTokenizer(
        output_dir=CASCADE_DATA_ROOT / "LUCA",
        dataset_name="LUCA",
        context="DISEASE",
        custom_attr_name_dict={"cell_type": "cell_type", "donor_id": "donor_id"},
    )
    tk.tokenize_data(data_directory, output_directory, output_prefix, file_format="h5ad")
"""
from __future__ import annotations

import gc
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pyarrow as py
import scipy.sparse as sp
from datasets import Dataset, disable_caching
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
logger = logging.getLogger(__name__)

CASCADE_DATA_ROOT = Path(os.environ.get("CASCADE_DATA_ROOT", "/n/holylfs06/LABS/mzitnik_lab/Lab/vgiunchiglia/DATASET"))

os.environ.setdefault("HF_DATASETS_CACHE", str(CASCADE_DATA_ROOT / ".hf_cache"))
disable_caching()

CONTEXT_TO_MEDIAN_FILE = {
    'ALL': "median_genes_all_all_{dataset}.pkl",
    'DISEASE': "median_genes_disease_all_{dataset}.pkl",
    'TISSUE': "median_genes_tissue_all_{dataset}.pkl",
    'CELLS': "median_genes_cells_all_{dataset}.pkl",
}
CONTEXT_TO_OBS_COLUMN = {
    'DISEASE': "disease",
    'TISSUE': "tissue",
    'CELLS': "cell_type_ontology_term_id",
}


def rank_genes(gene_vector, gene_tokens):
    """Sort gene tokens by descending fold-change score."""
    sorted_indices = np.argsort(-gene_vector)
    return gene_tokens[sorted_indices]


def tokenize_cell(gene_vector, gene_tokens):
    """Convert a normalized gene expression vector to a rank-value-encoded token
    sequence, masking undetected (zero-expression) genes."""
    nonzero_mask = np.nonzero(gene_vector)[0]
    return rank_genes(gene_vector[nonzero_mask], gene_tokens[nonzero_mask])


def clean_dataset_dict(d):
    """Coerce dataset_dict values to PyArrow-friendly types: NaN -> None, floats ->
    strings (mixed-type columns otherwise fail PyArrow's type inference), and the
    string sentinel 'NA' -> None."""
    for key, values in d.items():
        if key == "input_ids":
            continue
        cleaned = []
        for val in values:
            if isinstance(val, float) and np.isnan(val):
                cleaned.append(None)
            elif isinstance(val, float):
                cleaned.append(str(val))
            elif val == 'NA':
                cleaned.append(None)
            else:
                cleaned.append(val)
        d[key] = cleaned
    return d


class TranscriptomeTokenizer:
    """Tokenizes .h5ad (or .loom) files into rank-value-encoded HuggingFace Datasets,
    one context-specific augmentation at a time (Methods 9.4)."""

    def __init__(
        self,
        output_dir,
        dataset_name,
        custom_attr_name_dict=None,
        nproc=1,
        chunk_size=5000,
        batch_size=20000,
        token_dictionary_file=None,
        context="DISEASE",
        reference="out-ref",
    ):
        """
        Parameters
        ----------
        output_dir : Path
            Directory containing the `median_genes_*_{dataset_name}.pkl` reference
            files produced by `preprocessing/context_median_reference.py`.
        dataset_name : str
            Used to locate the median-reference and token-dictionary pickle files.
        custom_attr_name_dict : None, dict
            Dictionary of custom attributes to be added to the dataset. Keys are the
            names of the attributes on the input AnnData/loom; values are the names
            they should be stored under in the output dataset.
        nproc : int
            Number of processes to use for dataset mapping.
        chunk_size : int
            Number of cells processed per in-memory tokenization chunk.
        batch_size : int, optional
            Number of tokenized cells accumulated before a batch is flushed to disk
            via `save_batch`. If None, the whole file is tokenized in memory and
            written once via `create_dataset` instead (only safe for small datasets).
        token_dictionary_file : Path
            Pickle file containing the token dictionary (Ensembl IDs -> token id).
            Defaults to `output_dir / f"tokenizer_dictionary_{dataset_name}.pkl"`.
        context : {"ALL", "DISEASE", "TISSUE", "CELLS"}
            Which biological context to condition the fold-change reference on.
        reference : str
            Which reference group in the median file to use (only "out-ref" -
            cells outside the context category - is currently supported).
        """
        self.context = context
        self.dataset_name = dataset_name
        self.reference = reference
        self.custom_attr_name_dict = custom_attr_name_dict
        self.nproc = nproc
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)

        if context not in CONTEXT_TO_MEDIAN_FILE:
            raise ValueError(f"Unknown context '{context}'; expected one of {list(CONTEXT_TO_MEDIAN_FILE)}")
        self.ctx_columns = CONTEXT_TO_OBS_COLUMN.get(context)
        self.gene_median_file = self.output_dir / CONTEXT_TO_MEDIAN_FILE[context].format(dataset=dataset_name)

        token_dictionary_file = token_dictionary_file or (
            self.output_dir / f"tokenizer_dictionary_{dataset_name}.pkl")

        with open(self.gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # Gene universe for tokenization: the genes present in the median reference
        # (i.e. protein-coding + miRNA genes, per context_median_reference.py).
        self.gene_keys = list(self.gene_median_dict[list(self.gene_median_dict.keys())[0]]["out-ref"])
        self.classes = list(self.gene_median_dict.keys())
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    def tokenize_data(
        self,
        data_directory: Path | str,
        output_directory: Path | str,
        output_prefix: str,
        file_format: Literal["loom", "h5ad"] = "h5ad",
        use_generator: bool = False,
        col_ensembl: str = "ensembl_gene_id",
    ):
        """
        Tokenize files in `data_directory` and save as a HuggingFace Dataset in
        `output_directory` (or, if `batch_size` is set, as multiple batch-chunked
        datasets under `output_directory`).
        """
        if self.batch_size is None:
            tokenized_cells, cell_metadata = self._tokenize_files(Path(data_directory), file_format, col_ensembl)
            tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata, use_generator=use_generator)
            output_path = Path(output_directory) / output_prefix / self.reference / self.context
            tokenized_dataset.save_to_disk(str(output_path))
        else:
            self._tokenize_files_batched(Path(data_directory), output_directory, output_prefix, file_format, col_ensembl)

    def _tokenize_files(self, data_directory, file_format, col_ensembl):
        tokenized_cells = []
        cell_metadata = {attr: [] for attr in self.custom_attr_name_dict.values()} if self.custom_attr_name_dict else None
        tokenize_file_fn = self.tokenize_loom if file_format == "loom" else self.tokenize_anndata

        file_found = False
        for file_path in data_directory.glob(f"*.{file_format}"):
            file_found = True
            print(f"Tokenizing {file_path}")
            file_tokenized_cells, file_cell_metadata = tokenize_file_fn(file_path, col_ensembl=col_ensembl)
            tokenized_cells += file_tokenized_cells
            if self.custom_attr_name_dict is not None:
                for attr_key, out_key in self.custom_attr_name_dict.items():
                    cell_metadata[out_key] += file_cell_metadata[out_key]

        if not file_found:
            raise FileNotFoundError(f"No .{file_format} files found in directory {data_directory}.")
        return tokenized_cells, cell_metadata

    def _tokenize_files_batched(self, data_directory, output_directory, output_prefix, file_format, col_ensembl):
        tokenize_file_fn = self.tokenize_loom if file_format == "loom" else self.tokenize_anndata
        n_files = len(list(data_directory.glob(f"*.{file_format}")))
        if n_files > 1:
            print("WARNING: multiple input files found; only single-file batched tokenization is supported.")
        for file_path in data_directory.glob(f"*.{file_format}"):
            print(f"Tokenizing {file_path}")
            tokenize_file_fn(
                file_path, col_ensembl=col_ensembl,
                output_directory=output_directory, output_prefix=output_prefix,
            )

    def tokenize_anndata(self, adata_file_path, output_directory=None, output_prefix=None,
                          target_sum=10_000, col_ensembl="ensembl_gene_id"):
        adata = ad.read_h5ad(adata_file_path)

        file_cell_metadata = (
            {attr: [] for attr in self.custom_attr_name_dict.keys()} if self.custom_attr_name_dict else None)

        coding_loc = np.where([self.genelist_dict.get(i, False) for i in adata.var[col_ensembl]])[0]
        coding_ids = adata.var[col_ensembl].values[coding_loc]
        coding_tokens = np.array([self.gene_token_dict[i] for i in coding_ids])

        batched = self.batch_size is not None and output_directory is not None
        tokenized_cells = []
        counter = 0

        for class_ctx in self.classes:
            filter_pass_loc = np.where(adata.obs[self.ctx_columns].values == class_ctx)[0]
            norm_factor_vector = np.array(
                [self.gene_median_dict[class_ctx][self.reference][g] for g in adata.var[col_ensembl].values[coding_loc]])

            for i in tqdm(range(0, len(filter_pass_loc), self.chunk_size)):
                idx = filter_pass_loc[i: i + self.chunk_size]

                # Data is stored log1p-normalized (Methods 9.2); undo that here since
                # the fold-change reference median was computed on linear-scale counts.
                X_view = np.expm1(adata[idx, coding_loc].X).astype(np.float32)
                X_norm = sp.csr_matrix(X_view * target_sum / norm_factor_vector)

                tokenized_cells += [
                    rank_genes(X_norm[j].data, coding_tokens[X_norm[j].indices])
                    for j in range(X_norm.shape[0])
                ]

                if self.custom_attr_name_dict is not None:
                    for attr_key, out_key in self.custom_attr_name_dict.items():
                        file_cell_metadata[out_key] += adata[idx].obs[attr_key].tolist()

                gc.collect()
                del X_view, X_norm
                counter += 1

                if batched and (len(tokenized_cells) >= self.batch_size or i + self.chunk_size >= len(filter_pass_loc)):
                    self.save_batch(
                        tokenized_cells, file_cell_metadata, f"{class_ctx}{counter}",
                        output_directory, output_prefix,
                    )
                    tokenized_cells = []
                    if self.custom_attr_name_dict is not None:
                        file_cell_metadata = {attr: [] for attr in self.custom_attr_name_dict.keys()}

        return tokenized_cells, file_cell_metadata

    def tokenize_loom(self, loom_file_path, output_directory=None, output_prefix=None,
                       target_sum=10_000, col_ensembl="ensembl_gene_id"):
        import loompy as lp

        file_cell_metadata = (
            {attr: [] for attr in self.custom_attr_name_dict.keys()} if self.custom_attr_name_dict else None)

        with lp.connect(str(loom_file_path)) as data:
            coding_loc = np.where([self.genelist_dict.get(i, False) for i in data.ra[col_ensembl]])[0]
            norm_factor_vector = np.array([self.gene_median_dict[i] for i in data.ra[col_ensembl][coding_loc]])
            coding_ids = data.ra[col_ensembl][coding_loc]
            coding_tokens = np.array([self.gene_token_dict[i] for i in coding_ids])

            try:
                filter_pass_loc = np.where(data.ca["filter_pass"] == 1)[0]
            except AttributeError:
                print(f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells.")
                filter_pass_loc = np.arange(data.shape[1])

            tokenized_cells = []
            for _ix, _selection, view in data.scan(items=filter_pass_loc, axis=1):
                subview = view.view[coding_loc, :]
                subview_norm_array = subview[:, :] / subview.ca.n_counts * target_sum / norm_factor_vector[:, None]
                tokenized_cells += [
                    tokenize_cell(subview_norm_array[:, i], coding_tokens)
                    for i in range(subview_norm_array.shape[1])
                ]
                if self.custom_attr_name_dict is not None:
                    for attr_key, out_key in self.custom_attr_name_dict.items():
                        file_cell_metadata[out_key] += subview.ca[attr_key].tolist()

        return tokenized_cells, file_cell_metadata

    def save_batch(self, tokenized_cells, cell_metadata, batch_name, output_directory, output_prefix):
        """Write one batch of tokenized cells to disk as a HuggingFace Dataset."""
        dataset_dict = {
            "input_ids": tokenized_cells,
            "context": [self.context] * len(tokenized_cells),
            "reference": [self.reference] * len(tokenized_cells),
        }
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        dataset_dict["input_ids"] = [py.array(c) for c in dataset_dict["input_ids"]]
        dataset_dict.pop('cell_type_predicted', None)
        dataset_dict = clean_dataset_dict(dataset_dict)
        output_dataset = Dataset.from_dict(dataset_dict)
        output_dataset.cleanup_cache_files()

        def format_cell_features(example):
            example["length"] = len(example["input_ids"])
            return example

        output_dataset = output_dataset.map(format_cell_features, num_proc=self.nproc)
        output_path = Path(output_directory) / output_prefix / batch_name / f"{self.context}.dataset"
        output_dataset.save_to_disk(str(output_path))

    def create_dataset(self, tokenized_cells, cell_metadata, use_generator=False):
        """Build a single (non-batched) HuggingFace Dataset from all tokenized cells.
        Only used when `batch_size=None`; prefer `save_batch` for large datasets."""
        dataset_dict = {
            "input_ids": tokenized_cells,
            "context": [self.context] * len(tokenized_cells),
            "reference": [self.reference] * len(tokenized_cells),
        }
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        if use_generator:
            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: v[i] for k, v in dataset_dict.items()}
            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            dataset_dict["input_ids"] = [py.array(c) for c in dataset_dict["input_ids"]]
            dataset_dict.pop('cell_type_predicted', None)
            output_dataset = Dataset.from_dict(dataset_dict)

        def format_cell_features(example):
            example["length"] = len(example["input_ids"])
            return example

        return output_dataset.map(format_cell_features, num_proc=self.nproc)


def append_special_tokens(dataset: Dataset, vocab) -> Dataset:
    """Prepend the `<ctx>`, `<ref>`, and `<cls>` special tokens to every tokenized
    input sequence (Methods 9.4/9.5)."""
    return dataset.map(
        lambda example: {
            "input_ids": [vocab["<ctx>"], vocab["<ref>"], vocab["<cls>"]] + example["input_ids"],
            "length": example["length"] + 3,
        },
        num_proc=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1,
    )
