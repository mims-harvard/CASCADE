"""
Loads or extracts frozen CASCADE cell embeddings for downstream donor-level and
cell-level prediction tasks (Methods 9.11).
"""
import copy
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cascade.data import sampler as sampler_contrastive
from cascade.data import splits as donor_splits
from cascade.data.utils import load_and_merge_pickles
from cascade.explainer import config as cfg
from cascade.model import cascade_model


class EmbeddingDataLoader:
    """Loads frozen CASCADE embeddings for a dataset, either from a saved pickle
    or by running the pretrained transformer over its tokenized data. Datasets in
    `cfg.CONTEXT_AGNOSTIC_DATASETS` (e.g. HH) use pre-extracted embeddings only."""

    def __init__(self, dataset_name, model_path=None):
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.paths = cfg.get_default_paths(dataset_name)
        if model_path:
            self.paths['model_path'] = Path(model_path)

        self.is_context_agnostic = dataset_name.upper() in cfg.CONTEXT_AGNOSTIC_DATASETS

        torch.manual_seed(20)
        np.random.seed(20)
        random.seed(20)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_donor_split(self):
        """Look up the fixed donor-level train/test split for this dataset (Methods 9.11)."""
        dataset_key = self.dataset_name.upper() if isinstance(self.dataset_name, str) else self.dataset_name
        donors_split = donor_splits.SPLITS_BY_DATASET.get(dataset_key)
        if donors_split is None:
            raise ValueError(
                f"No donor split found for dataset '{self.dataset_name}' in cascade.data.splits.SPLITS_BY_DATASET"
            )
        return donors_split

    def load_dataset(self):
        print("Loading data...")
        path_to_collator = cfg.get_collator_path(self.dataset_name)

        if self.dataset_name.upper() in cfg.THREE_CONTEXT_DATASETS:
            merged_contexts = 'disease_cell_type_tissue'
        else:
            merged_contexts = cfg.MERGED_CONTEXTS

        print(f"Using merged_contexts: {merged_contexts}")
        print(f"Collator path: {path_to_collator}")

        output_data_dict = load_and_merge_pickles(
            self.paths['output_dir'], merged_context=merged_contexts, path_to_collator=path_to_collator
        )
        return cascade_model.SeqDataset(output_data_dict)

    def create_data_splits(self, output_data_dict):
        donors_split = self.get_donor_split()

        output_data_dict_train = copy.deepcopy(output_data_dict)
        output_data_dict_train.split_by_donors_id(donor_dict=donors_split)
        output_data_dict_train.split_by_donors(train=True)

        output_data_dict_val = copy.deepcopy(output_data_dict_train)
        output_data_dict_val.split_by_donors(train=False)

        return output_data_dict_train, output_data_dict_val, donors_split

    def load_token_dictionary(self):
        with open(self.paths['token_dictionary_file'], "rb") as input_file:
            return pickle.load(input_file)

    def initialize_model(self, token_dictionary):
        ntoken = len(token_dictionary)
        cascade_model.set_seed(20)

        mod = cascade_model.TransformerGenerator(
            d_model=cfg.D_MODEL,
            nhead=cfg.NHEAD,
            ntoken=ntoken,
            dim_embedding=cfg.DIM_EMBEDDING,
            nlayers=cfg.NLAYERS,
            vocab=token_dictionary,
            dropout=cfg.DROPOUT,
            pad_token=cfg.PAD_TOKEN,
            nclass=cfg.NCLASS,
            cell_emb_style=cfg.CELL_EMB_STYLE,
            extract_embeddings=True,
            context_specific_projections=cfg.CONTEXT_SPECIFIC_PROJECTIONS,
            DA=cfg.DA,
        ).to(self.device)

        try:
            model_dict = torch.load(self.paths['model_path'], map_location=torch.device('cpu'))
            mod.load_state_dict(model_dict["model_state_dict"])
            print(f"Loading all model params from {self.paths['model_path']}")
        except Exception:
            model_dict = mod.state_dict()
            pretrained_dict = torch.load(self.paths['model_path'], map_location=torch.device('cpu'))["model_state_dict"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            mod.load_state_dict(model_dict)
            print("Only subset of pre-trained model weights were loaded")

        return mod

    def create_data_loaders(self, output_data_dict_train, output_data_dict_val, output_data_dict):
        loader_train = cascade_model.prepare_dataloader(
            data_pt=output_data_dict_train, batch_size=cfg.BATCH, shuffle=True, num_workers=0,
            condition_fn=sampler_contrastive.condition_samples, sampler=False, class_label=None,
        )
        loader_val = cascade_model.prepare_dataloader(
            data_pt=output_data_dict_val, batch_size=cfg.BATCH, shuffle=True, num_workers=0,
            condition_fn=sampler_contrastive.condition_samples, sampler=False, class_label=None,
        )
        loader_all = cascade_model.prepare_dataloader(
            data_pt=output_data_dict, batch_size=cfg.BATCH, shuffle=True, num_workers=0,
            condition_fn=sampler_contrastive.condition_samples, sampler=False, class_label=None,
        )
        return loader_train, loader_val, loader_all

    def load_existing_embeddings(self, embeddings_path):
        print(f"📁 Loading existing embeddings from: {embeddings_path}")
        try:
            with open(embeddings_path, 'rb') as f:
                cell_data = pickle.load(f)
            print(f"✅ Loaded embeddings for {len(cell_data['embedding'])} cells")
            return cell_data
        except FileNotFoundError:
            print(f"❌ Embeddings file not found: {embeddings_path}")
            print("   Falling back to generating new embeddings...")
            return None
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            print("   Falling back to generating new embeddings...")
            return None

    def extract_embeddings(self, mod, loader_all, token_dictionary, save_embeddings_path=None):
        if cfg.DEBUG_MODE:
            print(f"🔧 DEBUG MODE: Extracting embeddings from first {cfg.DEBUG_MAX_BATCHES} batches only...")
        else:
            print("Extracting embeddings from all data...")

        cell_data = defaultdict(list)
        mod.eval()
        for batch_eval, batch_data_eval in enumerate(tqdm(loader_all)):
            if cfg.DEBUG_MODE and batch_eval >= cfg.DEBUG_MAX_BATCHES:
                print(f"🔧 DEBUG MODE: Stopping after {cfg.DEBUG_MAX_BATCHES} batches")
                break

            input_gene_ids_eval = torch.tensor(batch_data_eval["input_ids"]).to(self.device)
            src_key_padding_mask_eval = input_gene_ids_eval.eq(token_dictionary[cfg.PAD_TOKEN])

            with torch.no_grad():
                emb = mod(
                    data=batch_data_eval, src=input_gene_ids_eval, src_key_padding_mask=src_key_padding_mask_eval,
                    CCE=True, temperature=cfg.TEMPERATURE, device=self.device, CLASS=False, nclass=cfg.NCLASS,
                )
            torch.cuda.empty_cache()

            cell_data["embedding"].extend(emb.cpu())
            for key, value in batch_data_eval.items():
                try:
                    if isinstance(value, torch.Tensor):
                        cell_data[key].extend(value.cpu().tolist())
                    else:
                        cell_data[key].extend(value)
                except Exception as e:
                    print(f"⚠️ Skipped key {key} due to: {e}")

        cell_data["embedding"] = torch.stack(cell_data["embedding"])

        if save_embeddings_path is not None:
            self.save_embeddings(cell_data, save_embeddings_path)

        print(f"Extracted embeddings for {len(cell_data['embedding'])} cells from all data")
        return cell_data

    def save_embeddings(self, cell_data, save_path):
        try:
            serializable = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in cell_data.items()}
            with open(save_path, "wb") as sf:
                pickle.dump(serializable, sf)
            print(f"Saved embeddings and metadata to: {save_path}")
        except Exception as e:
            print(f"Warning: failed to save embeddings to {save_path}: {e}")

    def get_embeddings(self, existing_embeddings_path=None, save_embeddings_path=None):
        """Load pre-extracted embeddings if available, else run the transformer to extract them."""
        if self.is_context_agnostic:
            embeddings_path = existing_embeddings_path or self.paths.get('embeddings_path')
            print(f"🔬 Context-agnostic dataset detected - loading pre-extracted embeddings from: {embeddings_path}")
            cell_data = self.load_existing_embeddings(embeddings_path)
            if cell_data is None:
                raise FileNotFoundError(
                    f"Embeddings not found at {embeddings_path}. This dataset requires pre-extracted embeddings."
                )
            cell_data['donors_split'] = self.get_donor_split()
            return cell_data

        if existing_embeddings_path:
            cell_data = self.load_existing_embeddings(existing_embeddings_path)
            if cell_data is not None:
                return cell_data
            print(f"Could not load embeddings from provided path {existing_embeddings_path}; will attempt to generate new embeddings.")

        output_data_dict = self.load_dataset()
        output_data_dict_train, output_data_dict_val, donors_split = self.create_data_splits(output_data_dict)
        token_dictionary = self.load_token_dictionary()
        mod = self.initialize_model(token_dictionary)
        loader_train, loader_val, loader_all = self.create_data_loaders(output_data_dict_train, output_data_dict_val, output_data_dict)

        cell_data = self.extract_embeddings(mod, loader_all, token_dictionary, save_embeddings_path)
        cell_data['donors_split'] = donors_split
        return cell_data
