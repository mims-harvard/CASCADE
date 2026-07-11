"""
CASCADE model: context-aware transformer encoder, context-specific projection
heads, and the context-specific contrastive pre-training objective.

Corresponds to Methods 9.5-9.7 and 9.10 of the CASCADE paper (cell
representation, context-specific representation learning, context-specific
contrastive learning).
"""
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cascade.data import sampler as sampler_contrastive
from cascade.data.distributed_sampler import DistributedContrastiveSampler
from cascade.training.domain_adaptation import SinkhornDistance
from preprocessing import seattle_ad_metadata


def list_positive_labels(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return {key: locs for key, locs in tally.items()}


def contrastive_labels(data: Dict[str, torch.Tensor], context):
    """Pairwise positive-pair indicator matrix for `context` (Methods 9.10, a^(k)_ij)."""
    positive_dictionary = list_positive_labels(data[context])

    positive_indeces = []
    for key, value in positive_dictionary.items():
        res = [[a, b] for idx, a in enumerate(value) for b in value[idx + 1:]]
        res2 = [[b, a] for idx, a in enumerate(value) for b in value[idx + 1:]]
        positive_indeces.extend(res + res2)

    # Self-pairs are positive by definition
    diagonal = torch.arange(len(data[context])).reshape(len(data[context]), 1).expand(-1, 2)
    pos_indeces = torch.tensor(positive_indeces + diagonal.tolist())

    target = torch.zeros(len(data[context]), len(data[context]))
    target[pos_indeces[:, 0], pos_indeces[:, 1]] = 1
    return target


def labels_batch(data):
    """Assemble the per-anchor target matrix using each cell's own augmentation context."""
    labels = {}
    if 'cell_type' in data.keys():
        labels["CELLS"] = contrastive_labels(data, "cell_type")
    if 'disease' in data.keys():
        labels["DISEASE"] = contrastive_labels(data, "disease")
    if 'tissue' in data.keys():
        labels["TISSUE"] = contrastive_labels(data, "tissue")

    targets = []
    for i, ctx in enumerate(data["context"]):
        targets.append(labels[ctx][i].numpy())

    return torch.tensor(np.array(targets))


def context_cce(data, xcs, target_pos, target_neg, cell1, target, device, context_specific_projections=False):
    """Balanced binary cross-entropy contrastive loss per context (Methods 9.10, L^(k)_i)."""
    context_index = list_positive_labels(data["context"])
    losses_ctx = {}

    for context in np.unique(data["context"]):
        ctx_idx = context_index[context]

        if context_specific_projections:
            xcs_sub = xcs[context][ctx_idx].to(device)
        else:
            xcs_sub = xcs[ctx_idx].to(device)

        target_ctx = target[ctx_idx].to(device)
        target_pos_sub = target_pos[ctx_idx].to(device)
        target_neg_sub = target_neg[ctx_idx].to(device)

        loss = F.binary_cross_entropy((xcs_sub).sigmoid(), target_ctx, reduction="none").to(device)

        loss_pos = torch.zeros(cell1[ctx_idx].size(0), cell1.size(0)).to(device)
        loss_pos = loss_pos.masked_scatter(target_pos_sub, loss[target_pos_sub]).to(device)

        loss_neg = torch.zeros(cell1[ctx_idx].size(0), cell1.size(0)).to(device)
        loss_neg = loss_neg.masked_scatter(target_neg_sub, loss[target_neg_sub]).to(device)

        loss_pos = loss_pos.sum(dim=1)
        loss_neg = loss_neg.sum(dim=1)

        num_pos = target_ctx.sum(dim=1)
        num_neg = target_ctx.size(1) - num_pos

        if 0 in num_pos or 0 in num_neg:
            print("The batch has no positive or no negative samples for some cells - skipping - sampler must have an error")
            return None

        losses_ctx[context] = ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()

    return losses_ctx


def detect_labels_class(data, context, cell_type=None, dataset='AUTISM'):
    if context == "disease":
        labels_dis = data._data['disease']
        class_positive = "ASD" if dataset == 'AUTISM' else cell_type
        return (np.array(labels_dis) == class_positive).astype(int)

    if context == "cell_type":
        labels_ct = data._data['cell_type']
        return (np.array(labels_ct) == cell_type).astype(int)

    if context == 'tissue':
        labels_tis = data._data['tissue']
        class_positive = "ACC" if dataset == 'AUTISM' else cell_type
        return (np.array(labels_tis) == class_positive).astype(int)


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self._data = data

        if 'donor_id' not in self._data.keys():
            self._data['donor_id'] = data.pop('Donor')
        if "development_stage" in self._data.keys():
            self._data["age_binned"] = seattle_ad_metadata.bin_age(
                seattle_ad_metadata.replacement_age, self._data
            )

        if 'cell_type' in data.keys():
            self.cell_type = data["cell_type"]
        if 'disease' in data.keys():
            self.disease = data["disease"]
        if 'tissue' in data.keys():
            self.tissue = data["tissue"]
        # Context used to rank genes for this sample's augmented view
        self.context = data["context"]

    def __len__(self):
        return self._data["input_ids"].shape[0]

    def add_label(self, label_all, label_dis, label_ct):
        self.class_y = label_all
        self.y_disease = label_dis
        self.y_celltype = label_ct
        self._data["class_y"] = self.class_y

    def integer_encoding(self, label):
        self._data["class_y"] = self._data[label]

    def clean_cause_of_death(self):
        self._data['Cause_of_death'] = self._data['Cause_of_death'].map(seattle_ad_metadata.cause_of_death_mapping)
        conditions = list(set(self._data['Cause_of_death'].dropna()))

        binary_columns = self._data.apply(
            seattle_ad_metadata.generate_binary_columns, axis=1, conditions=conditions, column='Cause_of_death'
        )
        binary_df = pd.DataFrame(binary_columns.tolist(), columns=conditions)
        self._data.update(binary_df.to_dict(orient='list'))

    def filter_nonbinary(self):
        # Drop rows with a missing label, then balance classes by subsampling to the smallest class
        filter_nan = self._data["class_y"] != "nan"
        filter_nan &= ~pd.isna(self._data["class_y"])
        filtered_data = {k: np.array(v)[filter_nan] for k, v in self._data.items()}

        unique_classes = np.unique(filtered_data["class_y"])
        class_data = {}
        for cls in unique_classes:
            filter_class = filtered_data["class_y"] == cls
            class_data[cls] = {k: np.array(v)[filter_class] for k, v in filtered_data.items()}

        min_samples = min(len(class_data[cls]["input_ids"]) for cls in class_data)
        for cls in class_data:
            idx = random.sample(range(len(class_data[cls]["input_ids"])), min_samples)
            for k, v in class_data[cls].items():
                class_data[cls][k] = np.array(v)[idx]

        merged = {}
        for key in class_data[next(iter(class_data))].keys():
            mergedarray = []
            for cls in class_data:
                mergedarray.extend(class_data[cls][key].tolist())
            merged[key] = np.array(mergedarray)

        self._data = merged

    def split_by_donors_id(self, donor_dict=None):
        if 'donor_id' in self._data.keys():
            if donor_dict is None:
                donors_labels = np.unique(self._data["donor_id"])
                random.seed(42)
                train_ids = random.sample(set(donors_labels), k=np.round(0.8 * len(donors_labels)).astype(int))
                val_ids = [ids for ids in donors_labels if ids not in train_ids]
            else:
                train_ids = donor_dict['train_donors']
                val_ids = donor_dict['test_donors']
        else:
            random.seed(42)
            n_elements = len(self._data["context"])
            train_ids, val_ids = train_test_split(np.arange(n_elements), test_size=0.2, random_state=42)

        self.train_ids = train_ids
        self.val_ids = val_ids
        print(f'In total there are {len(train_ids)} in the training set and {len(val_ids)} in the val set')

    def split_by_donors(self, train):
        if 'donor_id' in self._data.keys():
            if train:
                self._data = {
                    k: np.array(v)[pd.Series(self._data["donor_id"]).isin(self.train_ids)]
                    for k, v in self._data.items()
                }
                print("Train:", self._data["input_ids"].shape)
            else:
                self._data = {
                    k: np.array(v)[pd.Series(self._data["donor_id"]).isin(self.val_ids)]
                    for k, v in self._data.items()
                }
                print("Test:", self._data["input_ids"].shape)
        else:
            if train:
                self._data = {key: np.array(values)[self.train_ids] for key, values in self._data.items()}
            else:
                self._data = {key: np.array(values)[self.val_ids] for key, values in self._data.items()}

    def filter_data(self):
        # Balance a binary classification task by subsampling the majority class
        filter_pos = self._data["class_y"] == 1
        filter_neg = self._data["class_y"] == 0
        pos = {k: np.array(v)[filter_pos] for k, v in self._data.items()}
        neg = {k: np.array(v)[filter_neg] for k, v in self._data.items()}

        minimum = min(len(pos["input_ids"]), len(neg["input_ids"]))
        idx_pos = random.sample(range(len(pos["input_ids"])), minimum)
        idx_neg = random.sample(range(len(neg["input_ids"])), minimum)

        pos_sub = {k: np.array(v)[idx_pos] for k, v in pos.items()}
        pos_neg = {k: np.array(v)[idx_neg] for k, v in neg.items()}
        assert len(pos_sub["input_ids"]) == len(pos_neg["input_ids"])

        merged = {}
        for key in pos_sub.keys():
            mergedarray = list(pos_sub[key]) + list(pos_neg[key])
            merged[key] = np.array(mergedarray)

        self._data = merged

    def __getitem__(self, idx):
        if isinstance(idx, list) and len(idx) > 0 and isinstance(idx[0], list):
            idx = idx[0]

        di = {}
        for k, v in self._data.items():
            if isinstance(v, list):
                v = np.array(v)
            val = v[idx]
            if isinstance(val, (np.generic, int, float)):
                di[k] = torch.tensor(val)
            else:
                di[k] = val
        return di


def prepare_dataloader(
    data_pt,
    batch_size: int,
    class_label,
    condition_fn=None,
    shuffle: bool = False,
    num_workers: int = 0,
    sampler: bool = True,
) -> DataLoader:
    if sampler:
        if dist.is_available() and dist.is_initialized():
            condition_sampler = DistributedContrastiveSampler(
                data_source=data_pt,
                condition_fn=condition_fn,
                batch_size=batch_size,
                shuffle=shuffle,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                pad_to_even=True,
            )
        else:
            condition_sampler = sampler_contrastive.ContrastiveLearningSampler(
                data_source=data_pt,
                condition_fn=condition_fn,
                batch_size=batch_size,
                shuffle=shuffle,
            )

        condition_sampler_batch = sampler_contrastive.IdentityBatchSampler(
            sampler=condition_sampler, batch_size=1, drop_last=False
        )
        return DataLoader(
            dataset=data_pt,
            num_workers=num_workers,
            pin_memory=False,
            batch_sampler=condition_sampler_batch,
            collate_fn=sampler_contrastive.my_collate,
        )

    if class_label in ["disease", "cell_type", "tissue"]:
        condition_sampler = sampler_contrastive.ClassificationSampler(
            data_source=data_pt, batch_size=batch_size, shuffle=shuffle
        )
    else:
        condition_sampler = sampler_contrastive.ClassificationSamplerMulti(
            data_source=data_pt, batch_size=batch_size, shuffle=shuffle
        )

    condition_sampler_batch = sampler_contrastive.IdentityBatchSampler(
        sampler=condition_sampler, batch_size=1, drop_last=False
    )
    return DataLoader(
        dataset=data_pt,
        num_workers=num_workers,
        pin_memory=False,
        batch_sampler=condition_sampler_batch,
        collate_fn=sampler_contrastive.my_collate,
    )


class GeneEncoder(nn.Module):
    """Shared learnable token embedding + LayerNorm (Methods 9.5, W_emb)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        return self.enc_norm(x)


class PositionalEncoder(nn.Module):
    """Fixed sinusoidal positional encoding (Methods 9.5)."""

    def __init__(self, d_model, max_seq_len=8000):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class BinaryClassificationDecoder(nn.Module):
    """Downstream classification head trained on frozen or fine-tuned cell embeddings."""

    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3, activation: callable = nn.ReLU):
        super().__init__()
        self._decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
        self.out_layer = nn.Linear(d_model, n_cls)
        self.softmax = nn.Softmax()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._decoder:
            x = layer(x)
        return self.softmax(self.out_layer(x))


class MLP(nn.Module):
    """Context-specific projection head f_theta_k (Methods 9.7)."""

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class TransformerGenerator(nn.Module):
    """
    CASCADE encoder: shared transformer over context-aware gene token sequences,
    context-specific projection heads, and the context-specific contrastive
    pre-training loss with optional Sinkhorn domain adaptation (Methods 9.5-9.7, 9.10).
    """

    def __init__(
        self,
        d_model,
        nhead,
        ntoken,
        dim_embedding,
        nlayers,
        vocab: Any,
        nclass,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        cell_emb_style: str = "cls",
        context_specific_projections: bool = False,
        constant_ctx=False,
        only_contrastive=False,
        extract_embeddings=False,
        DA=False,
        lambda_sinkhorn=0.005,
        merged_contexts='disease_cell_type_tissue',
    ):
        super().__init__()

        self.ntoken = ntoken
        self.pad_token_id = vocab[pad_token]
        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')
        self.extract_embeddings = extract_embeddings
        self.lambda_sinkhorn = lambda_sinkhorn
        self.cell_emb_style = cell_emb_style
        self.constant_ctx = constant_ctx
        self.DA = DA

        if self.cell_emb_style not in ["cls", "avg-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        self.encoder = GeneEncoder(ntoken, dim_embedding, padding_idx=vocab[pad_token])
        self.pos_encoder = PositionalEncoder(d_model)
        self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_embedding, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.only_contrastive = only_contrastive
        if not self.only_contrastive:
            self.decoder_class = BinaryClassificationDecoder(d_model, n_cls=nclass, nlayers=6)

        self.context_specific_projections = context_specific_projections
        if self.context_specific_projections:
            mlp_args = {
                "input_size": dim_embedding,
                "hidden_size": dim_embedding,
                "output_size": dim_embedding,
                "num_layers": 3,
            }
            contexts_by_merge = {
                'disease_cell_type_tissue': ["CELLS", "TISSUE", "DISEASE"],
                'disease_cell_type': ["CELLS", "DISEASE"],
                'cell_type_tissue': ["CELLS", "TISSUE"],
            }[merged_contexts]
            self.context_projectors = torch.nn.ModuleDict({k: MLP(**mlp_args) for k in contexts_by_merge})

    def _get_ctx_emb_from_layer(self, layer_output: Tensor) -> torch.Tensor:
        return layer_output[:, 0, :]

    def _get_cell_emb_from_layer(self, layer_output: Tensor) -> torch.Tensor:
        if self.cell_emb_style == "cls":
            return layer_output[:, 2, :]
        return torch.mean(layer_output, dim=1)

    def _encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        src_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        src = self.encoder(src.long())
        src = self.pos_encoder(src)
        total_embs = self.bn(src.permute(0, 2, 1)).permute(0, 2, 1)
        return self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask, mask=src_mask, is_causal=is_causal
        )

    def forward(
        self,
        data,
        src: Tensor,
        src_key_padding_mask: Tensor,
        temperature,
        device,
        nclass=2,
        CCE=False,
        CLASS=True,
        n_batch=2,
        n_epoch=1,
        return_overall_embeddings=False,
    ) -> Mapping[str, Tensor]:
        transformer_output = self._encode(src=src, src_key_padding_mask=src_key_padding_mask, is_causal=False)
        cell_emb = self._get_cell_emb_from_layer(transformer_output)

        output_losses = {}
        cell_emb_save = {}
        soft_out = None

        if self.training:
            if CCE and not self.extract_embeddings:
                cell1 = cell_emb
                ctx_emb = self._get_ctx_emb_from_layer(transformer_output)

                cell_emb_save["embedding"] = cell_emb
                cell_emb_save["ctx_embedding"] = ctx_emb
                cell_emb_save["disease"] = data["disease"]
                cell_emb_save["context"] = data["context"]

                if self.context_specific_projections:
                    # NOTE: this loops over context spaces rather than batching them,
                    # which keeps per-context gradient sizes constant when splitting
                    # across multiple GPUs.
                    xcs = {}
                    for k in ["CELLS", "DISEASE", "TISSUE"]:
                        if k not in self.context_projectors.keys():
                            continue
                        context_emb = self.context_projectors[k](cell1)

                        if self.DA and k == 'DISEASE':
                            if n_batch < 50 and n_epoch < 5:
                                output_losses['DA'] = 0
                            else:
                                if np.issubdtype(np.array(data['donor_id']).dtype, np.number):
                                    donors_id_tensors = torch.tensor(data['donor_id'], dtype=torch.long, device=cell1.device)
                                else:
                                    _, donor_id_ints = np.unique(data['donor_id'], return_inverse=True)
                                    donors_id_tensors = torch.tensor(donor_id_ints, dtype=torch.long, device=cell1.device)
                                unique_donors = donors_id_tensors.unique()

                                sinkhorn_loss = torch.tensor(0.0, device=cell1.device, requires_grad=True)
                                context_emb = self.context_projectors["DISEASE"](cell1)
                                for i in range(len(unique_donors)):
                                    for j in range(i + 1, len(unique_donors)):
                                        d1_embeddings = context_emb[donors_id_tensors == unique_donors[i]]
                                        d2_embeddings = context_emb[donors_id_tensors == unique_donors[j]]
                                        if d1_embeddings.size(0) > 0 and d2_embeddings.size(0) > 0:
                                            sinkhorn_loss = sinkhorn_loss + self.sink(
                                                d1_embeddings.unsqueeze(0), d2_embeddings.unsqueeze(0)
                                            )

                                output_losses['DA'] = self.lambda_sinkhorn * sinkhorn_loss

                        context_sim = F.cosine_similarity(context_emb[None, :, :], context_emb[:, None, :], dim=-1)
                        context_sim[torch.eye(context_emb.size(0)).bool()] = float("inf")
                        xcs[k] = context_sim / temperature
                else:
                    xcs = F.cosine_similarity(cell1[None, :, :], cell1[:, None, :], dim=-1)
                    # sigmoid(inf) = 1.0, so this correctly zeroes out the diagonal's contribution to the BCE loss
                    xcs[torch.eye(cell1.size(0)).bool()] = float("inf")
                    xcs = xcs / temperature

                target = labels_batch(data).to(device)
                target_pos = target.bool()
                target_neg = ~target_pos

                nt_bxent_loss_ctx_dict = context_cce(
                    data, xcs, target_pos, target_neg,
                    cell1.to(device), target, device,
                    context_specific_projections=self.context_specific_projections,
                )

                if nt_bxent_loss_ctx_dict is not None:
                    output_losses.update(nt_bxent_loss_ctx_dict)
                else:
                    output_losses = None

            if CLASS:
                soft_out = self.decoder_class(cell_emb)

            return output_losses, cell_emb_save, soft_out, None, None

        else:
            if CLASS:
                return self.decoder_class(cell_emb)
            elif CCE and self.extract_embeddings:
                if self.context_specific_projections:
                    xcs = {}
                    for k in ["CELLS", "DISEASE", "TISSUE"]:
                        if k not in self.context_projectors.keys():
                            continue
                        xcs[k] = self.context_projectors[k](cell_emb)
                    if return_overall_embeddings:
                        return xcs, cell_emb
                    return xcs
                return cell_emb
