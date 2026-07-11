"""
Custom batch samplers implementing the contrastive batching constraint from
Methods 9.10 / Supplementary Note 8: every anchor cell in a batch must have at
least one positive and one negative comparison for its sampled context.
"""
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from torch.utils.data import BatchSampler as TorchBatchSampler
from torch.utils.data import WeightedRandomSampler


def list_positive_labels_sampler(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return {key: locs for key, locs in tally.items() if len(locs) > 1}


def contrastive_labels_sampler(data: Dict[str, torch.Tensor], context):
    positive_dictionary = list_positive_labels_sampler(data[context])

    positive_indeces = []
    for key, value in positive_dictionary.items():
        res = [[a, b] for idx, a in enumerate(value) for b in value[idx + 1:]]
        res2 = [[b, a] for idx, a in enumerate(value) for b in value[idx + 1:]]
        positive_indeces.extend(res + res2)

    diagonal = torch.arange(len(data[context])).reshape(len(data[context]), 1).expand(-1, 2)
    pos_indeces = torch.tensor(positive_indeces + diagonal.tolist())

    target = torch.zeros(len(data[context]), len(data[context]))
    target[pos_indeces[:, 0], pos_indeces[:, 1]] = 1
    return target


def labels_batch_sampler(data):
    labels = {}
    if 'disease' in data.keys():
        labels["DISEASE"] = contrastive_labels_sampler(data, "disease")
    if 'tissue' in data.keys():
        labels["TISSUE"] = contrastive_labels_sampler(data, "tissue")
    if 'cell_type' in data.keys():
        labels["CELLS"] = contrastive_labels_sampler(data, "cell_type")

    targets = []
    for i, ctx in enumerate(data["context"]):
        targets.append(labels[ctx][i].numpy())

    return torch.tensor(np.array(targets))


def condition_samples(batch):
    """Require at least one positive and one negative comparison per anchor (Supp. Note 8)."""
    batch = {k: np.array(v) for k, v in batch.items()}
    target = labels_batch_sampler(batch)
    num_pos = target.sum(dim=1)
    num_neg = target.size(1) - num_pos
    return 0 not in num_pos and 0 not in num_neg


def my_collate(batch):
    collated = {}
    for key in batch[0].keys():
        values = [d[key] for d in batch]
        if key == "input_ids":
            tensor_values = []
            for val in values:
                if isinstance(val, np.ndarray):
                    tensor_values.append(torch.from_numpy(val))
                elif isinstance(val, torch.Tensor):
                    tensor_values.append(val)
                else:
                    tensor_values.append(torch.tensor(val))
            collated[key] = torch.stack(tensor_values)
        else:
            collated[key] = values
    return collated


class IdentityBatchSampler(TorchBatchSampler):
    """Wraps another sampler that already yields batches, passing each one through unchanged."""

    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class ContrastiveLearningSampler(TorchBatchSampler):
    """Draws candidate batches and yields only those satisfying `condition_fn`
    (Methods 9.10 / Supp. Note 8: every anchor needs >=1 positive and >=1 negative)."""

    def __init__(self, data_source, condition_fn, batch_size, shuffle=True):
        self.data_source = data_source
        self.condition_fn = condition_fn
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.data_source)))

    def __iter__(self):
        indices = torch.randperm(len(self.data_source)).tolist() if self.shuffle else self.indices

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                condition_batch = self.data_source[batch]
                if self.condition_fn(condition_batch):
                    yield batch
                batch = []

    def __len__(self):
        return len(self.data_source) // self.batch_size


class ClassificationSampler(TorchBatchSampler):
    """Batches for single-label classification fine-tuning."""

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.data_source)))

    def __iter__(self):
        indices = torch.randperm(len(self.data_source)).tolist() if self.shuffle else self.indices

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        return len(self.data_source) // self.batch_size


class ClassificationSamplerMulti(TorchBatchSampler):
    """Batches for multi-class classification fine-tuning (no positive/negative balancing)."""

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.data_source)))

    def __iter__(self):
        indices = torch.randperm(len(self.data_source)).tolist() if self.shuffle else self.indices

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        return len(self.data_source) // self.batch_size
