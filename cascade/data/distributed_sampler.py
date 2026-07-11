import math

import torch
from torch.utils.data import BatchSampler as TorchBatchSampler


class DistributedContrastiveSampler(TorchBatchSampler):
    """DDP-aware version of `sampler.ContrastiveLearningSampler`: shards indices
    across ranks before applying the positive/negative batch condition."""

    def __init__(self, data_source, condition_fn,
                 batch_size, shuffle=True, num_replicas=None, rank=None,
                 drop_last=False, pad_to_even=False):

        self.data_source = data_source
        self.condition_fn = condition_fn
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pad_to_even = pad_to_even
        self.seed = 0

        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.pad_to_even:
            self.num_samples = int((len(self.data_source) + self.num_replicas - 1) // self.num_replicas)
        else:
            if drop_last and len(self.data_source) % self.num_replicas != 0:
                self.num_samples = math.ceil((len(self.data_source) - self.num_replicas) / self.num_replicas)
            else:
                self.num_samples = math.ceil(len(self.data_source) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        if not self.pad_to_even:
            self.total_size = min(self.total_size, len(self.data_source))

        self.indices = list(range(len(self.data_source)))

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.data_source), generator=g).tolist()
        else:
            indices = self.indices

        if self.pad_to_even:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size, f"Indices are length {len(indices)} vs. total_size = {self.total_size}"

        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                condition_batch = self.data_source[batch]
                if self.condition_fn(condition_batch):
                    yield batch
                batch = []

    def __len__(self):
        effective_samples = self.total_size // self.num_replicas if self.pad_to_even else self.num_samples
        if self.drop_last:
            return effective_samples // self.batch_size
        return (effective_samples + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
