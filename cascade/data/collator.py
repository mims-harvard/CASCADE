"""
Assembles the context-aware token sequence for each cell: <UP> token, the
up-regulated gene ranking, <DOWN> token, the down-regulated gene ranking, then
padding (Methods 9.4).
"""
import pickle
from typing import List, Mapping, Optional

import torch
from tqdm import tqdm

CHUNKS = 100000


class DataCollatorContrastiveLearning:
    def __init__(
        self,
        path_to_collator,
        pad_token_id: Optional[int] = 0,
        up_token_id: int = 4,
        down_token_id: int = 5,
        max_length: Optional[int] = 2048,
        max_length_up: int = 1024,
        max_length_down: int = 1024,
        number_virtual_tokens: int = 3,
    ):
        """
        Args:
            pad_token_id: token id used for padding.
            max_length: maximum total sequence length (up + down + direction tokens).
            max_length_up / max_length_down: max genes kept in each ranked direction.
            number_virtual_tokens: leading tokens (ref/context/cls) trimmed off the
                reversed down-regulated ranking before it is re-truncated.
        """
        self.pad_token_id = pad_token_id
        self.number_virtual_tokens = number_virtual_tokens
        self.down_token_id = down_token_id
        self.up_token_id = up_token_id
        self.max_length = max_length + 2  # account for the <UP>/<DOWN> tokens
        self.max_length_down = max_length_down
        self.max_length_up = max_length_up
        self.path_to_collator = path_to_collator
        self.base_name = path_to_collator.stem

    def __call__(self, examples: List[Mapping[str, torch.Tensor]]) -> dict:
        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        device = torch.tensor(examples[0]["input_ids"]).device
        max_ori_len = 10000
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len
        self._max_length = _max_length

        padded_genes = []
        metadata_keys = [col for col in examples.column_names if col != "input_ids"]
        metadata_buffer = {key: [] for key in metadata_keys}

        for i in tqdm(range(len(examples))):
            genes = torch.tensor(examples[i]["input_ids"]).to(device)
            genes = self._sample_or_truncate_plus_pad(
                genes=genes, max_length=_max_length,
                max_length_down=self.max_length_down, max_length_up=self.max_length_up,
            )
            padded_genes.append(genes)
            row = examples[i]
            for key in metadata_keys:
                metadata_buffer[key].append(row[key])

            if CHUNKS and (i % CHUNKS == 0 and i > 0 or i == len(examples) - 1):
                padded_genes = torch.stack(padded_genes, dim=0).to(device)
                data_dict = {"input_ids": padded_genes}
                for key in metadata_keys:
                    data_dict[key] = metadata_buffer[key]

                new_base_name = f"{self.base_name}_{i}"
                path_to_collator_chunk = self.path_to_collator.with_name(new_base_name + self.path_to_collator.suffix)
                with open(path_to_collator_chunk, "wb") as f:
                    pickle.dump(data_dict, f)

                padded_genes = []
                metadata_buffer = {key: [] for key in metadata_keys}

    def _sample_or_truncate_plus_pad(
        self,
        genes: torch.LongTensor,
        max_length: int,
        max_length_down: int,
        max_length_up: int,
    ):
        # Down-regulated ranking is stored reversed so its strongest deviations
        # sit closest to the <DOWN> token, mirroring the up-regulated ordering.
        down = genes.flip(0)[:-self.number_virtual_tokens][:max_length_down]
        up = genes[:max_length_up]

        genes_out = torch.full((max_length,), self.pad_token_id, dtype=genes.dtype, device=genes.device)
        up_len, down_len = len(up), len(down)
        genes_out[0] = self.up_token_id
        genes_out[1:1 + up_len] = up
        genes_out[1 + up_len] = self.down_token_id
        genes_out[2 + up_len:2 + up_len + down_len] = down

        return genes_out
