"""
Perturbation-based gene explainer (Methods 9.9). Quantifies gene importance by
reordering a gene's position within its up-/down-regulated block (moving it to
the low-rank end of that block) and measuring the resulting change in the
downstream model prediction, by default via KL divergence.
"""
import logging

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class Explainer(torch.nn.Module):
    def __init__(
        self,
        model,
        clf_model,
        token_dictionary,
        debug: bool = False,
        CCE=True,
        temperature=0.1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        CLASS=False,
        nclass=2,
        PAD_TOKEN="<pad>",
        cell_level=True,
        perturbed_prediction_metric="kl_div",
        donor_level_tasks=None,
    ):
        super().__init__()
        self.model = model
        self.clf_model = clf_model
        self.token_dictionary = token_dictionary
        self.CCE = CCE
        self.temperature = temperature
        self.device = device
        self.CLASS = CLASS
        self.nclass = nclass
        self.PAD_TOKEN = token_dictionary[PAD_TOKEN]
        self.cell_level = cell_level
        self.perturbed_prediction_metric = perturbed_prediction_metric
        self.debug = debug
        self.donor_level_tasks = donor_level_tasks or []

        self.logger = logging.getLogger("explainer.Explainer")
        if self.debug:
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.addHandler(logging.NullHandler())

        self.down_token = token_dictionary["<down>"]
        self.up_token = token_dictionary["<up>"]
        self.max_gene_number = max(token_dictionary.values())
        self.whole_gene_indices = [v for k, v in token_dictionary.items() if k.startswith("ENSG")]

        self.model = self.model.to(self.device)
        self.clf_model = self.clf_model.to(self.device)

    def _ensure_batch_first(self, batch_data):
        """Ensure input_ids (and padding mask) have layout (batch, seq_len). Model expects batch first.
        Fixes (1, batch, seq_len), (seq_len, batch), or 1D (seq_len,) from dataloader/collation."""
        if not isinstance(batch_data, dict) or "input_ids" not in batch_data:
            return batch_data
        inp = batch_data["input_ids"]
        if not isinstance(inp, torch.Tensor):
            return batch_data
        if inp.dim() == 1:
            batch_data = dict(batch_data)
            batch_data["input_ids"] = inp.unsqueeze(0)
            return batch_data
        if inp.dim() == 3:
            # (1, batch, seq_len) -> squeeze to (batch, seq_len); model expects (batch, seq_len)
            if inp.size(0) == 1:
                batch_data = dict(batch_data)
                batch_data["input_ids"] = inp.squeeze(0)
                for key in ("attention_mask", "src_key_padding_mask"):
                    if key in batch_data and isinstance(batch_data[key], torch.Tensor) and batch_data[key].dim() == 3 and batch_data[key].size(0) == 1:
                        batch_data[key] = batch_data[key].squeeze(0)
                return batch_data
        if inp.dim() == 2 and inp.size(0) > inp.size(1):
            # (seq_len, batch) -> transpose to (batch, seq_len)
            batch_data = dict(batch_data)
            batch_data["input_ids"] = inp.transpose(0, 1)
            for key in ("attention_mask", "src_key_padding_mask"):
                if key in batch_data and isinstance(batch_data[key], torch.Tensor) and batch_data[key].dim() == 2:
                    batch_data[key] = batch_data[key].transpose(0, 1)
            return batch_data
        return batch_data

    def _perturb_gene_by_index(self, x, gene_position: int, up_regulated=False):
        """Move the token at `gene_position` to the low-rank end of its up-/down-regulated block."""
        if up_regulated:
            for b in range(x.shape[0]):
                pos_down = torch.where(x[b] == self.down_token)[0][0].item()
                after_gene = x[b, (gene_position + 1):(pos_down)]
                x[b] = torch.cat([x[b, :gene_position], after_gene, x[b, gene_position].unsqueeze(0), x[b, pos_down:]], dim=0)
        else:
            for b in range(x.shape[0]):
                pad_matches = torch.where(x[b] == self.PAD_TOKEN)[0]
                pos_down = torch.where(x[b] == self.down_token)[0][0].item()
                if pad_matches.numel() > 0:
                    assert pad_matches[0] > torch.where(x[b] == self.down_token)[0]
                    end_down = pad_matches[0].item()
                    after_gene = x[b, (gene_position + 1):(end_down)]
                    x[b] = torch.cat([x[b, :gene_position], after_gene, x[b, gene_position].unsqueeze(0), x[b, (end_down + 1):]], dim=0)
                else:
                    after_gene = x[b, (gene_position + 1):]
                    x[b] = torch.cat([x[b, :gene_position], after_gene, x[b, gene_position].unsqueeze(0)], dim=0)
        return x

    def _perturb_gene(self, x, gene_idx: int, perturbation_idx: int = None):
        """Move the token for `gene_idx` (if present in each sequence) to the low-rank end of its block."""
        gene_presence = torch.zeros(x.shape[0], dtype=torch.uint8)

        if gene_idx is None:
            self.logger.debug("_perturb_gene called with gene_idx=None")
            return x, gene_presence

        try:
            for b in range(x.shape[0]):
                matches = torch.where(x[b] == gene_idx)[0]
                if matches.numel() > 0:
                    gene_presence[b] = 1
                    gene_position = matches[0].item()

                    down_matches = torch.where(x[b] == self.down_token)[0]
                    up_matches = torch.where(x[b] == self.up_token)[0]
                    if down_matches.numel() == 0 or up_matches.numel() == 0:
                        if self.debug:
                            self.logger.debug(f"Missing up/down tokens in sequence b={b}; down_matches={down_matches}, up_matches={up_matches}")
                        continue

                    pos_down = down_matches[0].item()

                    if gene_position < pos_down:
                        after_gene = x[b, (gene_position + 1):(pos_down)]
                        x[b] = torch.cat([x[b, :gene_position], after_gene, x[b, gene_position].unsqueeze(0), x[b, pos_down:]], dim=0)
                    else:
                        pad_matches = torch.where(x[b] == self.PAD_TOKEN)[0]
                        if pad_matches.numel() > 0:
                            end_down = pad_matches[0].item()
                            after_gene = x[b, (gene_position + 1):(end_down)]
                            x[b] = torch.cat([x[b, :gene_position], after_gene, x[b, gene_position].unsqueeze(0), x[b, (end_down + 1):]], dim=0)
                        else:
                            after_gene = x[b, (gene_position + 1):]
                            x[b] = torch.cat([x[b, :gene_position], after_gene, x[b, gene_position].unsqueeze(0)], dim=0)

        except Exception as exc:
            self.logger.exception(f"Exception in _perturb_gene for gene_idx={gene_idx}: {exc}")

        return x, gene_presence

    def _group_and_pad(self, x, donor_ids):
        """Group cells by donor and pad to the same length within each donor group."""
        if not isinstance(donor_ids, torch.Tensor):
            donor_ids = torch.tensor(donor_ids)

        donor_sorted, sort_idx = torch.sort(donor_ids)
        x_sorted = x[sort_idx]

        unique_ids, counts = torch.unique_consecutive(donor_sorted, return_counts=True)
        split_sections = torch.split(x_sorted, counts.tolist())
        padded_x = pad_sequence(split_sections, batch_first=True)

        attn_mask = torch.zeros(padded_x.shape[0], padded_x.shape[1], dtype=torch.bool)
        for i, count in enumerate(counts):
            attn_mask[i, :count] = True

        return padded_x, unique_ids, attn_mask

    @torch.no_grad()
    def predict(self, z, attn_mask=None, donor_ids=None):
        """
        Make predictions with the classifier model.

        Args:
            z: embeddings tensor (batch_size, embed_dim) or (batch_size, seq_len, embed_dim)
            attn_mask: attention mask for donor-level models
            donor_ids: donor IDs for grouping cells (needed for donor-level models)
        """
        if self.cell_level:
            return self.clf_model(z)

        if donor_ids is not None and z.dim() == 2:
            z_grouped, unique_donors, attn_mask = self._group_and_pad(z, donor_ids)
            return self.clf_model(z_grouped, attn_mask)
        return self.clf_model(z, attn_mask)

    @torch.no_grad()
    def _perturbed_prediction(
        self,
        batch_data,
        x,
        gene_idx: int,
        perturbation_idx: int = None,
        prediction=None,
        by_global_position=False,
        global_perturb_idx=None,
        global_perturb_up_regulated=False,
    ):
        batch_data = self._ensure_batch_first(batch_data)
        if batch_data["input_ids"].dim() == 3:
            batch_data['input_ids'] = batch_data['input_ids'].squeeze(0)

        if by_global_position:
            xp = self._perturb_gene_by_index(x, gene_position=global_perturb_idx, up_regulated=global_perturb_up_regulated)
        else:
            x, gene_presence = self._perturb_gene(x, gene_idx=gene_idx, perturbation_idx=perturbation_idx)
            xp = x[gene_presence == 1, ...]

            if xp.numel() == 0:
                if prediction is None:
                    self.logger.debug("No gene occurrences found and no cached prediction provided; returning empty tensors")
                    return torch.empty((0, 0), device=self.device), gene_presence
                predictions = torch.full_like(prediction, fill_value=-1)
                return predictions, gene_presence

            if prediction is None:
                self.logger.debug("No cached prediction provided; creating placeholder tensor")
                predictions = torch.full((x.shape[0], self.nclass), fill_value=-1, device=self.device)
            else:
                predictions = torch.full_like(prediction, fill_value=-1)

        src_key_padding_mask_eval = self._src_key_padding_mask(xp)

        _, emb = self.model(
            data=batch_data,
            src=xp,
            src_key_padding_mask=src_key_padding_mask_eval,
            CCE=self.CCE,
            temperature=self.temperature,
            device=self.device,
            CLASS=self.CLASS,
            nclass=self.nclass,
            return_overall_embeddings=True,
        )

        # Get donor_ids if available for donor-level models
        donor_ids = None
        if not self.cell_level and 'donor_id' in batch_data:
            donor_ids_full = batch_data['donor_id']
            if isinstance(donor_ids_full, torch.Tensor):
                donor_ids_full = donor_ids_full.cpu()

            if not by_global_position:
                if gene_presence is not None and gene_presence.sum() < len(donor_ids_full):
                    if isinstance(donor_ids_full, list):
                        donor_ids = [donor_ids_full[i] for i in range(len(donor_ids_full)) if gene_presence[i] == 1]
                    else:
                        donor_ids = donor_ids_full[gene_presence == 1]
                else:
                    donor_ids = donor_ids_full
            else:
                donor_ids = donor_ids_full

        perturbed_prediction = self.predict(emb, donor_ids=donor_ids)

        if by_global_position:
            return perturbed_prediction

        try:
            predictions[gene_presence == 1, :] = perturbed_prediction
        except Exception as exc:
            self.logger.exception(f"Failed to assign perturbed predictions back into full tensor: {exc}")
            raise

        return predictions, gene_presence

    def _src_key_padding_mask(self, input_ids):
        return input_ids.eq(self.PAD_TOKEN)

    def _perturbed_prediction_metric(self, prediction, perturbed_prediction):
        if self.perturbed_prediction_metric == "kl_div":
            return F.kl_div(
                F.log_softmax(prediction, dim=1), F.log_softmax(perturbed_prediction, dim=1),
                reduction="none", log_target=True,
            ).sum(dim=-1)
        if self.perturbed_prediction_metric == "mse":
            return F.mse_loss(prediction, perturbed_prediction, dim=-1)
        raise ValueError(f"Invalid perturbed prediction metric: {self.perturbed_prediction_metric}")

    @torch.no_grad()
    def forward(
        self,
        batch_data,
        cached_prediction=None,
        perturbation_idx: int = -1,
        robust=True,
        global_number_to_perturb=None,
    ):
        """
        Perturbation-based gene explainer (Methods 9.9). Perturbs the top N/2 positions
        of the up-regulated block and the top N/2 positions of the down-regulated block
        separately, and scores each by the divergence it induces in the model's prediction.

        Args:
            batch_data: dict with at least "input_ids" (token sequences).
            cached_prediction: precomputed baseline prediction; recomputed if None.
            global_number_to_perturb: total number of positions N to perturb (split
                evenly between the up- and down-regulated blocks).

        Returns:
            (perturbed_predictions, importance_scores) stacked as
            [up-regulated results; down-regulated results].
        """
        batch_data = self._ensure_batch_first(batch_data)
        if batch_data["input_ids"].dim() == 3:
            batch_data["input_ids"] = batch_data["input_ids"].squeeze(0)

        if cached_prediction is None:
            src_key_padding_mask_eval = self._src_key_padding_mask(batch_data["input_ids"].to(self.device))
            _, embedding = self.model(
                data=batch_data,
                src=batch_data["input_ids"].clone().to(self.device),
                src_key_padding_mask=src_key_padding_mask_eval,
                CCE=self.CCE,
                temperature=self.temperature,
                device=self.device,
                CLASS=self.CLASS,
                nclass=self.nclass,
                return_overall_embeddings=True,
            )

            donor_ids = None
            if not self.cell_level and 'donor_id' in batch_data:
                donor_ids = batch_data['donor_id']
                if isinstance(donor_ids, torch.Tensor):
                    donor_ids = donor_ids.cpu()

            prediction = self.predict(embedding, donor_ids=donor_ids)
        else:
            prediction = cached_prediction

        wrapper = tqdm if robust else lambda x: x

        x = batch_data.get("input_ids")
        x = x.clone().detach().to(self.device)

        num_genes_to_perturb = global_number_to_perturb // 2

        up_regulated_predictions = []
        up_regulated_importance = []
        down_regulated_predictions = []
        down_regulated_importance = []

        print(f"Running upregulated tokens: {num_genes_to_perturb}")
        for i in wrapper(range(num_genes_to_perturb)):
            pert_pred = self._perturbed_prediction(
                batch_data, x.clone(), gene_idx=i, perturbation_idx=perturbation_idx, prediction=prediction,
                by_global_position=True, global_perturb_idx=i, global_perturb_up_regulated=True,
            )
            up_regulated_predictions.append(pert_pred)
            up_regulated_importance.append(self._perturbed_prediction_metric(prediction, pert_pred).clone().detach().cpu())

        print(f"Running downregulated tokens: {num_genes_to_perturb}")
        for i in wrapper(range(num_genes_to_perturb)):
            pert_pred = self._perturbed_prediction(
                batch_data, x.clone(), gene_idx=i, perturbation_idx=perturbation_idx, prediction=prediction,
                by_global_position=True, global_perturb_idx=i, global_perturb_up_regulated=False,
            )
            down_regulated_predictions.append(pert_pred)
            down_regulated_importance.append(self._perturbed_prediction_metric(prediction, pert_pred).clone().detach().cpu())

        up_regulated_tensor = torch.stack(up_regulated_predictions) if up_regulated_predictions else torch.empty(0)
        down_regulated_tensor = torch.stack(down_regulated_predictions) if down_regulated_predictions else torch.empty(0)
        up_regulated_importance_tensor = torch.stack(up_regulated_importance) if up_regulated_importance else torch.empty(0)
        down_regulated_importance_tensor = torch.stack(down_regulated_importance) if down_regulated_importance else torch.empty(0)

        return (
            torch.cat([up_regulated_tensor, down_regulated_tensor], dim=0),
            torch.cat([up_regulated_importance_tensor, down_regulated_importance_tensor], dim=0),
        )
