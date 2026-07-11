#!/usr/bin/env python3
"""
Attention-derived gene-importance baseline for CASCADE-Explainer validation
(Methods / Supplementary Note: "attention-derived baseline"). Extracts self-attention
weights from the pre-trained CASCADE encoder, aggregated across layers, heads, and
query positions to give one importance score per gene token per cell, then ranks
genes per cell type (and globally) as a simpler/faster alternative to the
perturbation-based gene explainer.

Usage:
    python -m analysis.alzheimers.attention_baseline \
        --output_path ./attention_outputs/attention_baseline_SEATTLE_cell_type.pkl \
        --dataset_name SEATTLE --task cell_type \
        --transformer_checkpoint /path/to/checkpoint.pt
"""
import argparse
import gc
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cascade.data.sampler import condition_samples
from cascade.data.utils import load_and_merge_pickles
from cascade.explainer.checkpoint_utils import load_ddp_checkpoint
from cascade.explainer.config import CASCADE_CKPT_ROOT, CASCADE_DATA_ROOT
from cascade.model.cascade_model import SeqDataset, TransformerGenerator, prepare_dataloader, set_seed

D_MODEL, NHEAD, NLAYERS, DIM_EMBEDDING, DROPOUT = 384, 6, 12, 384, 0.1
PAD_TOKEN = "<pad>"
NCLASS = 2
CELL_EMB_STYLE = 'avg-pool'
DA = True
SPECIAL_TOKEN_NAMES = ['<pad>', '<cls>', '<up>', '<down>', '<mask>', '<unk>']

# Gene-importance attention baseline is only run for the real paper datasets
# (AUTISM, SEATTLE, M2); organoid/CHOOSE configs are dropped as out of scope.
DATASET_CONFIGS = {
    'AUTISM': {
        'merged_contexts': 'disease_cell_type_tissue',
        'batch': 2048,
        'context_specific_projections': True,
        'collator_path': CASCADE_DATA_ROOT / 'AUTISM/collator_disease_cell_type_tissue_kempner_metadata_final_corrected.pkl',
    },
    'SEATTLE': {
        'merged_contexts': 'disease_cell_type_tissue',
        'batch': 256,
        'context_specific_projections': True,
        'collator_path': CASCADE_DATA_ROOT / 'SEATTLE/collator_humans/collator_disease_cell_type_tissue_kempner_metadata_final_corrected_100000.pkl',
    },
    'M2': {
        'merged_contexts': 'cell_type',
        'batch': 32,
        'context_specific_projections': True,
        'collator_path': CASCADE_DATA_ROOT / 'M2/collator_humans',
    },
}


class AttentionTransformerEncoder(nn.Module):
    """Wraps a TransformerEncoder to accumulate mean self-attention importance
    per token position on-the-fly, instead of materializing every layer's
    full attention matrix."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.num_layers = len(encoder.layers)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        batch_size, seq_len, _ = src.shape
        accumulated_importance = torch.zeros(batch_size, seq_len, device=src.device)

        for layer in self.encoder.layers:
            residual = output
            attn_output, attn_weights = layer.self_attn(
                output, output, output,
                attn_mask=mask, key_padding_mask=src_key_padding_mask,
                need_weights=True, average_attn_weights=True,
            )
            accumulated_importance += attn_weights.sum(dim=1)  # (batch, seq)
            del attn_weights
            output = residual + layer.dropout1(attn_output)
            output = layer.norm1(output)
            residual = output
            output = layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
            output = residual + layer.dropout2(output)
            output = layer.norm2(output)

        return output, accumulated_importance / self.num_layers


def aggregate_gene_importance_per_cell_type(importance_scores, input_ids, cell_types, token_dictionary):
    """Group per-position importance scores by gene token and by cell type."""
    id_to_token = {v: k for k, v in token_dictionary.items()}
    input_ids_np, importance_np = input_ids.cpu().numpy(), importance_scores.cpu().numpy()
    if isinstance(cell_types, torch.Tensor):
        cell_types = cell_types.cpu().numpy().tolist()
    elif isinstance(cell_types, np.ndarray):
        cell_types = cell_types.tolist()

    per_cell_type_scores = {}
    for batch_idx in range(input_ids_np.shape[0]):
        cell_type = cell_types[batch_idx]
        if isinstance(cell_type, (list, np.ndarray)):
            cell_type = cell_type[0] if len(cell_type) > 0 else 'unknown'
        if isinstance(cell_type, bytes):
            cell_type = cell_type.decode('utf-8')
        cell_type = str(cell_type)
        per_cell_type_scores.setdefault(cell_type, {})

        for pos_idx in range(input_ids_np.shape[1]):
            token_id = int(input_ids_np[batch_idx, pos_idx])
            token_name = id_to_token.get(token_id)
            if token_name is not None and token_name.startswith('ENSG'):
                per_cell_type_scores[cell_type].setdefault(token_id, []).append(float(importance_np[batch_idx, pos_idx]))

    per_cell_type_importance = {}
    for cell_type, gene_scores in per_cell_type_scores.items():
        per_cell_type_importance[cell_type] = {
            gene_id: {
                'mean': np.mean(scores), 'std': np.std(scores) if len(scores) > 1 else 0.0,
                'max': np.max(scores), 'min': np.min(scores), 'count': len(scores),
                'gene_name': id_to_token.get(gene_id, f'unknown_{gene_id}'),
            }
            for gene_id, scores in gene_scores.items()
        }
    return per_cell_type_importance


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--transformer_checkpoint', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='SEATTLE', choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--task', type=str, default='cell_type', help='Grouping task for per-class rankings (cell_type, disease, tissue)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: dataset-specific)')
    parser.add_argument('--max_batches', type=int, default=None, help='Maximum number of batches to process (for testing)')
    parser.add_argument('--healthy_only', action='store_true', help='Filter to healthy donors only')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name
    dataset_config = DATASET_CONFIGS[dataset_name]
    batch_size = args.batch_size or dataset_config['batch']
    context_specific_projections = dataset_config['context_specific_projections']
    merged_contexts = dataset_config['merged_contexts']
    task_key = args.task

    print(f"\n{'=' * 80}\nATTENTION-DERIVED GENE IMPORTANCE BASELINE (PER CELL TYPE)\n{'=' * 80}")
    print(f"Dataset: {dataset_name}  Task: {task_key}  Batch size: {batch_size}")

    token_dict_path = CASCADE_DATA_ROOT / dataset_name / f"tokenizer_dictionary_{dataset_name}.pkl"
    with open(token_dict_path, "rb") as f:
        token_dictionary = pickle.load(f)
    ntoken = len(token_dictionary)
    print(f"Token dictionary size: {ntoken}")

    collator_path = Path(dataset_config['collator_path'])
    collator_files = sorted(collator_path.glob('collator_*.pkl')) if collator_path.is_dir() else [collator_path]
    print(f"Will process {len(collator_files)} collator file(s)")

    set_seed(20)
    model = TransformerGenerator(
        d_model=D_MODEL, nhead=NHEAD, ntoken=ntoken, dim_embedding=DIM_EMBEDDING, nlayers=NLAYERS,
        vocab=token_dictionary, dropout=DROPOUT, pad_token=PAD_TOKEN, nclass=NCLASS,
        cell_emb_style=CELL_EMB_STYLE, extract_embeddings=True,
        context_specific_projections=context_specific_projections, DA=DA,
    ).to(device)
    _, model = load_ddp_checkpoint(args.transformer_checkpoint, model, device=device)
    model.eval()

    attention_encoder = AttentionTransformerEncoder(model.transformer_encoder)

    special_tokens = {token_dictionary[name] for name in SPECIAL_TOKEN_NAMES if name in token_dictionary}
    per_cell_type_gene_scores, global_gene_scores = {}, {}
    total_cells, total_batches = 0, 0

    for file_idx, collator_file in enumerate(collator_files):
        print(f"\nProcessing file {file_idx + 1}/{len(collator_files)}: {collator_file.name}")
        try:
            output_data_dict = load_and_merge_pickles(collator_file.parent, path_to_collator=collator_file,
                                                        merged_context=merged_contexts)
        except Exception as e:
            print(f"  Error loading {collator_file.name}: {e}")
            continue

        if args.healthy_only and 'disease' in output_data_dict and isinstance(output_data_dict['disease'], torch.Tensor):
            healthy_mask = output_data_dict['disease'] == 1
            n_total = len(output_data_dict['disease'])
            for key in list(output_data_dict.keys()):
                value = output_data_dict[key]
                if isinstance(value, torch.Tensor) and len(value) == n_total:
                    output_data_dict[key] = value[healthy_mask]
                elif isinstance(value, list) and len(value) == n_total:
                    output_data_dict[key] = [v for v, keep in zip(value, healthy_mask) if keep]

        dataset = SeqDataset(output_data_dict)
        dataloader = prepare_dataloader(
            data_pt=dataset, batch_size=batch_size, class_label=None,
            condition_fn=condition_samples, shuffle=False, num_workers=0, sampler=False,
        )
        max_batches = args.max_batches or len(dataloader)
        print(f"  Cells: {len(dataset)}, Batches: {len(dataloader)}")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, total=min(max_batches, len(dataloader)),
                                                          desc=f"  File {file_idx + 1}")):
                if batch_idx >= max_batches:
                    break
                try:
                    input_ids = batch_data["input_ids"]
                    if input_ids.dim() == 3:
                        input_ids = input_ids.squeeze(0)
                    input_ids = input_ids.to(device)
                    batch_cell_types = batch_data.get(task_key)
                    if batch_cell_types is None:
                        batch_cell_types = ['unknown'] * input_ids.shape[0]
                    elif isinstance(batch_cell_types, torch.Tensor):
                        batch_cell_types = batch_cell_types.cpu().numpy()

                    pad_token_id = token_dictionary[PAD_TOKEN]
                    src_key_padding_mask = input_ids.eq(pad_token_id)
                    src = model.encoder(input_ids.long())
                    src = model.pos_encoder(src)
                    src = model.bn(src.permute(0, 2, 1)).permute(0, 2, 1)
                    _, importance_scores = attention_encoder(src, src_key_padding_mask=src_key_padding_mask)

                    if special_tokens:
                        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                        for special_id in special_tokens:
                            special_mask |= (input_ids == special_id)
                        importance_scores[special_mask] = 0.0

                    batch_per_cell_type = aggregate_gene_importance_per_cell_type(
                        importance_scores, input_ids, batch_cell_types, token_dictionary)
                    for cell_type, gene_stats in batch_per_cell_type.items():
                        per_cell_type_gene_scores.setdefault(cell_type, {})
                        for gene_id, stats in gene_stats.items():
                            values = [stats['mean']] * stats['count']
                            per_cell_type_gene_scores[cell_type].setdefault(
                                gene_id, {'scores': [], 'gene_name': stats['gene_name']})['scores'].extend(values)
                            global_gene_scores.setdefault(
                                gene_id, {'scores': [], 'gene_name': stats['gene_name']})['scores'].extend(values)

                    total_cells += input_ids.shape[0]
                    total_batches += 1
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"\n  Error processing batch {batch_idx}: {e}")
                    continue

        del output_data_dict, dataset, dataloader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"  Completed. Total cells so far: {total_cells:,}")

    print(f"\nTotal processed: {total_cells:,} cells across {total_batches:,} batches")

    def rank_genes(gene_data):
        ranking = [{'gene_id': gid, 'gene_name': d['gene_name'], 'mean_importance': np.mean(d['scores']),
                    'std_importance': np.std(d['scores']) if len(d['scores']) > 1 else 0.0,
                    'max_importance': np.max(d['scores']), 'min_importance': np.min(d['scores']),
                    'count': len(d['scores'])} for gid, d in gene_data.items()]
        ranking.sort(key=lambda x: x['mean_importance'], reverse=True)
        return ranking

    per_cell_type_rankings = {ct: rank_genes(d) for ct, d in per_cell_type_gene_scores.items()}
    global_ranking = rank_genes(global_gene_scores)

    print(f"\nFound {len(per_cell_type_rankings)} unique {task_key} classes")
    print(f"Top 10 genes globally (across all {task_key}s):")
    for i, gene in enumerate(global_ranking[:10]):
        print(f"  {i + 1}. {gene['gene_name']}: mean={gene['mean_importance']:.6f} count={gene['count']}")

    output_data = {
        'per_cell_type_rankings': per_cell_type_rankings,
        'global_ranking': global_ranking,
        'metadata': {
            'dataset_name': dataset_name, 'task': task_key, 'aggregation_method': 'mean',
            'num_cell_types': len(per_cell_type_rankings), 'num_genes_global': len(global_ranking),
            'num_batches_processed': total_batches, 'num_cells_processed': total_cells,
            'num_files_processed': len(collator_files), 'transformer_checkpoint': str(args.transformer_checkpoint),
            'healthy_only': args.healthy_only, 'batch_size': batch_size,
            'cell_type_counts': {ct: sum(g['count'] for g in ranking) for ct, ranking in per_cell_type_rankings.items()},
        },
        'token_dictionary': token_dictionary,
    }

    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    csv_dir = output_path.parent / f"{output_path.stem}_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for cell_type, ranking in per_cell_type_rankings.items():
        safe_ct = str(cell_type).replace('/', '_').replace(' ', '_').replace(':', '_')[:50]
        with open(csv_dir / f"ranking_{safe_ct}.csv", 'w') as f:
            f.write("rank,gene_id,gene_name,mean_importance,std_importance,max_importance,count\n")
            for i, gene in enumerate(ranking):
                f.write(f"{i + 1},{gene['gene_id']},{gene['gene_name']},{gene['mean_importance']:.8f},"
                        f"{gene['std_importance']:.8f},{gene['max_importance']:.8f},{gene['count']}\n")

    with open(output_path.with_suffix('.global.csv'), 'w') as f:
        f.write("rank,gene_id,gene_name,mean_importance,std_importance,max_importance,count\n")
        for i, gene in enumerate(global_ranking):
            f.write(f"{i + 1},{gene['gene_id']},{gene['gene_name']},{gene['mean_importance']:.8f},"
                    f"{gene['std_importance']:.8f},{gene['max_importance']:.8f},{gene['count']}\n")

    print(f"\nDone. Cell types: {len(per_cell_type_rankings)}, genes ranked: {len(global_ranking)}")
    print(f"Results saved to: {output_path}\nPer-cell-type CSVs in: {csv_dir}/")


if __name__ == "__main__":
    main()
