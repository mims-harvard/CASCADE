"""
Multi-GPU/multi-node CASCADE embedding extraction, sharding batches across ranks
via DistributedSampler and saving embeddings + metadata incrementally in chunks
to bound memory usage on large datasets (e.g. HLCA, SEATTLE).

Extracts against checkpoints trained with context_specific_projections=True and
always keeps the shared-encoder cell_emb, never the projected context embeddings
(Methods 9.7): the model returns (xcs, cell_emb) when return_overall_embeddings=True,
and only cell_emb is saved.

Usage (see scripts/get_emb_parallel.sh for the SLURM multi-node launcher):
    torchrun --nproc_per_node=4 -m cascade.explainer.get_embeddings_parallel \\
        --dataset HLCA --checkpoint /path/to/ckpt.pt --output-path /path/to/embeddings.pkl
"""
import argparse
import gc
import os
import pickle
from collections import defaultdict
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from cascade.data import sampler as sampler_contrastive
from cascade.data.utils import load_and_merge_pickles
from cascade.explainer import config as cfg
from cascade.explainer.checkpoint_utils import load_ddp_checkpoint
from cascade.model import cascade_model


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_memory_usage():
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated, {torch.cuda.memory_reserved() / 1024**2:.1f}MB cached"
    process = psutil.Process(os.getpid())
    return f"CPU: {process.memory_info().rss / 1024**2:.1f}MB"


@torch.no_grad()
def process_chunk_and_save(loader, model, device, token_dictionary, output_path, chunk_size, rank):
    """Extract embeddings in chunks of `chunk_size` batches, saving each chunk to
    its own pickle to bound peak memory. Each rank processes a disjoint shard of
    batches via DistributedSampler and writes to its own output path."""
    model.eval()
    chunk_data = defaultdict(list)
    chunk_count = 0
    total_batches = len(loader)

    def save_chunk(count):
        chunk_output_path = output_path.parent / f"{output_path.stem}_chunk_{count:03d}.pkl"
        chunk_output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(chunk_data["embedding"]) == 0:
            print(f"⚠️ [Rank {rank}] Tried to save empty chunk {count}; skipping.")
            return
        chunk_data["embedding"] = torch.stack(chunk_data["embedding"])
        with open(chunk_output_path, "wb") as f:
            pickle.dump(dict(chunk_data), f)
        print(f"✅ [Rank {rank}] Saved chunk {count} ({len(chunk_data['embedding'])} samples) -> {chunk_output_path.name}")

    if rank == 0:
        print(f"Processing {total_batches} batches per GPU in chunks of {chunk_size}")
        print(f"Initial memory usage: {get_memory_usage()}")

    iterator = tqdm(loader, desc=f"GPU {rank} processing batches") if rank == 0 else loader
    progress_interval = max(10, total_batches // 20)

    for batch_idx, batch_data_eval in enumerate(iterator):
        if batch_idx > 0 and batch_idx % progress_interval == 0:
            print(f"[Rank {rank}] Progress: {batch_idx}/{total_batches} batches ({100*batch_idx/total_batches:.1f}%)")

        input_gene_ids_eval = batch_data_eval["input_ids"].to(device, non_blocking=True)
        src_key_padding_mask_eval = input_gene_ids_eval.eq(token_dictionary[cfg.PAD_TOKEN])

        # With context_specific_projections=True, forward() returns (xcs, cell_emb) when
        # return_overall_embeddings=True. We only keep cell_emb (Methods 9.7).
        _, emb = model(
            data=batch_data_eval, src=input_gene_ids_eval, src_key_padding_mask=src_key_padding_mask_eval,
            CCE=True, temperature=cfg.TEMPERATURE, device=device, CLASS=False, nclass=cfg.NCLASS,
            return_overall_embeddings=True,
        )

        chunk_data["embedding"].extend(emb.cpu())
        for key, value in batch_data_eval.items():
            try:
                if isinstance(value, torch.Tensor):
                    chunk_data[key].extend(value.cpu().tolist())
                else:
                    chunk_data[key].extend(value)
            except Exception as e:
                if rank == 0 and batch_idx == 0:
                    print(f"⚠️ Skipped key {key} due to: {e}")

        if (batch_idx + 1) % chunk_size == 0:
            chunk_count += 1
            save_chunk(chunk_count)
            del chunk_data
            gc.collect()
            chunk_data = defaultdict(list)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(chunk_data["embedding"]) > 0:
        chunk_count += 1
        print(f"[Rank {rank}] Final flush: saving leftover {len(chunk_data['embedding'])} samples")
        save_chunk(chunk_count)
    else:
        print(f"[Rank {rank}] Final flush: nothing to save")

    print(f"✅ [Rank {rank}] Finished processing all {total_batches} batches, generated {chunk_count} chunks")
    return chunk_count


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU CASCADE embedding extraction")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (must be a key in cascade.data.splits.SPLITS_BY_DATASET)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the transformer checkpoint (.pt)")
    parser.add_argument("--output-path", type=str, required=True, help="Base output path; rank/chunk suffixes are appended")
    parser.add_argument("--token-dictionary", type=str, default=None, help="Override the default tokenizer_dictionary_{dataset}.pkl path")
    parser.add_argument("--collator-path", type=str, default=None, help="Override the default collator path")
    parser.add_argument("--collator-root", type=str, default="", help="Optional subdirectory under the dataset directory where collator chunks live")
    parser.add_argument("--merged-contexts", type=str, default=None, help="Override the default merged_contexts (e.g. 'disease_cell_type_tissue')")
    parser.add_argument("--batch", type=int, default=2048, help="Batch size per GPU")
    parser.add_argument("--chunk-size", type=int, default=10, help="Number of batches per saved chunk")
    return parser.parse_args()


@record
def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    args = parse_args()

    if rank == 0:
        print("=" * 80)
        print("PARALLEL CHECKPOINT LOADING AND EMBEDDING EXTRACTION")
        print(f"Running on {world_size} GPUs, dataset={args.dataset}")
        print("=" * 80)

    dataset_name = args.dataset
    output_dir = cfg.CASCADE_DATA_ROOT / dataset_name
    merged_contexts = args.merged_contexts or ('disease_cell_type_tissue' if dataset_name.upper() == 'AUTISM' else cfg.MERGED_CONTEXTS)
    token_dictionary_file = Path(args.token_dictionary) if args.token_dictionary else output_dir / f"tokenizer_dictionary_{dataset_name}.pkl"
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output_path)

    if rank == 0:
        print(f"Checkpoint: {checkpoint_path.name}")
        print(f"Token dictionary: {token_dictionary_file}")

    with open(token_dictionary_file, "rb") as input_file:
        token_dictionary = pickle.load(input_file)
    ntoken = len(token_dictionary)
    if rank == 0:
        print(f"✓ Token dictionary loaded: {ntoken} tokens")

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
        only_contrastive=True,
        context_specific_projections=True,
        DA=cfg.DA,
        merged_contexts=merged_contexts,
    ).to(device)

    if rank == 0:
        print(f"✓ Model initialized ({sum(p.numel() for p in mod.parameters()):,} parameters)")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Verify the checkpoint actually changes weights from random init, to catch
    # silently-failed / no-op loads early.
    reference_param_before = None
    if hasattr(mod, 'transformer_encoder'):
        reference_param_before = mod.transformer_encoder.layers[0].self_attn.in_proj_weight.clone()

    try:
        model_dict = torch.load(checkpoint_path, map_location=device)
        mod.load_state_dict(model_dict["model_state_dict"])
        if rank == 0:
            print("✅ Successfully loaded checkpoint (direct load)")
    except Exception as e:
        if rank == 0:
            print(f"⚠️ Direct load failed: {e}; trying DDP-aware loader")
        _, mod = load_ddp_checkpoint(checkpoint_path=checkpoint_path, model=mod, device=device, verbose=(rank == 0))

    if rank == 0 and reference_param_before is not None:
        weights_changed = not torch.equal(reference_param_before, mod.transformer_encoder.layers[0].self_attn.in_proj_weight)
        if not weights_changed:
            raise RuntimeError("Checkpoint verification failed: transformer weights unchanged from initialization")
        print("✅ Checkpoint verification passed - weights were updated")

    mod = DDP(mod, device_ids=[local_rank])

    if rank == 0:
        print("\n📦 Loading data and collator...")

    collator_root_dir = output_dir / args.collator_root if args.collator_root else output_dir
    path_to_collator = Path(args.collator_path) if args.collator_path else (
        output_dir / f"collator_{merged_contexts}_kempner_metadata_final_corrected.pkl"
    )

    output_data_dict = load_and_merge_pickles(collator_root_dir, merged_context=merged_contexts, path_to_collator=path_to_collator)
    output_data_dict = cascade_model.SeqDataset(output_data_dict)

    if rank == 0:
        print("✓ Data loaded successfully")

    loader_all = cascade_model.prepare_dataloader(
        data_pt=output_data_dict,
        batch_size=args.batch,
        shuffle=False,  # DistributedSampler handles shuffling
        num_workers=0,
        condition_fn=sampler_contrastive.condition_samples,
        sampler=True,  # distributed sampler
        class_label=None,
    )

    if rank == 0:
        print(f"✓ Distributed dataloader created: {len(loader_all)} batches per GPU, {len(loader_all) * world_size} total")

    if dist.is_initialized():
        dist.barrier()

    rank_output_path = output_path.parent / f"{output_path.stem}_rank_{rank:02d}{output_path.suffix}"
    try:
        chunk_count = process_chunk_and_save(loader_all, mod, device, token_dictionary, rank_output_path, args.chunk_size, rank)
    except Exception:
        cleanup_distributed()
        raise

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print("\n" + "=" * 80)
        print("EMBEDDING EXTRACTION COMPLETE")
        print(f"Chunk files saved in: {rank_output_path.parent}, pattern: {output_path.stem}_rank_*_chunk_*.pkl")
        print(f"Final memory usage: {get_memory_usage()}")
        print("=" * 80)

    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        cleanup_distributed()
        raise SystemExit(1)
