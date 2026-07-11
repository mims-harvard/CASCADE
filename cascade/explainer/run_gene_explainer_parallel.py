"""
Perturbation-based gene explainer - parallel multi-GPU version (Methods 9.9,
second half). Runs inference over cell-level MLP models trained by
main_cell_level.py to extract gene-level importance scores, sharded across
ranks via the distributed contrastive sampler.

Usage (see scripts/gene_expl.sh for the SLURM multi-node launcher):
    torchrun --nproc_per_node=4 -m cascade.explainer.run_gene_explainer_parallel \\
        --output_path ./results/importances.pkl --dataset_name AUTISM \\
        --transformer_checkpoint /path/to/ckpt.pt --models_dir ./trained_models
"""
import argparse
import gc
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from cascade.data import sampler as sampler_contrastive
from cascade.data.utils import load_and_merge_pickles
from cascade.explainer import config as cfg
from cascade.explainer.cell_level import CellLevelMLP
from cascade.explainer.checkpoint_utils import load_ddp_checkpoint
from cascade.explainer.explainer import Explainer
from cascade.explainer.save_trained_models import save_trained_model  # noqa: F401 (re-exported for callers)
from cascade.model import cascade_model

# Dataset-specific configuration for embedding extraction / gene explanation.
# Populate with your own checkpoint/collator paths per dataset.
DATASET_CONFIGS = {
    'AUTISM': {
        'merged_contexts': 'disease_cell_type_tissue',
        'batch': 2048,
        'context_specific_projections': True,
    },
    'SEATTLE': {
        'merged_contexts': 'disease_cell_type_tissue',
        'batch': 32,
        'context_specific_projections': True,
    },
    'M2': {
        'merged_contexts': 'disease_cell_type',
        'batch': 32,
        'context_specific_projections': True,
    },
}

CLASSIFICATION_TASKS = ['disease', 'cell_type']
REGRESSION_TASKS = ['S.Score', 'G2M.Score', 'pseudotime', 'pseudotime_ranks']
# M2-only task: treatment = (Cre is None) or (Cre==0 & THR==1); see Methods 9.1
M2_CLASSIFICATION_TASKS = ['treatment']

DEBUG_MODE = False
DEBUG_MAX_BATCHES = 3
CHUNK_SIZE = 50


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank, world_size, local_rank = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return 0, 1, 0
    torch.cuda.set_device(local_rank)
    # Only barriers are needed here (no gradient all-reduce), so gloo is fine and more
    # stable than nccl for multi-node inference.
    dist.init_process_group(backend=os.environ.get("DIST_BACKEND", "gloo"))
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


class IndexedDataset(Dataset):
    """Wraps a dataset and an index subset so __getitem__(i) returns dataset[indices[i]].
    Deliberately implements only __getitem__ (no __getitems__) to avoid batched-indexing
    issues with HuggingFace Arrow datasets under DistributedSampler."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


def load_model_registry(model_dir):
    registry_file = Path(model_dir) / "model_registry.pkl"
    if not registry_file.exists():
        return None
    try:
        with open(registry_file, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def load_trained_model(task, task_type, input_dim, num_classes=None, model_dir=None, device=None,
                        context_suffix=None, registry=None):
    """Load a trained cell-level MLP (see main_cell_level.py / save_trained_models.py)."""
    model_dir = Path(model_dir)
    filepath = None

    if registry is not None:
        model_id = f"{task}_none_mlp_{task_type}_context_{context_suffix}" if context_suffix else f"{task}_none_mlp_{task_type}"
        if model_id in registry:
            filepath = Path(registry[model_id]['filepath'])
            if task_type == 'classification' and num_classes is None:
                num_classes = registry[model_id].get('additional_info', {}).get('n_classes')

    if filepath is None or not filepath.exists():
        filename = f"{task}_none_mlp_{task_type}_context_{context_suffix}.pt" if context_suffix else f"{task}_none_mlp_{task_type}.pt"
        filepath = model_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Trained model not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    additional_info = checkpoint.get('additional_info', {}) or {}

    if task_type == 'classification' and num_classes is None:
        if 'n_classes' in additional_info:
            num_classes = int(additional_info['n_classes'])
        else:
            state_dict = checkpoint.get('model_state_dict', {})
            for _k, v in reversed(list(state_dict.items())):
                if hasattr(v, 'ndim') and v.ndim == 2:
                    num_classes = int(v.shape[0])
                    break
            if num_classes is None:
                raise ValueError(f"Could not infer num_classes for task {task} from checkpoint {filepath}")

    model = CellLevelMLP(input_dim, num_classes, task_type='classification').to(device) if task_type == 'classification' \
        else CellLevelMLP(input_dim, 1, task_type='regression').to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Cell-Level Gene Explainability - Parallel Multi-GPU')
    parser.add_argument('--output_path', type=str, required=True, help='Output pickle path for the importances')
    parser.add_argument('--transformer_checkpoint', type=str, required=True, help='Path to the transformer checkpoint (.pt)')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory containing trained cell-level model checkpoints')
    parser.add_argument('--data_path', type=str, default=None, help='Path to the collator/embeddings data file (.pkl)')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g. AUTISM, SEATTLE, M2)')
    parser.add_argument('--task', type=str, default=None, help='Specific task to explain (default: all cell-level tasks)')
    parser.add_argument('--context_suffix', type=str, default='all')
    parser.add_argument('--chunk_size', type=int, default=1, help='Batches processed before saving intermediate results')
    parser.add_argument('--healthy_only', action='store_true', help='Filter to only healthy donors (disease==1)')
    parser.add_argument('--global_number_to_perturb', type=int, default=100, help='Total number of genes to perturb (split between up-/down-regulated blocks)')
    parser.add_argument('--collator_root', type=str, default='', help="Optional subdirectory under the dataset directory where collator chunks live")
    parser.add_argument('--wrap_ddp', action='store_true', help='Wrap the transformer with DDP (off by default for inference stability)')
    return parser.parse_args()


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    args = parse_args()
    output_path = Path(args.output_path)

    if rank == 0:
        print("=" * 80)
        print(f"CASCADE GENE EXPLAINABILITY - PARALLEL ({world_size} GPUs, dataset={args.dataset_name})")
        print("=" * 80)

    dataset_name = args.dataset_name
    cell_level_tasks = M2_CLASSIFICATION_TASKS if dataset_name.upper() == 'M2' else ['disease', 'cell_type']
    dataset_config = DATASET_CONFIGS.get(dataset_name, {})

    output_dir = cfg.CASCADE_DATA_ROOT / dataset_name
    token_dictionary_file = output_dir / f"tokenizer_dictionary_{dataset_name}.pkl"
    merged_contexts = dataset_config.get('merged_contexts', cfg.MERGED_CONTEXTS)
    batch = dataset_config.get('batch', 2048)
    context_specific_projections = dataset_config.get('context_specific_projections', True)

    MODEL = Path(args.transformer_checkpoint)
    if not MODEL.exists():
        if rank == 0:
            print(f"❌ Transformer checkpoint not found: {MODEL}")
        cleanup_distributed()
        return

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        if rank == 0:
            print(f"❌ Models directory not found: {models_dir}")
        cleanup_distributed()
        return

    torch.manual_seed(20)
    np.random.seed(20)

    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    collator_root = output_dir / args.collator_root if args.collator_root else output_dir
    path_to_collator = Path(args.data_path) if args.data_path else None

    output_data_dict = load_and_merge_pickles(collator_root, merged_context=merged_contexts, path_to_collator=path_to_collator)

    if args.healthy_only and 'disease' in output_data_dict:
        disease_labels = output_data_dict['disease']
        if isinstance(disease_labels, torch.Tensor):
            healthy_mask = (disease_labels == 1)
            n_total = len(disease_labels)
            for key in output_data_dict.keys():
                val = output_data_dict[key]
                if isinstance(val, torch.Tensor) and len(val) == n_total:
                    output_data_dict[key] = val[healthy_mask]
                elif isinstance(val, list) and len(val) == n_total:
                    output_data_dict[key] = [val[i] for i in range(n_total) if healthy_mask[i]]

    # M2 (Methods 9.1): task-specific cell subsets, no global Cre filter.
    #   treatment: (Cre is None) OR (Cre == 0 AND THR == 1)
    #   THR:       treatment == 1 AND Cre > 0
    m2_task_indices = {}
    if dataset_name.upper() == 'M2' and all(k in output_data_dict for k in ('Cre', 'THR', 'treatment')):
        cre = np.asarray(output_data_dict['Cre'], dtype=object)
        thr_vals = np.asarray(output_data_dict['THR'])
        treat_vals = np.asarray(output_data_dict['treatment'])

        def _is_one(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return False
            try:
                return float(v) == 1
            except (TypeError, ValueError):
                return False

        def _is_zero(v):
            try:
                return float(v) == 0.0
            except (TypeError, ValueError):
                return False

        def _is_positive(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return False
            try:
                return float(v) > 0
            except (TypeError, ValueError):
                return False

        cre_none = np.array([v is None for v in cre], dtype=bool)
        cre_zero = np.array([_is_zero(v) for v in cre], dtype=bool)
        cre_pos = np.array([_is_positive(v) for v in cre], dtype=bool)
        thr_one = np.array([_is_one(v) for v in thr_vals], dtype=bool)
        treat_one = np.array([_is_one(v) for v in treat_vals], dtype=bool)

        m2_task_indices['treatment'] = np.where(cre_none | (cre_zero & thr_one))[0]
        m2_task_indices['THR'] = np.where(treat_one & cre_pos)[0]

    output_data_dict = cascade_model.SeqDataset(output_data_dict)

    with open(token_dictionary_file, "rb") as f:
        token_dictionary = pickle.load(f)
    ntoken = len(token_dictionary)

    cascade_model.set_seed(20)
    transformer_model = cascade_model.TransformerGenerator(
        d_model=cfg.D_MODEL, nhead=cfg.NHEAD, ntoken=ntoken, dim_embedding=cfg.DIM_EMBEDDING,
        nlayers=cfg.NLAYERS, vocab=token_dictionary, dropout=cfg.DROPOUT, pad_token=cfg.PAD_TOKEN,
        nclass=cfg.NCLASS, cell_emb_style=cfg.CELL_EMB_STYLE, extract_embeddings=True,
        context_specific_projections=context_specific_projections, DA=cfg.DA, merged_contexts=merged_contexts,
    ).to(device)

    try:
        model_dict = torch.load(MODEL, map_location=device)
        transformer_model.load_state_dict(model_dict["model_state_dict"])
        if rank == 0:
            print("✅ Transformer model loaded (direct load)")
    except Exception:
        _, transformer_model = load_ddp_checkpoint(checkpoint_path=MODEL, model=transformer_model, device=device, verbose=(rank == 0))

    if args.wrap_ddp and world_size > 1:
        transformer_model = DDP(transformer_model, device_ids=[local_rank])
    transformer_model.eval()

    if dataset_name.upper() == 'M2' and m2_task_indices:
        loader_by_task = {}
        for task_name, indices in m2_task_indices.items():
            if len(indices) == 0:
                continue
            task_dataset = IndexedDataset(output_data_dict, indices)
            condition_fn = (lambda batch: True) if task_name == 'THR' else sampler_contrastive.condition_samples
            loader_by_task[task_name] = cascade_model.prepare_dataloader(
                data_pt=task_dataset, batch_size=batch, shuffle=False, num_workers=0,
                condition_fn=condition_fn, sampler=True, class_label=None,
            )
        loader_all = None
    else:
        loader_by_task = {}
        loader_all = cascade_model.prepare_dataloader(
            data_pt=output_data_dict, batch_size=batch, shuffle=False, num_workers=0,
            condition_fn=sampler_contrastive.condition_samples, sampler=True, class_label=None,
        )

    model_registry = load_model_registry(models_dir)
    tasks_to_process = [args.task] if args.task else [t for t in cell_level_tasks if t not in REGRESSION_TASKS]

    loaded_models = {}
    for task in tasks_to_process:
        if task in CLASSIFICATION_TASKS or (dataset_name.upper() == 'M2' and task in M2_CLASSIFICATION_TASKS):
            task_type = 'classification'
        elif task in REGRESSION_TASKS:
            task_type = 'regression'
        else:
            continue
        try:
            model = load_trained_model(
                task=task, task_type=task_type, input_dim=cfg.DIM_EMBEDDING, model_dir=models_dir,
                device=device, context_suffix=args.context_suffix, registry=model_registry,
            )
            loaded_models[task] = {'model': model, 'task_type': task_type}
            if rank == 0:
                print(f"  ✅ Loaded model for {task}")
        except FileNotFoundError as e:
            if rank == 0:
                print(f"  ❌ Model not found for task '{task}': {e}")

    if not loaded_models:
        if rank == 0:
            print("❌ No models loaded. Exiting.")
        cleanup_distributed()
        return

    if dist.is_initialized():
        dist.barrier()

    all_importances = {}
    for task, model_info in loaded_models.items():
        current_loader = loader_by_task.get(task) if (dataset_name.upper() == 'M2' and loader_by_task) else loader_all
        if current_loader is None:
            continue

        if rank == 0:
            print(f"\n{'='*80}\nTASK: {task} ({model_info['task_type']})\n{'='*80}")

        unwrapped_transformer = transformer_model.module if hasattr(transformer_model, 'module') else transformer_model
        explainer = Explainer(
            model=unwrapped_transformer, clf_model=model_info['model'], token_dictionary=token_dictionary,
            CCE=True, temperature=cfg.TEMPERATURE, device=device, CLASS=False, nclass=cfg.NCLASS,
            cell_level=True, donor_level_tasks=[],
        )

        all_importances[task] = {
            'task_name': task, 'task_type': model_info['task_type'], 'importances': [], 'batch_info': [],
            'input_ids': [], 'cell_type': [], 'tissue': [], 'disease': [],
            'perturbation_info': {'mode': 'by_global_position', 'global_number_to_perturb': args.global_number_to_perturb},
        }

        total_batches = len(current_loader)
        uses_distributed_sampler = isinstance(getattr(current_loader, "sampler", None), DistributedSampler)
        total_batches_per_rank = (
            (total_batches + world_size - 1 - rank) // world_size if (world_size > 1 and not uses_distributed_sampler) else total_batches
        )
        pbar = tqdm(total=total_batches_per_rank, desc=f"[Rank {rank}] {task}") if rank == 0 else None

        chunk_count = 0
        processed_batches = 0
        for batch_idx, batch_data_eval in enumerate(current_loader):
            if world_size > 1 and not uses_distributed_sampler and (batch_idx % world_size != rank):
                continue
            processed_batches += 1
            if DEBUG_MODE and processed_batches > DEBUG_MAX_BATCHES:
                break

            try:
                _, importances = explainer.forward(
                    batch_data_eval, cached_prediction=None, perturbation_idx=-1,
                    global_number_to_perturb=args.global_number_to_perturb, robust=True,
                )
                if importances is not None:
                    importances_np = importances.detach().cpu().numpy()
                    all_importances[task]['importances'].append(importances_np)
                    all_importances[task]['batch_info'].append({
                        'batch_idx': batch_idx, 'task': task, 'shape': importances_np.shape,
                        'n_samples': importances_np.shape[0] if importances_np.ndim > 0 else 1, 'rank': rank,
                    })
                    if isinstance(batch_data_eval, dict) and 'input_ids' in batch_data_eval:
                        input_ids_tensor = batch_data_eval['input_ids']
                        if isinstance(input_ids_tensor, torch.Tensor):
                            all_importances[task]['input_ids'].append(input_ids_tensor.detach().cpu().numpy())
                    if isinstance(batch_data_eval, dict):
                        for key in ('cell_type', 'tissue', 'disease'):
                            if key not in batch_data_eval:
                                continue
                            val = batch_data_eval[key]
                            arr = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
                            all_importances[task][key].append(arr.tolist() if arr.ndim >= 1 else [arr.item()])

                torch.cuda.empty_cache()

                if (processed_batches % args.chunk_size == 0) or (processed_batches == total_batches_per_rank):
                    chunk_count += 1
                    chunk_output_path = output_path.parent / f"{output_path.stem}_rank_{rank:02d}_{task}_{args.context_suffix}_chunk_{chunk_count:03d}.pkl"
                    with open(chunk_output_path, 'wb') as f:
                        pickle.dump({
                            'task_data': {task: all_importances[task]},
                            'metadata': {
                                'rank': rank, 'task': task, 'world_size': world_size, 'chunk_number': chunk_count,
                                'dataset_name': dataset_name, 'merged_contexts': merged_contexts, 'batch_size': batch,
                                'context_suffix': args.context_suffix, 'transformer_checkpoint': str(MODEL),
                            },
                        }, f)
                    gc.collect()
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ⚠️ [Rank {rank}] Error processing batch {batch_idx} for task {task}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print(f"\n🎉 Explainability analysis completed successfully!")
        print(f"📄 Results saved with pattern: {output_path.parent}/{output_path.stem}_rank_*_*.pkl")

    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        cleanup_distributed()
        raise SystemExit(1)
