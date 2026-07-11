"""
DistributedDataParallel training entrypoint for CASCADE pre-training. Wraps
`cascade.training.trainer`'s parser and train() loop with DDP setup, dataset
sharding, and checkpoint resume/save. This is the entrypoint used for the
paper's runs (see scripts/hlca_multinode.sh).
"""
import os
import time
import logging
from pathlib import Path
import pickle
import wandb
import numpy as np

# Increase Gloo barrier timeout to prevent premature timeout during data loading
# Default is 1800s (30 min), increasing to 7200s (2 hours)
os.environ.setdefault('GLOO_BARRIER_TIMEOUT', '7200')

# Determine ranks early to configure WANDB before importing modules that use it
_DEF_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
_DEF_RANK = int(os.environ.get("RANK", 0))
_DEF_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

if _DEF_RANK != 0:
    os.environ["WANDB_DISABLED"] = "true"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset

from cascade.training import trainer

# Root directories for tokenized datasets and checkpoints. Override via
# environment variables to run outside the original HPC cluster layout.
CASCADE_DATA_ROOT = Path(os.environ.get("CASCADE_DATA_ROOT", "/n/holylfs06/LABS/mzitnik_lab/Lab/vgiunchiglia/DATASET"))
CASCADE_CKPT_ROOT = Path(os.environ.get("CASCADE_CKPT_ROOT", "/n/netscratch/mzitnik_lab/Lab/oqueen/CASCADE-checkpoints"))


def init_distributed():
    """Initialize torch.distributed from environment variables set by torchrun."""
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")


def get_rank_info():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    return rank, world_size, local_rank


def shard_dataset_evenly(dataset, world_size, rank):
    """Create a Subset view that shards the dataset evenly by rank.

    This avoids modifying internal dataloader logic while ensuring each process sees a disjoint shard.
    """
    if world_size <= 1:
        return dataset
    total_len = len(dataset)
    indices = list(range(rank, total_len, world_size))
    return Subset(dataset, indices)


def filter_raw_data_by_cre(raw_data_dict):
    """
    Filter raw_data_dict to keep only rows where Cre is "bigger than 0".
    Cre values can be None or strings (e.g. numeric strings like "0.5", "1").
    Used for the M2 mouse-thyroid dataset, where Cre-transcript detection
    marks cells with active DN-THR/WT-THR expression (Methods 9.1).

    Modifies raw_data_dict in place by subsetting all array-like values.
    """
    if 'Cre' not in raw_data_dict:
        raise ValueError("M2 dataset requires 'Cre' key in raw_data_dict")
    cre = np.asarray(raw_data_dict['Cre'], dtype=object)
    n = len(cre)

    def keep_row(v):
        if v is None or (isinstance(v, float) and (v != v)):
            return False
        if isinstance(v, str) and v.strip() == '':
            return False
        try:
            return float(v) > 0
        except (TypeError, ValueError):
            return False

    mask = np.array([keep_row(v) for v in cre], dtype=bool)
    n_kept = int(mask.sum())
    print(f"\n[M2] Cre filter: keeping {n_kept} / {n} cells (Cre > 0); filtered out {n - n_kept} cells")

    for key in list(raw_data_dict.keys()):
        val = raw_data_dict[key]
        if isinstance(val, torch.Tensor) and val.shape[0] == n:
            raw_data_dict[key] = val[mask]
        elif isinstance(val, np.ndarray) and val.shape[0] == n:
            raw_data_dict[key] = val[mask]
        elif isinstance(val, list) and len(val) == n:
            raw_data_dict[key] = [val[i] for i in np.where(mask)[0]]
    return raw_data_dict


class ProgressTrackingDataLoader:
    """Wraps a dataloader to periodically log within-epoch progress to W&B."""

    def __init__(self, loader, epoch, epoch_log_interval, rank, world_size, use_wandb, nepochs):
        self.loader = loader
        self.epoch = epoch
        self.epoch_log_interval = epoch_log_interval
        self.rank = rank
        self.world_size = world_size
        self.use_wandb = use_wandb
        self.nepochs = nepochs

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        total_batches = len(self.loader)
        for batch_idx, batch in enumerate(self.loader):
            if (self.rank == 0 or self.world_size == 1) and self.use_wandb and batch_idx % self.epoch_log_interval == 0:
                total_batches_global = total_batches * int(self.world_size)
                estimated_global_batches_processed = (batch_idx + 1) * int(self.world_size)
                epoch_progress_total = 0.0
                if total_batches_global > 0 and self.nepochs > 0:
                    iterations_done_global = int(self.epoch) * total_batches_global + estimated_global_batches_processed
                    total_iterations_all_epochs = total_batches_global * max(1, int(self.nepochs))
                    epoch_progress_total = float(iterations_done_global) / float(total_iterations_all_epochs)

                wandb.log({
                    "epoch": self.epoch,
                    "epoch_progress": epoch_progress_total,
                    "epoch_progress_local": float(batch_idx) / float(total_batches) if total_batches > 0 else 0.0,
                    "epoch_batch": batch_idx,
                    "total_batches": total_batches,
                })
            yield batch


def find_latest_checkpoint(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("**/*.pt"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(checkpoint_path: Path, model, optimizer, scheduler, scaler, device, logger, rank=0):
    """Loads a checkpoint and returns (start_epoch, global_step, loaded_args)."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state = checkpoint["model_state_dict"]
        if hasattr(model, "module"):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        loaded_args = checkpoint.get("args", {})
        return start_epoch, global_step, loaded_args
    except Exception as e:
        if rank == 0:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return 0, 0, {}


def validate_checkpoint_compatibility(loaded_args, current_args, logger, rank=0):
    """Checks that architecture-defining args match between a checkpoint and the current run."""
    architecture_keys = ["d_model", "nhead", "dim_embedding", "nlayers", "cell_emb_style", "context_specific_projections"]
    current_args_dict = vars(current_args)
    for key in architecture_keys:
        if key in loaded_args and key in current_args_dict and loaded_args[key] != current_args_dict[key]:
            if rank == 0:
                logger.error(f"Checkpoint mismatch on '{key}': checkpoint={loaded_args[key]}, current={current_args_dict[key]}")
            return False
    return True


def generate_wandb_run_name(args, timestamp, rank=0):
    name_parts = []
    if hasattr(args, 'dataset') and args.dataset:
        name_parts.append(f"ds-{args.dataset}")
    name_parts.append(f"L{args.nlayers}")
    name_parts.append(f"H{args.nhead}")
    name_parts.append(f"D{args.d_model}")
    name_parts.append(f"bs{args.batch}")
    name_parts.append(f"lr{args.lr}")
    if hasattr(args, 'CLASS') and args.CLASS:
        name_parts.append("CLASS")
    if hasattr(args, 'DA') and args.DA:
        name_parts.append("DA")
    if hasattr(args, 'context_specific_projections') and args.context_specific_projections:
        name_parts.append("CSP")
    if hasattr(args, 'cell_emb_style') and args.cell_emb_style != "cls":
        name_parts.append(f"emb-{args.cell_emb_style}")
    return f"{'_'.join(name_parts)}_{timestamp}"


def train_with_periodic_logging(model, loader, loader_eval, vocab, scaler, optimizer, scheduler,
                                 device, logger, epoch, temperature, log_interval, merged_contexts,
                                 CLASS, criterion_class, args, OUTPUT_DIR, rank, world_size, global_step):
    """Wrapper function that adds periodic epoch logging to the training process."""
    total_batches = len(loader)
    epoch_log_interval = getattr(args, 'epoch_log_interval', 100)

    if (rank == 0 or world_size == 1) and args.use_wandb:
        total_batches_global = len(loader) * int(world_size) if total_batches > 0 else 0
        epoch_progress_total = 0.0
        if total_batches_global > 0 and getattr(args, 'nepochs', 1) > 0:
            iterations_done_global = int(epoch) * total_batches_global
            total_iterations_all_epochs = total_batches_global * max(1, int(args.nepochs))
            epoch_progress_total = float(iterations_done_global) / float(total_iterations_all_epochs)

        wandb.log({
            "epoch": epoch,
            "epoch_progress": epoch_progress_total,
            "epoch_progress_local": 0.0,
            "epoch_batch": 0,
            "total_batches": total_batches,
        })

    progress_loader = ProgressTrackingDataLoader(
        loader, epoch, epoch_log_interval, rank, world_size, args.use_wandb, args.nepochs
    )

    updated_global_step = trainer.train(
        model=model,
        loader=progress_loader,
        loader_eval=loader_eval,
        vocab=vocab,
        scaler=scaler,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        pad_token=args.pad_token,
        epoch=epoch,
        temperature=temperature,
        log_interval=log_interval,
        merged_contexts=merged_contexts,
        CLASS=CLASS,
        criterion_class=criterion_class,
        args=args,
        OUTPUT_DIR=OUTPUT_DIR,
        rank=rank,
        world_size=world_size,
        global_step=global_step,
    )

    if (rank == 0 or world_size == 1) and args.use_wandb:
        total_batches_global = len(loader) * int(world_size) if total_batches > 0 else 0
        iterations_done_global = int(epoch) * total_batches_global + int(total_batches)
        total_iterations_all_epochs = total_batches_global * max(1, int(args.nepochs)) if total_batches_global > 0 else 0
        epoch_progress_total = float(iterations_done_global) / float(total_iterations_all_epochs) if total_iterations_all_epochs > 0 else 1.0

        wandb.log({
            "epoch": epoch,
            "epoch_progress": epoch_progress_total,
            "epoch_progress_local": 1.0,
            "epoch_batch": total_batches,
            "total_batches": total_batches,
        })

    return updated_global_step


def ddp_main():
    if "local_rank" not in trainer.parser._option_string_actions:
        trainer.parser.add_argument("--local_rank", type=int, default=_DEF_LOCAL_RANK)

    trainer.parser.add_argument("--dl_num_workers", type=int, default=0)
    trainer.parser.add_argument("--epoch_log_interval", type=int, default=100, help="Log epoch progress every N batches")
    trainer.parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    trainer.parser.add_argument("--auto_resume", action="store_true", help="Automatically resume from the latest checkpoint in CKPT_DIR")

    args = trainer.parser.parse_args()

    logger = logging.getLogger("CASCADE")

    init_distributed()
    rank, world_size, local_rank = get_rank_info()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Ensure only rank 0 uses W&B visibly; others remain disabled regardless of args.use_wandb
    if rank != 0:
        args.use_wandb = False
        if getattr(args, "val", True):
            args.val = False

    DATASET_NAME = args.dataset
    timestamp = time.strftime(f"{DATASET_NAME}_%Y%m%d-%H%M%S")
    OUTPUT_DIR = CASCADE_DATA_ROOT / DATASET_NAME
    CKPT_DIR = CASCADE_CKPT_ROOT / DATASET_NAME / f"ckpt_{timestamp}"
    if args.collator_root:
        CKPT_DIR = CASCADE_CKPT_ROOT / DATASET_NAME / args.collator_root / f"ckpt_{timestamp}"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    TOKEN_DICTIONARY_FILE = OUTPUT_DIR / f"tokenizer_dictionary_{DATASET_NAME}.pkl"

    donors_split = getattr(trainer.P3_donors_split_metadata, DATASET_NAME) if args.val else None

    trainer.model_generative.set_seed(args.seed)

    with open(TOKEN_DICTIONARY_FILE, "rb") as input_file:
        token_dictionary = pickle.load(input_file)

    ntoken = len(token_dictionary)
    merged_contexts = "_".join(args.contexts)

    partial = not (args.partial == "False" or args.partial == "false" or args.partial is False)

    only_contrastive = not args.CLASS

    model = trainer.model_generative.TransformerGenerator(
        d_model=args.d_model,
        nhead=args.nhead,
        ntoken=ntoken,
        dim_embedding=args.dim_embedding,
        nlayers=args.nlayers,
        vocab=token_dictionary,
        dropout=args.dropout,
        pad_token=args.pad_token,
        nclass=args.nclass,
        cell_emb_style=args.cell_emb_style,
        context_specific_projections=args.context_specific_projections,
        constant_ctx=args.constant_ctx,
        only_contrastive=only_contrastive,
        DA=args.DA,
        lambda_sinkhorn=args.lambda_sinkhorn,
        merged_contexts=merged_contexts,
    ).to(device)

    if torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(model)

    criterion_class = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.schedule_ratio)

    if args.pretrained:
        try:
            model_dict = torch.load(args.pretrained, map_location=torch.device('cpu'))
            model.module.load_state_dict(model_dict["model_state_dict"])
            logger.info(f"Loading all model params from {args.pretrained}")
        except Exception:
            current_state = model.module.state_dict()
            pretrained_dict = torch.load(args.pretrained, map_location=torch.device('cpu'))["model_state_dict"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in current_state and v.shape == current_state[k].shape}
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            current_state.update(pretrained_dict)
            model.module.load_state_dict(current_state)
            logger.warning("Only subset of pre-trained model weights were loaded")
        try:
            optimizer.load_state_dict(torch.load(args.pretrained, map_location=torch.device('cpu'))["optimizer_state_dict"])
        except ValueError as e:
            print(f"Skipping optimizer state loading due to mismatch: {e}")

    start_epoch = 0
    global_step = 0

    checkpoint_path = None
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
    elif args.auto_resume:
        checkpoint_path = find_latest_checkpoint(CKPT_DIR)
        if checkpoint_path and rank == 0:
            logger.info(f"Auto-resuming from latest checkpoint: {checkpoint_path}")

    if checkpoint_path:
        if rank == 0:
            logger.info(f"Attempting to resume from checkpoint: {checkpoint_path}")

        start_epoch, global_step, loaded_args = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, device, logger, rank
        )

        if start_epoch > 0:
            if not validate_checkpoint_compatibility(loaded_args, args, logger, rank):
                if rank == 0:
                    logger.error("Checkpoint is incompatible with current configuration. Aborting.")
                return
            if rank == 0:
                logger.info(f"Resuming training from epoch {start_epoch}, global step {global_step}")
        else:
            if rank == 0:
                logger.warning("Failed to load checkpoint, starting training from scratch")
    else:
        if rank == 0:
            logger.info("No checkpoint specified or found, starting training from scratch")

    if args.use_wandb and rank == 0:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = args.wandb_run_name or generate_wandb_run_name(args, timestamp, rank)

        wandb_config = {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "dim_embedding": args.dim_embedding,
            "dropout": args.dropout,
            "cell_emb_style": args.cell_emb_style,
            "lr": args.lr,
            "schedule_ratio": args.schedule_ratio,
            "nepochs": args.nepochs,
            "contexts": args.contexts,
            "nlayers": args.nlayers,
            "batch": args.batch,
            "CLASS": args.CLASS,
            "partial-tuning": partial,
            "temperature": args.temperature,
            "context_specific_projections": args.context_specific_projections,
            'val': args.val,
            'DA': args.DA,
            'dataset': args.dataset,
            'lambda_sinkhorn': args.lambda_sinkhorn,
            'world_size': world_size,
            'global_rank': rank,
            'seed': args.seed,
        }

        wandb_kwargs = {"project": "sc-cascade", "name": run_name, "config": wandb_config}
        if args.wandb_run_group:
            wandb_kwargs["group"] = args.wandb_run_group
        if args.wandb_tags:
            wandb_kwargs["tags"] = args.wandb_tags
        if start_epoch > 0:
            wandb_kwargs["notes"] = f"Resumed from epoch {start_epoch-1}, global step {global_step}"
            wandb_kwargs["tags"] = wandb_kwargs.get("tags", []) + ["resumed"]

        logger.info(f"Initializing wandb with run name: {run_name}")
        wandb.init(**wandb_kwargs)

    if args.donors:
        path_to_collator = OUTPUT_DIR / f"collator_{merged_contexts}_kempner_metadata_final_corrected.pkl"
    else:
        path_to_collator = OUTPUT_DIR / f"collator_{merged_contexts}_kempner_metadata.pkl"

    collator_root = OUTPUT_DIR / args.collator_root if args.collator_root else OUTPUT_DIR

    if args.chunks_collator:
        raw_data_dict = trainer.utils.load_and_merge_pickles(
            directory=collator_root, path_to_collator=path_to_collator, merged_context=merged_contexts,
        )
        if DATASET_NAME == 'M2':
            filter_raw_data_by_cre(raw_data_dict)
        output_data_dict = trainer.model_generative.SeqDataset(raw_data_dict)
    else:
        # `dataset` must already be a tokenized+shuffled HuggingFace Dataset assembled by
        # the preprocessing pipeline (see preprocessing/) and loaded before this point.
        collator = trainer.DataCollatorContrastiveLearning(
            max_length_up=args.max_length_up, max_length_down=args.max_length_down, path_to_collator=path_to_collator,
        )
        raw_data_dict = collator(dataset)
        if DATASET_NAME == 'M2':
            filter_raw_data_by_cre(raw_data_dict)
        output_data_dict = trainer.model_generative.SeqDataset(raw_data_dict)
        torch.save(output_data_dict, path_to_collator)

    output_data_dict_val = None

    if args.CLASS:
        if args.class_task in ["disease", "cell_type", 'tissue']:
            y_disease = trainer.model_generative.detect_labels_class(output_data_dict, context="disease", dataset=DATASET_NAME)
            output_data_dict.y_disease = y_disease
            y_celltype = trainer.model_generative.detect_labels_class(output_data_dict, context="cell_type", cell_type=args.ct_class, dataset=DATASET_NAME)
            output_data_dict.y_celltype = y_celltype
            y_class = trainer.model_generative.detect_labels_class(output_data_dict, context=args.class_task, cell_type=args.ct_class, dataset=DATASET_NAME)
            output_data_dict.y_class = y_class
            output_data_dict.add_label(y_class, y_disease, y_celltype)
            output_data_dict.filter_data()
        else:
            output_data_dict.integer_encoding(args.class_task)

    # NOTE: sharding is left to the distributed sampler rather than a manual Subset,
    # since Subset can interact poorly with samplers that expect list-style indexing
    # on the original dataset.
    output_data_dict_shard = output_data_dict

    if rank == 0:
        print(f"Rank {rank}: [{time.time():.2f}] Starting training dataloader creation")

    if args.sampler:
        dataloader = trainer.model_generative.prepare_dataloader(
            data_pt=output_data_dict_shard,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.dl_num_workers,
            condition_fn=trainer.sampler_contrastive.condition_samples,
            sampler=True,
            class_label=args.class_task,
        )
    else:
        dataloader = trainer.model_generative.prepare_dataloader(
            data_pt=output_data_dict_shard,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.dl_num_workers,
            sampler=False,
            class_label=args.class_task,
        )

    if rank == 0:
        print(f"Rank {rank}: [{time.time():.2f}] Finished training dataloader creation")

    if dist.is_initialized():
        dist.barrier()

    dataloader_val = None
    if args.val and rank == 0:
        print(f"Rank {rank}: [{time.time():.2f}] Starting validation data preparation")

        trainer.model_generative.set_seed(args.seed)

        if args.donors:
            path_to_train_collator = OUTPUT_DIR / f"collator_train_{merged_contexts}_kempner_metadata_final_corrected.pkl"
            path_to_val_collator = OUTPUT_DIR / f"collator_val_{merged_contexts}_kempner_metadata_final_corrected.pkl"
        else:
            path_to_train_collator = OUTPUT_DIR / f"collator_train_{merged_contexts}_kempner_metadata.pkl"
            path_to_val_collator = OUTPUT_DIR / f"collator_val_{merged_contexts}_kempner_metadata.pkl"

        if path_to_train_collator.exists():
            with open(path_to_train_collator, "rb") as f:
                output_data_dict = pickle.load(f)
            with open(path_to_val_collator, "rb") as f:
                output_data_dict_val = pickle.load(f)
        else:
            if args.donors:
                output_data_dict.split_by_donors_id(donor_dict=donors_split)
            else:
                output_data_dict.split_by_donors_id()
            output_data_dict_val = trainer.copy.deepcopy(output_data_dict)
            output_data_dict.split_by_donors(train=True)
            output_data_dict_val.split_by_donors(train=False)
            if not args.chunks_collator:
                torch.save(output_data_dict, path_to_train_collator)
                torch.save(output_data_dict_val, path_to_val_collator)

        if args.CLASS:
            if args.class_task in ["disease", "cell_type", 'tissue']:
                y_disease = trainer.model_generative.detect_labels_class(output_data_dict_val, context="disease", dataset=DATASET_NAME)
                output_data_dict_val.y_disease = y_disease
                y_celltype = trainer.model_generative.detect_labels_class(output_data_dict_val, context="cell_type", cell_type=args.ct_class, dataset=DATASET_NAME)
                output_data_dict_val.y_celltype = y_celltype
                y_class = trainer.model_generative.detect_labels_class(output_data_dict_val, context=args.class_task, cell_type=args.ct_class, dataset=DATASET_NAME)
                output_data_dict_val.y_class = y_class
                output_data_dict_val.add_label(y_class, y_disease, y_celltype)
            else:
                output_data_dict_val.integer_encoding(args.class_task)

        print(f"Rank {rank}: [{time.time():.2f}] Starting validation dataloader creation")

        if args.sampler:
            dataloader_val = trainer.model_generative.prepare_dataloader(
                data_pt=output_data_dict_val,
                batch_size=args.batch,
                shuffle=True,
                num_workers=args.dl_num_workers,
                condition_fn=trainer.sampler_contrastive.condition_samples,
                sampler=True,
                class_label=args.class_task,
            )
        else:
            dataloader_val = trainer.model_generative.prepare_dataloader(
                data_pt=output_data_dict_val,
                batch_size=args.batch_eval,
                shuffle=True,
                num_workers=args.dl_num_workers,
                sampler=False,
                class_label=args.class_task,
            )

        print(f"Rank {rank}: [{time.time():.2f}] Finished validation dataloader creation")

    if dist.is_initialized():
        if rank == 0:
            print(f"Rank {rank}: [{time.time():.2f}] Waiting at final barrier before training")
        dist.barrier()
        if rank == 0:
            print(f"Rank {rank}: [{time.time():.2f}] All ranks synchronized and ready to train")
            if start_epoch > 0:
                logger.info(f"Resuming training from epoch {start_epoch}/{args.nepochs}, global step {global_step}")
            else:
                logger.info(f"Starting training from epoch 0/{args.nepochs}")

    try:
        for epoch in range(start_epoch, args.nepochs):
            print("Training epoch: ", epoch)
            global_step = train_with_periodic_logging(
                model=model,
                loader=dataloader,
                loader_eval=dataloader_val,
                vocab=token_dictionary,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                logger=logger,
                epoch=epoch,
                temperature=args.temperature,
                log_interval=args.log_interval,
                merged_contexts=merged_contexts,
                CLASS=args.CLASS,
                criterion_class=criterion_class,
                args=args,
                OUTPUT_DIR=CKPT_DIR,
                rank=rank,
                world_size=world_size,
                global_step=global_step,
            )

            if dist.is_initialized():
                dist.barrier()
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    ddp_main()
