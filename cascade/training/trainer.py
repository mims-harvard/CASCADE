"""
Shared CLI parser, checkpointing helpers, and the train()/test() epoch loops
used by cascade/training/train_ddp.py. Implements the context-specific
contrastive pre-training objective and optional classification fine-tuning
(Methods 9.10-9.11).
"""
import argparse
import copy
import threading
import time
import time as time_module
import warnings

import numpy as np
import psutil
import torch
import torch.distributed as dist
import wandb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.profiler import record_function
from torch.utils.data import DataLoader

from cascade.data import sampler as sampler_contrastive
from cascade.data import splits as P3_donors_split_metadata
from cascade.data import utils
from cascade.data.collator import DataCollatorContrastiveLearning
from cascade.model import cascade_model as model_generative

warnings.filterwarnings("ignore")

PAD_TOKEN = "<pad>"
START_TIME = None

parser = argparse.ArgumentParser(description="Training script for CASCADE")
parser.add_argument('--pretrained', type=str, help='path to pretrained model')
parser.add_argument('--class_task', type=str, default="sex", help='downstream classification target')
parser.add_argument('--ct_class', type=str, help='cell type used for one-vs-rest classification tasks')
parser.add_argument('--seed', type=int, default=20, help='seed number')
parser.add_argument('--d_model', type=int, default=384, help='model dimension')
parser.add_argument('--list_data', nargs='+', type=str, help='list of dataset directories')
parser.add_argument('--nhead', type=int, default=6, help='number of attention heads')
parser.add_argument('--dim_embedding', type=int, default=384, help='embedding dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
parser.add_argument('--nlayers', type=int, default=12, help='number of encoder layers')
parser.add_argument('--cell_emb_style', type=str, default="cls", help='method to derive cell embedding')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--chunks_collator', type=bool, default=True, action=argparse.BooleanOptionalAction, help='read pre-chunked collator output from disk')
parser.add_argument('--use_wandb', type=bool, default=True, action=argparse.BooleanOptionalAction, help='wandb as logger')
parser.add_argument('--nepochs', type=int, default=500, help='number of epochs')
parser.add_argument('--schedule_ratio', type=float, default=0.9, help='LR scheduler decay ratio')
parser.add_argument('--contexts', nargs='+', type=str, default=["disease"], help='list of contexts to use')
parser.add_argument('--pad_token', type=str, default=PAD_TOKEN, help='pad token')
parser.add_argument('--temperature', type=float, default=0.1, help='contrastive loss temperature')
parser.add_argument('--log_interval', type=int, default=2, help='logging interval for logger')
parser.add_argument('--CLASS', type=bool, default=False, action=argparse.BooleanOptionalAction, help='whether to do the classification task')
parser.add_argument('--constant_ctx', type=bool, default=False, action=argparse.BooleanOptionalAction, help='whether to keep the other context dimensions constant or not')
parser.add_argument('--partial', type=bool, default=False, help='whether to fine-tune the full model or only the head')
parser.add_argument('--val', type=bool, default=False, action=argparse.BooleanOptionalAction, help='whether to use a validation dataset split by donors in the classification task')
parser.add_argument('--batch', type=int, default=256, help='batch size')
parser.add_argument('--batch_eval', type=int, default=60, help='eval batch size')
parser.add_argument('--max_length_up', type=int, default=1024, help='max up-regulated genes kept per cell')
parser.add_argument('--max_length_down', type=int, default=1024, help='max down-regulated genes kept per cell')
parser.add_argument('--nclass', type=int, default=2, help='number of classes for the classification head')
parser.add_argument('--iteration', type=int, default=2, help='number of sampling iterations')
parser.add_argument('--sampler', type=bool, default=True, action=argparse.BooleanOptionalAction, help='use custom sampler to avoid batches without negative/positive samples')

parser.add_argument("--no_save_checkpoints", action="store_true", help="switch to turn off checkpoint saving, for testing")
parser.add_argument("--track_memory", action="store_true", help="track CPU memory usage during training")
parser.add_argument("--profile", action="store_true", help="enable PyTorch profiler for performance analysis")
parser.add_argument("--checkpoint_frequency", type=int, default=1000, help="save checkpoint every N training steps")

parser.add_argument('--lambda_sinkhorn', type=float, default=0.005, help='domain adaptation loss weight')
parser.add_argument('--dataset', type=str, default="AUTISM", help='dataset name, must be a key in cascade.data.splits.SPLITS_BY_DATASET')
parser.add_argument("--donors", action="store_true", help="use donor-level metadata-corrected collator paths")
parser.add_argument("--DA", action="store_true", help="enable Sinkhorn domain adaptation on the disease projection space")
parser.add_argument("--context_specific_projections", action="store_true", help="If true, breaks the embedding space down into per-context embedding spaces, instead of one")
parser.add_argument("--wandb_run_name", type=str, default=None, help="Custom name for wandb run")
parser.add_argument("--wandb_run_group", type=str, default=None, help="Group name for wandb runs")
parser.add_argument("--wandb_tags", nargs='+', type=str, default=None, help="Tags for wandb run (e.g., --wandb_tags exp1 baseline)")

parser.add_argument("--speed_bench", action="store_true", help="Runs small experiment to test iteration speed")
parser.add_argument("--collator_root", type=str, default="", help="Optional subdirectory under the dataset directory where collator chunks live. If empty (default), use the dataset directory directly.")


class MemoryTracker:
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.memory_log = []
        self.running = False
        self.thread = None

    def start_tracking(self):
        self.running = True
        self.thread = threading.Thread(target=self._track_memory)
        self.thread.daemon = True
        self.thread.start()

    def stop_tracking(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _track_memory(self):
        while self.running:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_data = {
                    'timestamp': time_module.time(),
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024,
                    'memory_percent': process.memory_percent(),
                    'cpu_percent': process.cpu_percent(),
                }
                self.memory_log.append(memory_data)
                if wandb.run is not None:
                    wandb.log({
                        'memory_rss_mb': memory_data['rss_mb'],
                        'memory_vms_mb': memory_data['vms_mb'],
                        'memory_percent': memory_data['memory_percent'],
                        'cpu_percent': memory_data['cpu_percent'],
                    })
                time_module.sleep(self.log_interval)
            except Exception as e:
                print(f"Memory tracking error: {e}")
                time_module.sleep(self.log_interval)

    def get_current_memory(self):
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent(),
            }
        except Exception as e:
            print(f"Error getting memory info: {e}")
            return None


def save_checkpoint(state, ckpt_dir, filename: str):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / filename
    torch.save(state, ckpt_path)
    return ckpt_path


def train(
    model,
    loader: DataLoader,
    vocab,
    scaler,
    optimizer,
    scheduler,
    device,
    logger,
    epoch,
    pad_token,
    temperature,
    log_interval,
    merged_contexts,
    criterion_class,
    args,
    OUTPUT_DIR,
    loader_eval=None,
    CLASS=True,
    weight_dis=None,
    weight_cell=None,
    memory_tracker=None,
    profiler_obj=None,
    rank=0,
    world_size=1,
    global_step=0,
) -> int:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_loss_epoch = 0.0
    total_loss_epoch_disease = 0.0
    total_loss_epoch_tissue = 0.0
    total_loss_epoch_cell_type = 0.0
    total_auroc_epoch_class = 0.0
    total_f1_epoch_class = 0
    total_aproc_epoch_class = 0.0
    total_loss_epoch_da = 0.0
    start_time = time.time()
    best_auroc = 0

    num_batches = len(loader)

    for batch, batch_data in enumerate(loader):
        class_results_roc = {}
        class_results_rpc = {}

        if memory_tracker and batch % 10 == 0:
            current_memory = memory_tracker.get_current_memory()
            if current_memory:
                logger.info(
                    f"Memory usage at batch {batch}: "
                    f"RSS={current_memory['rss_mb']:.1f}MB, "
                    f"VMS={current_memory['vms_mb']:.1f}MB, "
                    f"CPU={current_memory['cpu_percent']:.1f}%"
                )
                if args.use_wandb and wandb.run is not None:
                    wandb.log({
                        'batch_memory_rss_mb': current_memory['rss_mb'],
                        'batch_memory_vms_mb': current_memory['vms_mb'],
                        'batch_cpu_percent': current_memory['cpu_percent'],
                        'batch_number': batch,
                    })

        input_gene_ids = torch.tensor(batch_data["input_ids"]).to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])

        CCE = not CLASS

        forward_kwargs = dict(
            data=batch_data,
            src=input_gene_ids,
            src_key_padding_mask=src_key_padding_mask,
            CCE=CCE,
            temperature=temperature,
            device=device,
            CLASS=CLASS,
            nclass=args.nclass,
            n_batch=batch,
        )
        if profiler_obj is not None:
            with record_function("model_forward"):
                loss_cce_dict, cell_emb_save, soft_out, _, _ = model(**forward_kwargs)
        else:
            loss_cce_dict, cell_emb_save, soft_out, _, _ = model(**forward_kwargs)

        if CLASS:
            if args.class_task == "cell_type":
                y_ct = torch.tensor(batch_data["class_y"]).to(device)
                loss_class = criterion_class(soft_out, y_ct)

                y_pred = torch.argmax(soft_out, dim=1)
                y_ct_np = y_ct.detach().clone().cpu().numpy()
                y_pred_np = y_pred.detach().clone().cpu().numpy()
                roc_ct = roc_auc_score(y_ct_np, y_pred_np)
                if roc_ct > best_auroc:
                    best_auroc = roc_ct

                rpc_ct = average_precision_score(y_ct_np, y_pred_np)
                avr_f1 = f1_score(y_ct_np, y_pred_np, average="macro")
                class_results_roc[f"auroc_{args.ct_class}"] = roc_ct
                class_results_rpc[f"auprc_{args.ct_class}"] = rpc_ct
                class_results_roc[f"best_auroc_{args.ct_class}"] = best_auroc
                class_results_roc[f"f1_score_{args.ct_class}"] = avr_f1

                loss_cce_dict[f"loss_cell_type_{args.ct_class}"] = loss_class.detach().clone().cpu().numpy()

                total_auroc_epoch_class += roc_ct
                total_aproc_epoch_class += rpc_ct
                total_f1_epoch_class += avr_f1
            else:
                y_dis_np = np.array(batch_data["class_y"])
                valid_indices = (~np.isnan(y_dis_np)) & (y_dis_np != "nan")

                y_dis = torch.tensor(y_dis_np[valid_indices].astype(float)).to(device)
                soft_out_filtered = soft_out[valid_indices].to(torch.float32)
                y_dis = y_dis.to(torch.int64)
                loss_class = criterion_class(soft_out_filtered, y_dis)

                y_pred = torch.argmax(soft_out_filtered, dim=1)
                values_present = len(np.unique(y_dis.cpu().numpy()))
                values = args.nclass
                if values > 2:
                    y_dis_onehot = np.eye(values)[y_dis.detach().clone().cpu().numpy()]
                    roc_disease = roc_auc_score(y_dis_onehot, soft_out_filtered.detach().clone().cpu().numpy(), multi_class="ovr", average="micro")
                elif values == 2 and values_present == 2:
                    roc_disease = roc_auc_score(y_dis.detach().clone().cpu().numpy(), y_pred.detach().clone().cpu().numpy())
                else:
                    roc_disease = 0

                if roc_disease > best_auroc:
                    best_auroc = roc_disease

                f1_disease = f1_score(y_dis.detach().clone().cpu().numpy(), y_pred.detach().clone().cpu().numpy(), average="weighted")
                rpc_disease = 0  # placeholder

                class_results_roc[f"best_auroc_{args.class_task}"] = best_auroc
                class_results_roc[f"auroc_{args.class_task}"] = roc_disease
                class_results_rpc[f"f1_{args.class_task}"] = f1_disease
                loss_cce_dict[f"loss_{args.class_task}"] = loss_class.detach().clone().cpu().numpy()

                total_auroc_epoch_class += roc_disease
                total_aproc_epoch_class += rpc_disease
                total_f1_epoch_class += f1_disease

        num_val = [x for x in range(500, 4000 + 1, 500)] + [num_batches - 1]
        if args.class_task in ('tissue', 'disease'):
            num_val = [x for x in range(50, 4000 + 1, 100)] + [num_batches - 1]

        if args.val and CLASS and batch in num_val and not CCE:
            test(
                loader_eval=loader_eval,
                temperature=temperature,
                device=device,
                merged_contexts=merged_contexts,
                CLASS=CLASS,
                args=args,
                model=model,
                vocab=vocab,
                CCE=CCE,
                OUTPUT_DIR=OUTPUT_DIR,
                memory_tracker=memory_tracker,
                profiler_obj=profiler_obj,
            )
            model.train()

        if loss_cce_dict is None:
            continue

        if weight_dis is not None:
            loss = weight_dis * loss_cce_dict["DISEASE"] + (1 - weight_dis) * loss_cce_dict["CELLS"]
        else:
            loss = sum(loss_cce_dict.values())

        if args.use_wandb:
            wandb.log(class_results_roc)
            wandb.log(class_results_rpc)

        log_dict = {}
        for key, value in loss_cce_dict.items():
            if isinstance(value, torch.Tensor):
                log_dict[key] = value.detach().clone().cpu() if value.is_cuda else value.detach().clone()
            else:
                log_dict[key] = value
        if args.use_wandb:
            wandb.log(log_dict)

        if args.DA:
            if args.use_wandb:
                wandb.log({'loss_da': loss_cce_dict["DA"]})

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                print(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )

        model.zero_grad()

        if profiler_obj is not None:
            with record_function("backward_pass"):
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"NaN/Inf detected in gradients of {name}")
                    exit()

        if profiler_obj is not None:
            with record_function("optimizer_step"):
                scaler.step(optimizer)
                scaler.update()
        else:
            scaler.step(optimizer)
            scaler.update()

        if profiler_obj is not None:
            profiler_obj.step()

        total_loss += loss.detach().clone().cpu().item()
        total_loss_epoch += loss.detach().clone().cpu().item()

        global_step += 1
        global_step_all = int(global_step) * int(world_size) if int(world_size) > 1 else int(global_step)

        if rank == 0 and (not getattr(args, "no_save_checkpoints", False)) and global_step_all % args.checkpoint_frequency == 0:
            ckpt_dir = OUTPUT_DIR / "CHECKPOINTS"
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = (
                f"ckpt_step_{global_step_all}_epoch_{epoch}_class_{int(bool(CLASS))}_"
                f"bs_{args.batch}_t_{args.temperature}_seed_{args.seed}_{timestamp}.pt"
            )
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            state = {
                "epoch": epoch,
                "global_step": global_step_all,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": vars(args),
            }
            path = save_checkpoint(state, ckpt_dir, filename)
            logger.info(f"Saved checkpoint at global step {global_step_all}: {path}")

        if dist.is_available() and dist.is_initialized():
            # Barrier here (after saving the checkpoint) ensures all ranks iterate at the
            # same rate; not the most efficient, but keeps them in sync.
            dist.barrier()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval

            local_batch_idx = batch + 1
            try:
                num_batches = len(loader)
            except Exception:
                pass

            epoch_fraction_local = float(local_batch_idx) / float(num_batches) if num_batches > 0 else 0.0
            total_batches_global = num_batches * int(world_size)
            estimated_global_batches_processed = local_batch_idx * int(world_size)
            epoch_fraction_global = float(estimated_global_batches_processed) / float(total_batches_global) if total_batches_global > 0 else epoch_fraction_local
            estimated_global_step = int(global_step) * int(world_size) + local_batch_idx

            logger.info(
                f"| epoch_frac {epoch_fraction_global:.4f} (local {epoch_fraction_local:.4f}) | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.5f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | global_step_est={estimated_global_step}"
            )

            if memory_tracker:
                current_memory = memory_tracker.get_current_memory()
                if current_memory:
                    logger.info(
                        f"Memory at log interval: "
                        f"RSS={current_memory['rss_mb']:.1f}MB, "
                        f"VMS={current_memory['vms_mb']:.1f}MB, "
                        f"CPU={current_memory['cpu_percent']:.1f}%"
                    )

            total_loss = 0
            start_time = time.time()

        if args.speed_bench and batch > 50:
            global START_TIME
            logger.info(f"Speed benchmark complete. Total execution time: {time.time() - START_TIME:.2f} seconds")
            exit()

    if CLASS:
        curr_epoch_auroc = total_auroc_epoch_class / len(loader)
        curr_epoch_aproc = total_aproc_epoch_class / len(loader)
        curr_epoch_f1 = total_f1_epoch_class / len(loader)
        if args.use_wandb:
            wandb.log({
                f"epoch_auroc_{args.ct_class}": curr_epoch_auroc,
                f"epoch_aproc_{args.ct_class}": curr_epoch_aproc,
                f"epoch_f1_{args.ct_class}": curr_epoch_f1,
            })

    current_disease_loss = total_loss_epoch_disease / len(loader) if "DISEASE" in loss_cce_dict.keys() else 0
    current_cell_loss = total_loss_epoch_cell_type / len(loader) if "CELLS" in loss_cce_dict.keys() else 0
    current_tissue_loss = total_loss_epoch_tissue / len(loader) if "TISSUE" in loss_cce_dict.keys() else 0
    if args.DA:
        current_epoch_da = total_loss_epoch_da / len(loader)

    logger.info(
        f"Training epoch {epoch}, Training Loss overall: {total_loss_epoch/len(loader)}, "
        f"Training loss disease: {current_disease_loss}, Training loss cell type: {current_cell_loss}, "
        f"Training loss tissue: {current_tissue_loss}"
    )

    metrics_to_log_epochs = {
        "loss_cce_epoch": total_loss_epoch / len(loader),
        "loss_disease_epoch": current_disease_loss,
        "loss_tissue_epoch": current_tissue_loss,
        "loss_cell_type_epoch": current_cell_loss,
    }

    if memory_tracker:
        current_memory = memory_tracker.get_current_memory()
        if current_memory:
            logger.info(
                f"End of epoch {epoch} memory: "
                f"RSS={current_memory['rss_mb']:.1f}MB, "
                f"VMS={current_memory['vms_mb']:.1f}MB, "
                f"CPU={current_memory['cpu_percent']:.1f}%"
            )
            if args.use_wandb and wandb.run is not None:
                wandb.log({
                    f'epoch_{epoch}_memory_rss_mb': current_memory['rss_mb'],
                    f'epoch_{epoch}_memory_vms_mb': current_memory['vms_mb'],
                    f'epoch_{epoch}_cpu_percent': current_memory['cpu_percent'],
                })

    if args.use_wandb:
        try:
            metrics_to_log_epochs.update({
                'epoch_fraction_local': epoch_fraction_local,
                'epoch': epoch_fraction_global,
                'estimated_global_step': estimated_global_step,
            })
        except Exception:
            pass

        wandb.log(metrics_to_log_epochs)
        if args.DA:
            wandb.log({'loss_DA_epoch': current_epoch_da})

    return global_step


@torch.no_grad()
def test(loader_eval, temperature, device, merged_contexts, CLASS, args, model, vocab, CCE, OUTPUT_DIR, memory_tracker=None, profiler_obj=None):
    model.eval()

    total_auroc_epoch_class_val = 0
    total_aproc_epoch_class_val = 0
    total_f1_epoch_class_val = 0
    class_results_roc_eval = {}
    class_results_rpc_eval = {}

    if memory_tracker:
        current_memory = memory_tracker.get_current_memory()
        if current_memory:
            print(f"Validation start memory: RSS={current_memory['rss_mb']:.1f}MB, VMS={current_memory['vms_mb']:.1f}MB, CPU={current_memory['cpu_percent']:.1f}%")

    for batch_eval, batch_data_eval in enumerate(loader_eval):
        input_gene_ids_eval = torch.tensor(batch_data_eval["input_ids"]).to(device)
        src_key_padding_mask_eval = input_gene_ids_eval.eq(vocab[PAD_TOKEN])

        forward_kwargs = dict(
            data=batch_data_eval,
            src=input_gene_ids_eval,
            src_key_padding_mask=src_key_padding_mask_eval,
            CCE=CCE,
            temperature=temperature,
            device=device,
            CLASS=CLASS,
            nclass=args.nclass,
        )
        if profiler_obj is not None:
            with record_function("model_eval"):
                out_model = model(**forward_kwargs)
        else:
            out_model = model(**forward_kwargs)

        if CLASS:
            if args.class_task == "cell_type":
                y_ct = torch.tensor(batch_data_eval["class_y"]).to(device)
                y_pred = torch.argmax(out_model, dim=1)

                y_ct_np = y_ct.detach().clone().cpu().numpy()
                y_pred_np = y_pred.detach().clone().cpu().numpy()

                roc_ct = roc_auc_score(y_ct_np, y_pred_np)
                rpc_ct = average_precision_score(y_ct_np, y_pred_np)
                avr_f1 = f1_score(y_ct_np, y_pred_np, average="macro")
                class_results_roc_eval[f"auroc_val_{args.ct_class}"] = roc_ct
                class_results_rpc_eval[f"auprc_val_{args.ct_class}"] = rpc_ct
                class_results_rpc_eval[f"f1_val_{args.ct_class}"] = avr_f1
                total_auroc_epoch_class_val += roc_ct
                total_aproc_epoch_class_val += rpc_ct
                total_f1_epoch_class_val += avr_f1

                if args.use_wandb:
                    wandb.log(class_results_roc_eval)
                    wandb.log(class_results_rpc_eval)
            else:
                y_dis_np = np.array(batch_data_eval["class_y"])
                valid_indices = (~np.isnan(y_dis_np)) & (y_dis_np != "nan")

                y_ct = torch.tensor(y_dis_np[valid_indices].astype(float)).to(device)
                y_ct = y_ct.to(torch.int64)
                soft_out_filtered = out_model[valid_indices].to(torch.float32)
                y_pred = torch.argmax(soft_out_filtered, dim=1)
                values_present = len(np.unique(y_ct.cpu().numpy()))
                values = args.nclass
                if values > 2:
                    y_ct_onehot = np.eye(values)[y_ct.cpu().numpy()]
                    roc_disease = roc_auc_score(y_ct_onehot, soft_out_filtered.detach().cpu().numpy(), multi_class="ovr", average="micro")
                    f1_disease = f1_score(y_ct.cpu().numpy(), y_pred.cpu().numpy(), average="weighted")
                elif values == 2 and values_present == 2:
                    roc_disease = roc_auc_score(y_ct.cpu().numpy(), y_pred.cpu().numpy())
                    f1_disease = f1_score(y_ct.cpu().numpy(), y_pred.cpu().numpy(), average="weighted")
                else:
                    roc_disease = 0
                    f1_disease = f1_score(y_ct.cpu().numpy(), y_pred.cpu().numpy(), average="weighted")

                class_results_roc_eval[f"auroc_val_{args.class_task}"] = roc_disease
                class_results_roc_eval[f"f1_val_{args.class_task}"] = f1_disease

                total_auroc_epoch_class_val += roc_disease
                total_f1_epoch_class_val += f1_disease
                if args.use_wandb:
                    wandb.log(class_results_roc_eval)

    if CLASS:
        curr_epoch_auroc = total_auroc_epoch_class_val / len(loader_eval)
        curr_epoch_f1 = total_f1_epoch_class_val / len(loader_eval)
        if args.use_wandb:
            wandb.log({
                f"avg_auroc_val_{args.class_task}_{args.ct_class}": curr_epoch_auroc,
                f"avg_f1_val_{args.class_task}_{args.ct_class}": curr_epoch_f1,
            })
            if args.class_task == "cell_type":
                curr_epoch_aproc = total_aproc_epoch_class_val / len(loader_eval)
                wandb.log({f"avg_aproc_val_{args.ct_class}": curr_epoch_aproc})

    if memory_tracker:
        current_memory = memory_tracker.get_current_memory()
        if current_memory:
            print(f"Validation end memory: RSS={current_memory['rss_mb']:.1f}MB, VMS={current_memory['vms_mb']:.1f}MB, CPU={current_memory['cpu_percent']:.1f}%")
