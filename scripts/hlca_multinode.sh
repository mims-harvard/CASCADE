#!/bin/bash
#SBATCH -J hlca_train_2node
#SBATCH --account=<slurm_account> -p <slurm_partition>
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4                            # 4 GPUs per node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1                     # 1 task per node for torchrun
#SBATCH -t 3-00:00
#SBATCH --mem=1024G
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.out

# Fill in for your cluster's module/environment setup.
# module load <cuda/conda modules>
# conda activate <env>

# Repo root (this script lives in scripts/, so the repo root is one level up)
base="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$base"

# Root directories for tokenized datasets and checkpoints (see cascade/training/train_ddp.py)
export CASCADE_DATA_ROOT="${CASCADE_DATA_ROOT:-/path/to/DATASET}"
export CASCADE_CKPT_ROOT="${CASCADE_CKPT_ROOT:-/path/to/CASCADE-checkpoints}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ib=( $( host $HOSTNAME | awk '{print $NF}' ) )
head_node_port=( $( comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1 ) )

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^docker0,lo  # Exclude problematic interfaces

# Force IPv4 (some clusters don't support IPv6)
export GLOO_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_FAMILY=AF_INET

echo "=== MULTI-NODE SETUP ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "Head node: $head_node"
echo "Head node IB (IP): $head_node_ib"
python --version
nvidia-smi || true
echo "=========================="

hlca_disease="${CASCADE_DATA_ROOT}/HLCA/DISEASE/DISEASE-ID.dataset"

srun torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --max_restarts=0 \
  --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d \
  --rdzv-id=456 \
  --rdzv_endpoint="${head_node_ib}:${head_node_port}" \
  -m cascade.training.train_ddp \
    --list_data "$hlca_disease" \
    --batch 32 \
    --nlayers 12 \
    --cell_emb_style avg-pool \
    --context_specific_projections \
    --donors \
    --DA \
    --dataset 'HLCA' \
    --nepochs 50 \
    --checkpoint_frequency 2000 \
    --wandb_run_name "hlca_train_2node_disease" \
    --wandb_run_group "hlca_multinode_experiments" \
    --wandb_tags experiment baseline hlca multinode 2nodes disease
