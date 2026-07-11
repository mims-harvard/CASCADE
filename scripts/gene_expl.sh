#!/bin/bash
#SBATCH -J explain_gene_parallel
#SBATCH --account=<slurm_account> -p <slurm_partition>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=720G
#SBATCH -t 0-12:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

# Fill in for your cluster's module/environment setup.
# module load <cuda/conda modules>
# conda activate <env>

base="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$base"

export CASCADE_DATA_ROOT="${CASCADE_DATA_ROOT:-/path/to/DATASET}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ib=( $( host $head_node | awk '{print $NF}' ) )
head_node_port=( $( comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1 ) )

export MASTER_ADDR=$head_node_ib
export MASTER_PORT=$head_node_port

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export GLOO_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_FAMILY=AF_INET
export DIST_BACKEND=${DIST_BACKEND:-gloo}

echo "=== MULTI-NODE SETUP ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"
python --version
nvidia-smi || true
echo "=========================="

# Example: gene-level explainability for AUTISM. Requires cell-level models
# already trained via main_cell_level.py (see scripts/run_train_cells.sh).
srun torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=4 \
  --max_restarts=0 \
  --rdzv_backend=c10d \
  --rdzv-id=456 \
  --rdzv_endpoint="${head_node_ib}:${head_node_port}" \
  -m cascade.explainer.run_gene_explainer_parallel \
    --output_path "${CASCADE_DATA_ROOT}/AUTISM/explainer/importances.pkl" \
    --dataset_name AUTISM \
    --transformer_checkpoint "${CASCADE_DATA_ROOT}/AUTISM/ckpt.pt" \
    --models_dir "${CASCADE_DATA_ROOT}/AUTISM/trained_models" \
    --global_number_to_perturb 100

echo ""
echo "=== GENE EXPLAINABILITY COMPLETE ==="
