#!/bin/bash
#SBATCH -J extract_emb_parallel
#SBATCH --account=<slurm_account> -p <slurm_partition>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G
#SBATCH -t 1-00:00
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
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export GLOO_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_FAMILY=AF_INET

echo "=== MULTI-NODE SETUP ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"
python --version
nvidia-smi || true
echo "=========================="

# Example: extract HLCA embeddings from a trained checkpoint. Adjust dataset,
# checkpoint, and output path per run.
srun torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc-per-node=4 \
  --max-restarts=0 \
  --rdzv-id=789 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$head_node_ib:$head_node_port \
  -m cascade.explainer.get_embeddings_parallel \
    --dataset HLCA \
    --checkpoint "${CASCADE_DATA_ROOT}/HLCA/ckpt.pt" \
    --output-path "${CASCADE_DATA_ROOT}/HLCA/embeddings/embeddings.pkl" \
    --batch 2048 \
    --chunk-size 10

echo ""
echo "=== EMBEDDING EXTRACTION COMPLETE ==="
echo "Chunk files saved with pattern: *_rank_*_chunk_*.pkl"
