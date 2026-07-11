#!/bin/bash
#SBATCH -J donor_level_class
#SBATCH --account=<slurm_account> -p <slurm_partition>
#SBATCH --gres=gpu:3
#SBATCH --mem=1400G
#SBATCH -t 0-12:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

# Fill in for your cluster's module/environment setup.
# module load <cuda/conda modules>
# conda activate <env>

base="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$base"

export CASCADE_DATA_ROOT="${CASCADE_DATA_ROOT:-/path/to/DATASET}"
export CASCADE_CKPT_ROOT="${CASCADE_CKPT_ROOT:-/path/to/CASCADE-checkpoints}"

echo "Donor-level prediction (SEATTLE) with chunked embeddings..."
echo "Start time: $(date)"

OUTPUT_DIR="${CASCADE_CKPT_ROOT}/SEATTLE/collator_humans/models"
CHUNK_DIR="${CASCADE_CKPT_ROOT}/SEATTLE/collator_humans"
CHUNK_PREFIX="embeddings_parallel_rank"

echo "Output directory: $OUTPUT_DIR"
echo "Chunk dir: $CHUNK_DIR"

python -m cascade.explainer.main_donor_level \
  --dataset SEATTLE \
  --chunk-dir "$CHUNK_DIR" \
  --chunk-prefix "$CHUNK_PREFIX" \
  --save-model-dir "$OUTPUT_DIR"

echo ""
echo "Donor-level run completed!"
echo "End time: $(date)"
