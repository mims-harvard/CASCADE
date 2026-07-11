#!/bin/bash
#SBATCH -J cell_level_class
#SBATCH --account=<slurm_account> -p <slurm_partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH -t 0-6:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

# Fill in for your cluster's module/environment setup.
# module load <cuda/conda modules>
# conda activate <env>

base="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$base"

export CASCADE_DATA_ROOT="${CASCADE_DATA_ROOT:-/path/to/DATASET}"
export CASCADE_CKPT_ROOT="${CASCADE_CKPT_ROOT:-/path/to/CASCADE-checkpoints}"

echo "Training cell-level classifiers on M2 dataset..."
echo "Start time: $(date)"

OUTPUT_DIR="${CASCADE_CKPT_ROOT}/M2/collator_humans/models"
echo "Output directory: $OUTPUT_DIR"

python -m cascade.explainer.main_cell_level \
  --dataset M2 \
  --chunk-dir "${CASCADE_CKPT_ROOT}/M2/collator_humans" \
  --chunk-prefix embeddings_parallel_rank \
  --save-model-dir "$OUTPUT_DIR" \
  --eval-only \
  --predict-full-data

# For a fresh training run instead of --eval-only:
# python -m cascade.explainer.main_cell_level \
#   --dataset M2 \
#   --chunk-dir "${CASCADE_CKPT_ROOT}/M2/collator_humans" \
#   --chunk-prefix embeddings_parallel_rank \
#   --save-model-dir "$OUTPUT_DIR"

echo ""
echo "Training completed!"
echo "End time: $(date)"
