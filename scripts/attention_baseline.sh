#!/bin/bash
#SBATCH -J attn_baseline
#SBATCH --account=<slurm_account> -p <slurm_partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 1-00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

# Fill in for your cluster's module/environment setup.
# module load <cuda/conda modules>
# conda activate <env>

base="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$base"

export CASCADE_DATA_ROOT="${CASCADE_DATA_ROOT:-/path/to/DATASET}"
export CASCADE_CKPT_ROOT="${CASCADE_CKPT_ROOT:-/path/to/CASCADE-checkpoints}"

DATASET="SEATTLE"
TASK="cell_type"
CHECKPOINT="${CASCADE_CKPT_ROOT}/SEATTLE/ckpt.pt"
OUTPUT_DIR="${CASCADE_CKPT_ROOT}/attention_baseline_results"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Attention-derived gene-importance baseline"
echo "=============================================="
echo "Dataset:     $DATASET"
echo "Task:        $TASK"
echo "Checkpoint:  $CHECKPOINT"
echo "Output:      $OUTPUT_DIR"
echo "=============================================="

python -m analysis.alzheimers.attention_baseline \
    --output_path "${OUTPUT_DIR}/attention_baseline_${DATASET}_${TASK}.pkl" \
    --dataset_name "$DATASET" \
    --task "$TASK" \
    --transformer_checkpoint "$CHECKPOINT" \
    --batch_size 128

echo ""
echo "=== ATTENTION BASELINE COMPLETE ==="
