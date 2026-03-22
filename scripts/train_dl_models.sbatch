#!/usr/bin/bash
#SBATCH --job-name=bp-dl-train
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Train deep learning models for BP estimation.
# Usage: sbatch --export=CONFIG=ppg,TARGET=sbp,MODEL=cnn scripts/train_dl_models.sbatch
# Or submit all Phase 1 DL jobs:
#   for model in cnn lstm cnn_lstm transformer; do
#     for target in sbp dbp; do
#       sbatch --export=CONFIG=ppg,TARGET=$target,MODEL=$model scripts/train_dl_models.sbatch
#     done
#   done

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

# Default values if not set via --export
CONFIG="${CONFIG:-ppg}"
TARGET="${TARGET:-sbp}"
MODEL="${MODEL:-cnn}"

echo "=== DL Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG | Model: $MODEL | Target: $TARGET"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

SCRATCH="/scratch/kamarthi.v/bp-checkpoints"
mkdir -p "$SCRATCH"

python scripts/train_dl_models.py \
    --config "$CONFIG" \
    --target "$TARGET" \
    --model "$MODEL" \
    --subset-dir data/subsets \
    --checkpoint-dir "$SCRATCH" \
    --results-dir results \
    --resume \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --patience 20

echo ""
echo "End: $(date)"
