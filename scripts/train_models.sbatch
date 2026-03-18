#!/bin/bash
#SBATCH --job-name=bp-train
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=56
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Blood Pressure Inference -- Model Training
# Trains all 5 models x 2 targets with exhaustive checkpointing.
# Auto-resumes from any checkpoint on resubmission.

set -euo pipefail

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

mkdir -p logs

echo "=== Model Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

SCRATCH_DIR="/scratch/kamarthi.v/bp-checkpoints"

python scripts/train_models.py \
    --features data/features/features.csv \
    --checkpoint-dir "$SCRATCH_DIR" \
    --results-dir results \
    --resume

echo "=== Training Complete ==="
echo "End: $(date)"

# Also run evaluation
python scripts/evaluate.py \
    --features data/features/features.csv \
    --checkpoint-dir "$SCRATCH_DIR" \
    --results-dir results

echo "=== Evaluation Complete ==="
echo "End: $(date)"
