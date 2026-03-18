#!/bin/bash
#SBATCH --job-name=bp-tune
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=56
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Blood Pressure Inference -- Hyperparameter Tuning
# Optuna with SQLite persistence. Auto-resumes on resubmission.

set -euo pipefail

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

mkdir -p logs

echo "=== Hyperparameter Tuning ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

# Symlink Optuna DBs to scratch for performance
SCRATCH_DIR="/scratch/kamarthi.v/bp-checkpoints"
mkdir -p "$SCRATCH_DIR/optuna"

python scripts/train_models.py \
    --features data/features/features.csv \
    --checkpoint-dir "$SCRATCH_DIR" \
    --results-dir results \
    --tune \
    --resume \
    --n-trials 100

echo "=== Tuning Complete ==="
echo "End: $(date)"
