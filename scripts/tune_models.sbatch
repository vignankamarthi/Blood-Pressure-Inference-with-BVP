#!/usr/bin/bash
#SBATCH --job-name=bp-tune
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=56
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Hyperparameter tuning with ablation config support.
# Usage: sbatch --export=CONFIG=ppg scripts/tune_models.sbatch

set -uo pipefail

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

mkdir -p logs

CONFIG="${CONFIG:-ppg}"
MODELS="${MODELS:-ridge,dt,rf,xgb,lgbm}"
SCRATCH_DIR="/scratch/kamarthi.v/bp-checkpoints"

echo "=== Hyperparameter Tuning ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG | Models: $MODELS"
echo "Start: $(date)"

python scripts/train_models.py \
    --config "$CONFIG" \
    --models "$MODELS" \
    --checkpoint-dir "$SCRATCH_DIR" \
    --results-dir results \
    --tune \
    --resume \
    --n-trials 100

echo "=== Tuning Complete ==="
echo "End: $(date)"
