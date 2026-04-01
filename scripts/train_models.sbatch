#!/usr/bin/bash
#SBATCH --job-name=bp-train
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Model Training with ablation config support.
# Usage: sbatch --export=CONFIG=ppg scripts/train_models.sbatch
#        sbatch --export=CONFIG=ppg_ecg,MODELS=rf scripts/train_models.sbatch

set -uo pipefail

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

mkdir -p logs

CONFIG="${CONFIG:-ppg}"
MODELS="${MODELS:-ridge,dt,rf,xgb,lgbm}"
FEATURES="${FEATURES:-data/features/$CONFIG/features_labeled.csv}"
SCRATCH_DIR="/scratch/kamarthi.v/bp-checkpoints"

echo "=== Model Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG | Models: $MODELS | Features: $FEATURES"
echo "Start: $(date)"

python scripts/train_models.py \
    --config "$CONFIG" \
    --models "$MODELS" \
    --features "$FEATURES" \
    --checkpoint-dir "$SCRATCH_DIR" \
    --results-dir results \
    --resume

echo "=== Training Complete ==="
echo "End: $(date)"

python scripts/evaluate.py \
    --features "$FEATURES" \
    --checkpoint-dir "$SCRATCH_DIR/$CONFIG" \
    --results-dir "results/$CONFIG"

echo "=== Evaluation Complete ==="
echo "End: $(date)"
