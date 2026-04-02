#!/usr/bin/bash
#SBATCH --job-name=bp-shap
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# GradientSHAP temporal interpretability analysis.
# Run AFTER DL training completes.
# Usage: sbatch --export=MODEL=resnet_bigru,TARGET=sbp scripts/run_gradient_shap.sbatch

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

# ─── Dependency check ───
check_dep() {
    python -c "import $1" 2>/dev/null || {
        echo "Installing missing dependency: $2"
        pip install "$2" || { echo "FATAL: Failed to install $2"; exit 1; }
    }
}

check_dep torch "torch"
check_dep captum "captum"

MODEL="${MODEL:-resnet_bigru}"
TARGET="${TARGET:-sbp}"

echo "=== GradientSHAP Analysis ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL | Target: $TARGET"
echo "Start: $(date)"

SCRATCH="/scratch/kamarthi.v/bp-checkpoints"

python scripts/run_gradient_shap.py \
    --model-name "$MODEL" \
    --target "$TARGET" \
    --checkpoint-dir "$SCRATCH" \
    --subset-dir data/subsets \
    --test-set CalFree \
    --n-samples 200 \
    --output-dir results/dl/gradient_shap

echo ""
echo "End: $(date)"
