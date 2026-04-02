#!/usr/bin/bash
#SBATCH --job-name=bp-dl-train
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Train DL models (ResNet-BiGRU, ResNet-1D) on raw PPG for BP estimation.
# Usage: sbatch --export=TARGET=sbp,MODEL=resnet_bigru scripts/train_dl_models.sbatch
# Submit all DL jobs:
#   for model in resnet_bigru resnet; do
#     for target in sbp dbp; do
#       sbatch --export=TARGET=$target,MODEL=$model scripts/train_dl_models.sbatch
#     done
#   done

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

# ─── Dependency check: install if missing, die if install fails ───
check_dep() {
    python -c "import $1" 2>/dev/null || {
        echo "Installing missing dependency: $2"
        pip install "$2" || { echo "FATAL: Failed to install $2"; exit 1; }
    }
}

check_dep torch "torch"
check_dep captum "captum"

# Verify CUDA is available (critical for DL training performance)
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_OK" != "True" ]; then
    echo "WARNING: CUDA not available. Training will run on CPU (10x slower)."
    echo "To fix: pip install torch --index-url https://download.pytorch.org/whl/cu121"
fi

# DL trains on PPG only -- no config ablation
TARGET="${TARGET:-sbp}"
MODEL="${MODEL:-resnet_bigru}"
TEST_SETS="${TEST_SETS:-CalFree,CalBased,AAMI}"

echo "=== DL Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL | Target: $TARGET | Test Sets: $TEST_SETS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA available: $CUDA_OK"
echo "Start: $(date)"

SCRATCH="/scratch/kamarthi.v/bp-checkpoints"
mkdir -p "$SCRATCH"

python scripts/train_dl_models.py \
    --target "$TARGET" \
    --model "$MODEL" \
    --subset-dir data/subsets \
    --checkpoint-dir "$SCRATCH" \
    --results-dir results \
    --test-sets "$TEST_SETS" \
    --resume \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --patience 20

echo ""
echo "End: $(date)"
