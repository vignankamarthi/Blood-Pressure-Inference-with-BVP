#!/usr/bin/bash
# Phase 2: Ablation with Champions
# Usage: bash scripts/run_phase2.sh <best_ml_model> <best_dl_model>
# Example: bash scripts/run_phase2.sh rf transformer

set -uo pipefail

BEST_ML="${1:?Usage: bash scripts/run_phase2.sh <best_ml_model> <best_dl_model>}"
BEST_DL="${2:?Usage: bash scripts/run_phase2.sh <best_ml_model> <best_dl_model>}"

echo "=== Phase 2: Ablation with Champions ==="
echo "Best ML: $BEST_ML"
echo "Best DL: $BEST_DL"
echo ""

for CONFIG in ppg ppg_ecg; do
    echo "Config: $CONFIG"

    # ML champion
    sbatch --export=CONFIG=$CONFIG,MODELS=$BEST_ML scripts/tune_models.sbatch
    sbatch --export=CONFIG=$CONFIG,MODELS=$BEST_ML scripts/train_models.sbatch
    echo "  ML: $BEST_ML submitted"

    # DL champion
    for TARGET in sbp dbp; do
        sbatch --export=CONFIG=$CONFIG,TARGET=$TARGET,MODEL=$BEST_DL scripts/train_dl_models.sbatch
        echo "  DL: $BEST_DL / $TARGET submitted"
    done
    echo ""
done

echo "Phase 2: 8 jobs submitted (2 configs x (1 ML tune + 1 ML train + 2 DL train))"
echo "Monitor: squeue -u kamarthi.v"
