#!/usr/bin/bash
# Phase 1: PPG-Only Championship
# Submits all 9 models (5 ML + 4 DL) for PPG-only on both SBP and DBP.
# Total: 18 trainings (10 ML in 1 job + 8 DL in 8 jobs)

set -uo pipefail

echo "=== Phase 1: PPG-Only Championship ==="
echo "Submitting ML training (5 models, ppg config)..."
sbatch scripts/tune_models.sbatch
sbatch scripts/train_models.sbatch

echo ""
echo "Submitting DL training (4 models x 2 targets, ppg config)..."
for MODEL in cnn lstm cnn_lstm transformer; do
    for TARGET in sbp dbp; do
        sbatch --export=CONFIG=ppg,TARGET=$TARGET,MODEL=$MODEL scripts/train_dl_models.sbatch
        echo "  Submitted: $MODEL / $TARGET"
    done
done

echo ""
echo "Phase 1: 9 jobs submitted (1 tune + 1 ML train + 8 DL train)"
echo "Monitor: squeue -u kamarthi.v"
echo "When done: cat results/ppg/leaderboard.json"
