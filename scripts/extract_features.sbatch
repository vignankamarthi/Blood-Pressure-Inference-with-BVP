#!/usr/bin/bash
#SBATCH --job-name=bp-extract
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=56
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Blood Pressure Inference -- Feature Extraction (Rust)
# Extracts Catch22 + entropy + statistical features from PulseDB PPG segments.
# Checkpoint-aware: auto-resumes if checkpoint.json exists.

set -euo pipefail

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"

mkdir -p logs

echo "=== Feature Extraction ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

RUST_BIN="$PROJECT_DIR/Feature-Extraction-Rust-Complete/target/release/feature_extraction_rust_complete"

# Auto-detect resume
CHECKPOINT="data/features/checkpoint.json"
if [ -f "$CHECKPOINT" ]; then
    echo "Checkpoint found -- resuming..."
    RESUME_FLAG="--resume"
else
    echo "No checkpoint -- starting fresh..."
    RESUME_FLAG=""
fi

$RUST_BIN extract \
    --input-dir data/raw \
    --output-dir data/features \
    --frameworks catch22,entropy,stats \
    --workers "$SLURM_CPUS_PER_TASK" \
    --dimensions 7 \
    --delays 2 \
    $RESUME_FLAG \
    -v 1

echo "=== Feature Extraction Complete ==="
echo "End: $(date)"
echo "Output: data/features/features.csv"
