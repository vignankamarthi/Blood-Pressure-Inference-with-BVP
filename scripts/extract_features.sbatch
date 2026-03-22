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

# Feature Extraction (Rust) -- extracts 40 features from each signal type.
# Runs 3 times: PPG, ECG, ABP. Each has separate output dir + checkpoint.
# Auto-resumes: re-submit safely after wall-time kill.

set -uo pipefail

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"

mkdir -p logs

RUST_BIN="$PROJECT_DIR/Feature-Extraction-Rust-Complete/target/release/feature_extraction_rust_complete"

if [ ! -f "$RUST_BIN" ]; then
    echo "ERROR: Rust binary not found. Run: sbatch scripts/build_rust.sbatch"
    exit 1
fi

echo "=== Feature Extraction (3 signals) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

for SIGNAL in ppg ecg abp; do
    INPUT_DIR="data/signals/$SIGNAL"
    OUTPUT_DIR="data/features/$SIGNAL"

    if [ ! -d "$INPUT_DIR" ]; then
        echo "SKIP $SIGNAL: $INPUT_DIR not found"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    CHECKPOINT="$OUTPUT_DIR/checkpoint.json"
    RESUME_FLAG=""
    if [ -f "$CHECKPOINT" ]; then
        echo "[$SIGNAL] Checkpoint found -- resuming..."
        RESUME_FLAG="--resume"
    else
        echo "[$SIGNAL] Starting fresh..."
    fi

    echo "[$SIGNAL] Extracting features from $INPUT_DIR -> $OUTPUT_DIR"
    echo "[$SIGNAL] Start: $(date)"

    $RUST_BIN extract \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --frameworks catch22,entropy,stats \
        --workers "$SLURM_CPUS_PER_TASK" \
        --dimensions 7 \
        --delays 2 \
        $RESUME_FLAG \
        -v 1

    echo "[$SIGNAL] Done: $(date)"
    echo ""
done

echo "=== Feature Extraction Complete ==="
echo "End: $(date)"
echo "Outputs:"
for SIGNAL in ppg ecg abp; do
    if [ -f "data/features/$SIGNAL/features.csv" ]; then
        ROWS=$(wc -l < "data/features/$SIGNAL/features.csv")
        echo "  $SIGNAL: $ROWS rows"
    fi
done
