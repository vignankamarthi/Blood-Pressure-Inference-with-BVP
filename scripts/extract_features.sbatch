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
# Iterates 3 signals x 5 subsets = 15 Rust runs.
# Each (signal, subset) pair has its own checkpoint -- safe to resubmit after wall-time kill.

set -uo pipefail

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"

mkdir -p logs

RUST_BIN="$PROJECT_DIR/Feature-Extraction-Rust-Complete/target/release/feature_extraction_rust_complete"

if [ ! -f "$RUST_BIN" ]; then
    echo "ERROR: Rust binary not found. Run: sbatch scripts/build_rust.sbatch"
    exit 1
fi

echo "=== Feature Extraction (3 signals x 5 subsets) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

for SIGNAL in ppg ecg abp; do
    SIGNAL_DIR="data/signals/$SIGNAL"

    if [ ! -d "$SIGNAL_DIR" ]; then
        echo "SKIP $SIGNAL: $SIGNAL_DIR not found"
        continue
    fi

    for SUBSET_DIR in "$SIGNAL_DIR"/*/; do
        [ -d "$SUBSET_DIR" ] || continue

        SUBSET_NAME=$(basename "$SUBSET_DIR")
        OUTPUT_DIR="data/features/$SIGNAL/$SUBSET_NAME"

        mkdir -p "$OUTPUT_DIR"

        # Check if already complete (features.csv exists with data rows)
        if [ -f "$OUTPUT_DIR/features.csv" ]; then
            ROWS=$(wc -l < "$OUTPUT_DIR/features.csv")
            if [ "$ROWS" -gt 1 ]; then
                echo "[$SIGNAL/$SUBSET_NAME] Already complete ($ROWS rows) -- skipping"
                continue
            fi
        fi

        CHECKPOINT="$OUTPUT_DIR/checkpoint.json"
        RESUME_FLAG=""
        if [ -f "$CHECKPOINT" ]; then
            echo "[$SIGNAL/$SUBSET_NAME] Checkpoint found -- resuming..."
            RESUME_FLAG="--resume"
        else
            echo "[$SIGNAL/$SUBSET_NAME] Starting fresh..."
        fi

        echo "[$SIGNAL/$SUBSET_NAME] Extracting features: $SUBSET_DIR -> $OUTPUT_DIR"
        echo "[$SIGNAL/$SUBSET_NAME] Start: $(date)"

        $RUST_BIN extract \
            --input-dir "$SUBSET_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --frameworks catch22,entropy,stats \
            --workers "$SLURM_CPUS_PER_TASK" \
            --dimensions 7 \
            --delays 2 \
            $RESUME_FLAG \
            -v 1

        echo "[$SIGNAL/$SUBSET_NAME] Done: $(date)"
        echo ""
    done
done

echo "=== Feature Extraction Complete ==="
echo "End: $(date)"
echo ""
echo "Outputs:"
for SIGNAL in ppg ecg abp; do
    for SUBSET_DIR in data/signals/$SIGNAL/*/; do
        [ -d "$SUBSET_DIR" ] || continue
        SUBSET_NAME=$(basename "$SUBSET_DIR")
        FEATURE_FILE="data/features/$SIGNAL/$SUBSET_NAME/features.csv"
        if [ -f "$FEATURE_FILE" ]; then
            ROWS=$(wc -l < "$FEATURE_FILE")
            echo "  $SIGNAL/$SUBSET_NAME: $ROWS rows"
        else
            echo "  $SIGNAL/$SUBSET_NAME: MISSING"
        fi
    done
done
