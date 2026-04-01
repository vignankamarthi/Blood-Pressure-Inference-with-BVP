#!/usr/bin/bash
#SBATCH --job-name=bp-export-csv
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Export .npz subset signals to per-signal CSV directories for Rust extraction.

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

echo "=== Export Signals to CSV ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

python scripts/export_signals_csv.py \
    --subset-dir data/subsets \
    --output-dir data/signals

echo ""
echo "End: $(date)"
