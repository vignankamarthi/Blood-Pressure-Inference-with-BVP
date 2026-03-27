#!/usr/bin/bash
#SBATCH --job-name=bp-inspect-info
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Inspect Info .mat file structure to debug subset generation parsing.

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

export DATA_DIR="$PROJECT_DIR/data/raw"

echo "=== Info File Inspection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

python scripts/inspect_info_helper.py

echo ""
echo "End: $(date)"
